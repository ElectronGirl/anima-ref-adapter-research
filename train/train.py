"""Train the Anima reference adapter (reference frame with shifted RoPE + LoRA) on prepare.py output.

  python train/train.py --prep runs/train/pilot_prep --out runs/train/pilot_r32

Pairs are (target, reference) images of the same group; the target is noised (flow matching,
Anima's shift-3 schedule) and the model predicts its velocity while reading the clean reference
frame. Only LoRA weights train; the base model, text encoder and VAE stay frozen.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402
from sample import Prep, run_eval  # noqa: E402


class PairSampler:
    """Uniform over (target, reference) pairs; each batch shares one target resolution."""

    def __init__(self, pairs_by_size: dict[tuple[int, int], list[tuple[str, str]]], batch_size: int, seed: int):
        self.rng = random.Random(seed)
        self.batch_size = batch_size
        self.buckets = {size: list(pairs) for size, pairs in pairs_by_size.items() if pairs}
        self.queues = {size: [] for size in self.buckets}
        self.sizes = list(self.buckets)
        self.weights = [len(self.buckets[s]) for s in self.sizes]

    def next(self) -> tuple[tuple[int, int], list[tuple[str, str]]]:
        size = self.rng.choices(self.sizes, weights=self.weights)[0]
        queue = self.queues[size]
        batch = []
        while len(batch) < self.batch_size:
            if not queue:
                queue.extend(self.buckets[size])
                self.rng.shuffle(queue)
            batch.append(queue.pop())
        return size, batch


def lr_at(step: int, cfg: dict) -> float:
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    progress = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    floor = cfg["min_lr_ratio"]
    return cfg["lr"] * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def center_crop(latent: torch.Tensor, size: int | None) -> torch.Tensor:
    if not size:
        return latent
    h, w = latent.shape[-2:]
    top, left = max(0, (h - size) // 2), max(0, (w - size) // 2)
    return latent[..., top:top + size, left:left + size]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="defaults to the config stored in the prep index")
    parser.add_argument("--comfy-root", default="/workspace/ComfyUI")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--checkpoint-blocks", type=int)
    parser.add_argument("--debug-crop", type=int, help="debug: center-crop latents to N x N (small-GPU smoke test)")
    parser.add_argument("--no-sample", action="store_true")
    args = parser.parse_args()

    prep = Prep(args.prep)
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else prep.index["config"]
    train_cfg, adapter_cfg = config["train"], config["adapter"]
    if args.steps:
        train_cfg["steps"] = args.steps
    if args.batch_size:
        train_cfg["batch_size"] = args.batch_size
    if args.checkpoint_blocks is not None:
        train_cfg["checkpoint_blocks"] = args.checkpoint_blocks
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")

    seed = train_cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    anima_env.setup_comfy(args.comfy_root)
    device = torch.device("cuda")
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)
    check = anima_env.self_test(dm, device)
    print(f"self-test ok: {check}", flush=True)

    modules = anima_env.inject_lora(dm, adapter_cfg["lora_rank"], adapter_cfg["lora_alpha"])
    params = [p for m in modules.values() for p in (m.lora_down, m.lora_up)]
    optimizer = torch.optim.AdamW(params, lr=train_cfg["lr"], betas=(0.9, 0.999),
                                  weight_decay=train_cfg["weight_decay"])
    print(f"lora modules={len(modules)} params={sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)

    # Model-space latents in RAM (bf16).
    lf = anima_env.latent_format()

    def model_space(raw: torch.Tensor) -> torch.Tensor:
        return lf.process_in(raw[None].float())[0].to(torch.bfloat16)

    targets = {k: model_space(v) for k, v in prep.targets.items()}
    refs = {k: model_space(v) for k, v in prep.refs.items()}
    records = prep.index["records"]
    pairs_by_size = defaultdict(list)
    for group in prep.index["groups"].values():
        if group["split"] != "train":
            continue
        for target_id in group["record_ids"]:
            for ref_id in group["record_ids"]:
                if ref_id != target_id:
                    rec = records[target_id]
                    pairs_by_size[(rec["width"], rec["height"])].append((target_id, ref_id))
    total_pairs = sum(len(v) for v in pairs_by_size.values())
    print(f"train pairs={total_pairs} buckets={ {f'{w}x{h}': len(v) for (w, h), v in pairs_by_size.items()} }", flush=True)
    sampler = PairSampler(pairs_by_size, train_cfg["batch_size"], seed)
    mix_names = list(config["data"]["prompt_mix"])
    mix_weights = [config["data"]["prompt_mix"][k] for k in mix_names]
    rng = random.Random(seed + 1)

    metadata = {"rank": adapter_cfg["lora_rank"], "alpha": adapter_cfg["lora_alpha"],
                "targets": list(anima_env.LORA_TARGETS), "ref_offset_h": adapter_cfg["ref_offset_h"],
                "ref_offset_w": adapter_cfg["ref_offset_w"], "ref_sigma": 0.0, "base": config["comfy"]["unet_name"]}
    vae = None
    log_path = args.out / "train_log.jsonl"
    shift = train_cfg["shift"]
    running, started, window_start = [], time.perf_counter(), time.perf_counter()
    sample_at = set(train_cfg.get("sample_at", [])) | {train_cfg["steps"]}

    for step in range(train_cfg["steps"]):
        (width, height), batch = sampler.next()
        x0 = torch.stack([center_crop(targets[t], args.debug_crop) for t, _ in batch]).to(device)
        ref = torch.stack([center_crop(refs[f"{r}@{width}x{height}"], args.debug_crop) for _, r in batch]).to(device)
        contexts = []
        for target_id, _ in batch:
            variant = rng.choices(mix_names, weights=mix_weights)[0]
            contexts.append(prep.context(records[target_id]["prompts"][variant]))
        context = anima_env.pad_contexts(contexts).to(device, torch.bfloat16)

        u = torch.rand(len(batch), device=device)
        sigma = (shift * u / (1 + (shift - 1) * u)).clamp(1e-3, 1.0)
        noise = torch.randn_like(x0, dtype=torch.float32)
        s = sigma.view(-1, 1, 1, 1, 1)
        xt = ((1 - s) * x0.float() + s * noise).to(torch.bfloat16)
        target_v = noise - x0.float()
        hp = x0.shape[-2] // 2
        offset = (round(adapter_cfg["ref_offset_h"] * hp) + rng.randint(0, adapter_cfg["ref_offset_jitter"]),
                  int(adapter_cfg["ref_offset_w"]))

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, train_cfg)
        pred = anima_env.ref_forward(dm, xt, sigma, context, ref=ref, ref_offset=offset,
                                     checkpoint_blocks=train_cfg["checkpoint_blocks"])
        loss = torch.nn.functional.mse_loss(pred.float(), target_v)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, train_cfg["grad_clip"])
        optimizer.step()
        running.append(loss.item())

        done = step + 1
        if done % train_cfg["log_every"] == 0 or done == train_cfg["steps"]:
            now = time.perf_counter()
            line = {"step": done, "loss": round(sum(running) / len(running), 5), "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": round(float(grad_norm), 4), "s_per_step": round((now - window_start) / len(running), 3),
                    "elapsed_h": round((now - started) / 3600, 3),
                    "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            line["eta_h"] = round(line["s_per_step"] * (train_cfg["steps"] - done) / 3600, 2)
            print(json.dumps(line), flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line) + "\n")
            running, window_start = [], time.perf_counter()
        if done % train_cfg["save_every"] == 0 or done == train_cfg["steps"]:
            anima_env.save_lora(modules, args.out / f"lora_step{done:05d}.safetensors", {**metadata, "step": done})
        if done in sample_at and not args.no_sample:
            if vae is None:
                vae = anima_env.load_vae(config["comfy"]["vae_name"])
            t0 = time.perf_counter()
            run_eval(dm, modules, prep, vae, args.out / "samples" / f"step{done:05d}", config, device,
                     baseline_dir=args.out / "samples" / "baseline")
            print(f"samples at step {done} took {time.perf_counter() - t0:.0f}s", flush=True)
            window_start = time.perf_counter()
    print(f"training done in {(time.perf_counter() - started) / 3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
