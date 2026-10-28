"""Train the two-reference adapter (character slot + style slot) on prepare_dual.py output.

  python train/train_dual.py --prep runs/train/dual_prep --out runs/train/dual_r32 \
      --init-lora runs/train/pilot_r32/lora_step02000.safetensors

Each step draws one mode for the whole batch (frames must line up across the batch):
  split      character slot = same character, other style; style slot = same style, other character
  character  character slot only; the prompt names the artists with probability text_fallback
  style      style slot only; the prompt names the character with probability text_fallback
  same       both slots from the target's own group (one image in both, or two images)
The target is noised as in train.py; only the LoRA trains.
"""
from __future__ import annotations

import argparse
import json
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
from sample import Prep, run_eval_dual  # noqa: E402
from train import center_crop, lr_at  # noqa: E402

MODE_SLOTS = {"split": ("character", "style"), "character": ("character",), "style": ("style",),
              "same": ("character", "style")}


class TripletSampler:
    """Per step: a mode, then a target-size bucket of targets that support the mode, then refs."""

    def __init__(self, prep: Prep, dual_cfg: dict, batch_size: int, seed: int):
        self.rng = random.Random(seed)
        self.batch_size, self.cfg = batch_size, dual_cfg
        records, groups = prep.index["records"], prep.index["groups"]
        self.candidates = {}
        buckets = defaultdict(list)
        for gid, group in groups.items():
            if group["split"] != "train":
                continue
            char_refs = [r for g in group["partners"]["character"] for r in groups[g]["record_ids"]]
            style_refs = [r for g in group["partners"]["style"] for r in groups[g]["record_ids"]]
            for target in group["record_ids"]:
                same = [r for r in group["record_ids"] if r != target]
                self.candidates[target] = {"character": char_refs, "style": style_refs, "same": same}
                size = (records[target]["width"], records[target]["height"])
                for mode in MODE_SLOTS:
                    needs = {"split": char_refs and style_refs, "character": char_refs, "style": style_refs,
                             "same": same}[mode]
                    if needs:
                        buckets[(mode, size)].append(target)
        self.buckets = dict(buckets)
        self.queues = {key: [] for key in self.buckets}

    def counts(self) -> dict[str, int]:
        out = defaultdict(int)
        for (mode, _), targets in self.buckets.items():
            out[mode] += len(targets)
        return dict(out)

    def next(self):
        modes = [m for m in self.cfg["modes"] if any(k[0] == m for k in self.buckets)]
        mode = self.rng.choices(modes, weights=[self.cfg["modes"][m] for m in modes])[0]
        keys = [k for k in self.buckets if k[0] == mode]
        key = self.rng.choices(keys, weights=[len(self.buckets[k]) for k in keys])[0]
        queue, batch = self.queues[key], []
        while len(batch) < self.batch_size:
            if not queue:
                queue.extend(self.buckets[key])
                self.rng.shuffle(queue)
            target = queue.pop()
            cands = self.candidates[target]
            if mode == "same":
                first = self.rng.choice(cands["same"])
                both = len(cands["same"]) < 2 or self.rng.random() < self.cfg["same_image_both_slots"]
                second = first if both else self.rng.choice([r for r in cands["same"] if r != first])
                refs = {"character": first, "style": second}
            else:
                refs = {slot: self.rng.choice(cands[slot]) for slot in MODE_SLOTS[mode]}
            batch.append((target, refs))
        return mode, key[1], batch


def prompt_key(mode: str, rng: random.Random, mix: dict[str, float], text_fallback: float) -> str:
    if mode == "character" and rng.random() < text_fallback:
        return "with_artist"
    if mode == "style" and rng.random() < text_fallback:
        return "with_character"
    return rng.choices(list(mix), weights=list(mix.values()))[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="defaults to the config stored in the prep index")
    parser.add_argument("--comfy-root", default="/workspace/ComfyUI")
    parser.add_argument("--init-lora", type=Path, help="start from a single-reference adapter LoRA")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--checkpoint-blocks", type=int)
    parser.add_argument("--debug-crop", type=int, help="debug: center-crop latents to N x N (small-GPU smoke test)")
    parser.add_argument("--no-sample", action="store_true")
    args = parser.parse_args()

    prep = Prep(args.prep)
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else prep.index["config"]
    train_cfg, adapter_cfg = config["train"], config["adapter"]
    for key in ("steps", "batch_size", "checkpoint_blocks"):
        if getattr(args, key) is not None:
            train_cfg[key] = getattr(args, key)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")

    seed = train_cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    anima_env.setup_comfy(args.comfy_root)
    device = torch.device("cuda")
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)
    print(f"self-test ok: {anima_env.self_test(dm, device)}", flush=True)

    modules = anima_env.inject_lora(dm, adapter_cfg["lora_rank"], adapter_cfg["lora_alpha"])
    if args.init_lora:
        init_meta = anima_env.load_lora(modules, args.init_lora)
        print(f"initialised from {args.init_lora} (step {init_meta.get('step')})", flush=True)
    params = [p for m in modules.values() for p in (m.lora_down, m.lora_up)]
    optimizer = torch.optim.AdamW(params, lr=train_cfg["lr"], betas=(0.9, 0.999), weight_decay=train_cfg["weight_decay"])
    print(f"lora modules={len(modules)} params={sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)

    lf = anima_env.latent_format()

    def model_space(raw: torch.Tensor) -> torch.Tensor:
        return lf.process_in(raw[None].float())[0].to(torch.bfloat16)

    # Converted one at a time so the raw copies are freed as we go (the reference set is tens of GB).
    targets = {k: model_space(prep.targets.pop(k)) for k in list(prep.targets)}
    refs = {k: model_space(prep.refs.pop(k)) for k in list(prep.refs)}
    records = prep.index["records"]
    sampler = TripletSampler(prep, config["dual"], train_cfg["batch_size"], seed)
    print(f"targets per mode: {sampler.counts()}", flush=True)
    rng = random.Random(seed + 1)

    metadata = {"rank": adapter_cfg["lora_rank"], "alpha": adapter_cfg["lora_alpha"],
                "targets": list(anima_env.LORA_TARGETS), "slots": adapter_cfg["slots"], "ref_sigma": 0.0,
                "base": config["comfy"]["unet_name"], "version": 2,
                "init": str(args.init_lora) if args.init_lora else None}
    vae = None
    log_path = args.out / "train_log.jsonl"
    shift = train_cfg["shift"]
    running, mode_loss = [], defaultdict(list)
    started = window_start = time.perf_counter()
    sample_at = set(train_cfg.get("sample_at", [])) | {train_cfg["steps"]}

    for step in range(train_cfg["steps"]):
        mode, (width, height), batch = sampler.next()
        crop = args.debug_crop
        x0 = torch.stack([center_crop(targets[t], crop) for t, _ in batch]).to(device)
        slot_refs = {slot: torch.stack([center_crop(refs[f"{r[slot]}@{width}x{height}"], crop) for _, r in batch]).to(device)
                     for slot in MODE_SLOTS[mode]}
        contexts = [prep.context(records[t]["prompts"][prompt_key(mode, rng, config["data"]["prompt_mix"],
                                                                  config["dual"]["text_fallback"])])
                    for t, _ in batch]
        context = anima_env.pad_contexts(contexts).to(device, torch.bfloat16)

        u = torch.rand(len(batch), device=device)
        sigma = (shift * u / (1 + (shift - 1) * u)).clamp(1e-3, 1.0)
        noise = torch.randn_like(x0, dtype=torch.float32)
        s = sigma.view(-1, 1, 1, 1, 1)
        xt = ((1 - s) * x0.float() + s * noise).to(torch.bfloat16)
        target_v = noise - x0.float()
        jitter = [rng.randint(0, adapter_cfg["ref_offset_jitter"]) for _ in anima_env.SLOTS]
        frames = anima_env.slot_frames(slot_refs, x0.shape[-2], adapter_cfg, jitter)

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, train_cfg)
        pred = anima_env.ref_forward(dm, xt, sigma, context, refs=frames, checkpoint_blocks=train_cfg["checkpoint_blocks"])
        loss = torch.nn.functional.mse_loss(pred.float(), target_v)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, train_cfg["grad_clip"])
        optimizer.step()
        running.append(loss.item())
        mode_loss[mode].append(loss.item())

        done = step + 1
        if done % train_cfg["log_every"] == 0 or done == train_cfg["steps"]:
            now = time.perf_counter()
            line = {"step": done, "loss": round(sum(running) / len(running), 5),
                    "mode_loss": {m: round(sum(v) / len(v), 5) for m, v in mode_loss.items()},
                    "lr": optimizer.param_groups[0]["lr"], "grad_norm": round(float(grad_norm), 4),
                    "s_per_step": round((now - window_start) / len(running), 3),
                    "elapsed_h": round((now - started) / 3600, 3),
                    "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            line["eta_h"] = round(line["s_per_step"] * (train_cfg["steps"] - done) / 3600, 2)
            print(json.dumps(line), flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line) + "\n")
            running, mode_loss, window_start = [], defaultdict(list), time.perf_counter()
        if done % train_cfg["save_every"] == 0 or done == train_cfg["steps"]:
            anima_env.save_lora(modules, args.out / f"lora_step{done:05d}.safetensors", {**metadata, "step": done})
        if done in sample_at and not args.no_sample:
            if vae is None:
                vae = anima_env.load_vae(config["comfy"]["vae_name"])
            t0 = time.perf_counter()
            run_eval_dual(dm, modules, prep, vae, args.out / "samples" / f"step{done:05d}", config, device,
                          refs=refs, targets=targets)
            print(f"samples at step {done} took {time.perf_counter() - t0:.0f}s", flush=True)
            window_start = time.perf_counter()
    print(f"training done in {(time.perf_counter() - started) / 3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
