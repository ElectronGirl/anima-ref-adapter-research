"""Train the v2 character adapter: one character reference plus a style_transfer strength s.

A learned vector times s is added to every reference token (anima_env.ref_forward ref_bias). Each
step draws one mode for the whole batch:
  identity (s=0)  reference = the target's character drawn by another artist; the prompt names the
                  target's artists; with probability aug_prob the reference is re-encoded with its
                  fine detail scrambled (downsample/blur/tone jitter), so its style is unusable
  copy     (s=1)  reference = another image of the target's group (same character, same artist);
                  the prompt has no artist, so the style must come from the reference
Without the knob the two modes contradict each other (copy the reference's style or not); with it,
each is unambiguous.

  python train/train_char_v2.py --prep runs/train/dual_r32_prep --config configs/train_char_v2.json \
      --out runs/train/char_v2
  # continue (same or larger --steps) from the newest state saved with the checkpoints
  python train/train_char_v2.py ... --out runs/train/char_v2 --resume runs/train/char_v2/state_step01500.pt
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
from PIL import Image, ImageEnhance, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402
from eval_char_v2 import image_rows, loss_gaps  # noqa: E402
from sample import Prep  # noqa: E402
from train import lr_at  # noqa: E402

STRENGTH = {"identity": 0.0, "copy": 1.0}


def scramble_style(image: Image.Image, rng: random.Random) -> Image.Image:
    """Keep shapes and colours (the identity), lose line work and texture (much of the style)."""
    image = image.convert("RGB")
    factor = rng.uniform(2.0, 4.0)
    small = image.resize((max(8, round(image.width / factor)), max(8, round(image.height / factor))), Image.BICUBIC)
    image = small.resize(image.size, Image.BICUBIC)
    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
    return ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))


class PairSampler:
    """Per step: a mode, then a target-size bucket of targets that support it, then one reference each."""

    def __init__(self, prep: Prep, modes: dict[str, float], batch_size: int, seed: int):
        self.rng = random.Random(seed)
        self.modes, self.batch_size = modes, batch_size
        records, groups = prep.index["records"], prep.index["groups"]
        self.candidates, buckets = {}, defaultdict(list)
        for group in groups.values():
            if group["split"] != "train":
                continue
            char_refs = [r for g in group["partners"]["character"] for r in groups[g]["record_ids"]]
            for target in group["record_ids"]:
                same = [r for r in group["record_ids"] if r != target]
                self.candidates[target] = {"identity": char_refs, "copy": same}
                size = (records[target]["width"], records[target]["height"])
                for mode in modes:
                    if self.candidates[target][mode]:
                        buckets[(mode, size)].append(target)
        self.buckets = dict(buckets)
        self.queues = {key: [] for key in self.buckets}

    def counts(self) -> dict[str, int]:
        out = defaultdict(int)
        for (mode, _), targets in self.buckets.items():
            out[mode] += len(targets)
        return dict(out)

    def next(self):
        mode = self.rng.choices(list(self.modes), weights=list(self.modes.values()))[0]
        keys = [k for k in self.buckets if k[0] == mode]
        key = self.rng.choices(keys, weights=[len(self.buckets[k]) for k in keys])[0]
        queue, batch = self.queues[key], []
        while len(batch) < self.batch_size:
            if not queue:
                queue.extend(self.buckets[key])
                self.rng.shuffle(queue)
            target = queue.pop()
            batch.append((target, self.rng.choice(self.candidates[target][mode])))
        return mode, key[1], batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comfy-root", default="/workspace/ComfyUI")
    parser.add_argument("--init-lora", type=Path, help="start from a reference adapter LoRA (its character slot)")
    parser.add_argument("--steps", type=int, help="total steps (on --resume: a new total; the lr schedule follows it)")
    parser.add_argument("--resume", type=Path, help="state_stepNNNNN.pt written next to the checkpoints")
    parser.add_argument("--no-sample", action="store_true")
    args = parser.parse_args()

    prep = Prep(args.prep)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train_cfg, adapter_cfg, v2_cfg = config["train"], config["adapter"], config["char_v2"]
    if args.steps:
        train_cfg["steps"] = args.steps
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
    embed = torch.nn.Parameter(torch.zeros(dm.model_channels, device=device))
    if args.init_lora:
        init_meta = anima_env.load_lora(modules, args.init_lora)
        init_embed = anima_env.load_style_embed(args.init_lora)
        if init_embed is not None:
            with torch.no_grad():
                embed.copy_(init_embed)
        print(f"initialised from {args.init_lora} (step {init_meta.get('step')})", flush=True)
    params = [p for m in modules.values() for p in (m.lora_down, m.lora_up)]
    optimizer = torch.optim.AdamW([{"params": params, "lr_scale": 1.0},
                                   {"params": [embed], "lr_scale": v2_cfg["embed_lr_scale"]}],
                                  lr=train_cfg["lr"], betas=(0.9, 0.999), weight_decay=train_cfg["weight_decay"])
    print(f"lora modules={len(modules)} params={sum(p.numel() for p in params) / 1e6:.1f}M "
          f"style embed dim={embed.numel()}", flush=True)
    state = None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for name, module in modules.items():
                module.lora_down.copy_(state["lora"][name][0])
                module.lora_up.copy_(state["lora"][name][1])
            embed.copy_(state["embed"])
        optimizer.load_state_dict(state["optimizer"])
        print(f"resumed from {args.resume} (step {state['step']})", flush=True)
    start_step = state["step"] if state else 0

    lf = anima_env.latent_format()

    def model_space(raw: torch.Tensor) -> torch.Tensor:
        return lf.process_in(raw[None].float())[0].to(torch.bfloat16)

    targets = {k: model_space(prep.targets.pop(k)) for k in list(prep.targets)}
    refs = {k: model_space(prep.refs.pop(k)) for k in list(prep.refs)}
    records = prep.index["records"]
    sampler = PairSampler(prep, v2_cfg["modes"], train_cfg["batch_size"], seed)
    print(f"targets per mode: {sampler.counts()}", flush=True)
    rng = random.Random(seed + 1)
    if state:  # continue the exact data order and randomness
        sampler.rng.setstate(state["sampler_rng"])
        sampler.queues = state["sampler_queues"]
        rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
    vae = anima_env.load_vae(config["comfy"]["vae_name"])
    fill = int(config["data"].get("letterbox_fill", 255))

    def scrambled_ref(record_id: str, width: int, height: int) -> torch.Tensor:
        image = scramble_style(prep.image(record_id), rng)
        raw = anima_env.vae_encode_image(vae, anima_env.letterbox(image, width, height, fill))
        return model_space(raw.clone())  # out of inference mode: the batch goes through autograd

    metadata = {"rank": adapter_cfg["lora_rank"], "alpha": adapter_cfg["lora_alpha"],
                "targets": list(anima_env.LORA_TARGETS), "slots": {"character": adapter_cfg["slots"]["character"]},
                "ref_sigma": 0.0, "base": config["comfy"]["unet_name"], "version": 3, "style_transfer": True,
                "init": str(args.init_lora) if args.init_lora else None}
    log_path = args.out / "train_log.jsonl"
    shift = train_cfg["shift"]
    running, mode_loss = [], defaultdict(list)
    sample_at = set(train_cfg.get("sample_at", [])) | {train_cfg["steps"]}

    def evaluate(step: int, images: bool) -> None:
        t0 = time.perf_counter()
        out_dir = args.out / "samples" / f"step{step:05d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        result = loss_gaps(dm, prep, refs, targets, embed.detach(), config, device)
        result["step"] = step
        (out_dir / "loss_gaps.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(json.dumps({"eval_step": step, **result["gaps"]}), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"eval_step": step, **result}) + "\n")
        if images:
            image_rows(dm, prep, refs, embed.detach(), vae, out_dir, config, device)
        print(f"eval at step {step} took {time.perf_counter() - t0:.0f}s", flush=True)

    def save_state(step: int) -> None:
        """Everything needed to continue bit-for-bit: LoRA, embed, optimizer moments, data order, RNGs.
        Only the newest state is kept (~0.6 GB)."""
        path = args.out / f"state_step{step:05d}.pt"
        torch.save({"step": step, "embed": embed.detach().cpu(), "optimizer": optimizer.state_dict(),
                    "lora": {n: (m.lora_down.detach().cpu(), m.lora_up.detach().cpu()) for n, m in modules.items()},
                    "sampler_rng": sampler.rng.getstate(), "sampler_queues": sampler.queues, "rng": rng.getstate(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()},
                   path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
        for old in args.out.glob("state_step*.pt"):
            if old != path:
                old.unlink()

    if args.init_lora and not state and not args.no_sample:
        evaluate(0, images=False)  # the starting point (from scratch every gap is 0)
    started = window_start = time.perf_counter()
    for step in range(start_step, train_cfg["steps"]):
        mode, (width, height), batch = sampler.next()
        x0 = torch.stack([targets[t] for t, _ in batch]).to(device)
        ref_list = []
        for _, r in batch:
            if mode == "identity" and rng.random() < v2_cfg["aug_prob"]:
                ref_list.append(scrambled_ref(r, width, height))
            else:
                ref_list.append(refs[f"{r}@{width}x{height}"])
        ref = torch.stack(ref_list).to(device)
        mix = config["data"]["prompt_mix"]
        keys = ["with_artist" if mode == "identity" else rng.choices(list(mix), weights=list(mix.values()))[0]
                for _ in batch]
        context = anima_env.pad_contexts([prep.context(records[t]["prompts"][k]) for (t, _), k in zip(batch, keys)])
        context = context.to(device, torch.bfloat16)

        u = torch.rand(len(batch), device=device)
        sigma = (shift * u / (1 + (shift - 1) * u)).clamp(1e-3, 1.0)
        noise = torch.randn_like(x0, dtype=torch.float32)
        s = sigma.view(-1, 1, 1, 1, 1)
        xt = ((1 - s) * x0.float() + s * noise).to(torch.bfloat16)
        target_v = noise - x0.float()
        jitter = [rng.randint(0, adapter_cfg["ref_offset_jitter"]), 0]
        frames = anima_env.slot_frames({"character": ref}, x0.shape[-2], adapter_cfg, jitter)
        bias = (STRENGTH[mode] * embed).reshape(1, -1).expand(len(batch), -1)

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, train_cfg) * group["lr_scale"]
        pred = anima_env.ref_forward(dm, xt, sigma, context, refs=frames, ref_bias=bias,
                                     checkpoint_blocks=train_cfg["checkpoint_blocks"])
        loss = torch.nn.functional.mse_loss(pred.float(), target_v)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params + [embed], train_cfg["grad_clip"])
        optimizer.step()
        running.append(loss.item())
        mode_loss[mode].append(loss.item())

        done = step + 1
        if done % train_cfg["log_every"] == 0 or done == train_cfg["steps"]:
            now = time.perf_counter()
            line = {"step": done, "loss": round(sum(running) / len(running), 5),
                    "mode_loss": {m: round(sum(v) / len(v), 5) for m, v in mode_loss.items()},
                    "lr": optimizer.param_groups[0]["lr"], "grad_norm": round(float(grad_norm), 4),
                    "embed_norm": round(float(embed.detach().norm()), 4),
                    "s_per_step": round((now - window_start) / len(running), 3),
                    "elapsed_h": round((now - started) / 3600, 3),
                    "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            line["eta_h"] = round(line["s_per_step"] * (train_cfg["steps"] - done) / 3600, 2)
            print(json.dumps(line), flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line) + "\n")
            running, mode_loss, window_start = [], defaultdict(list), time.perf_counter()
        if done % train_cfg["save_every"] == 0 or done == train_cfg["steps"]:
            anima_env.save_lora(modules, args.out / f"lora_step{done:05d}.safetensors", {**metadata, "step": done},
                                extra={anima_env.STYLE_EMBED_KEY: embed})
            save_state(done)
        if done in sample_at and not args.no_sample:
            evaluate(done, images=True)
            window_start = time.perf_counter()
    print(f"training done in {(time.perf_counter() - started) / 3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
