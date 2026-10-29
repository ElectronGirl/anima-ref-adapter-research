"""Train a single-purpose reference adapter at several resolutions.

--task identity   character only: reference = the target's character drawn by another artist, prompt
                  names the target's artists (style from the text), with probability aug_prob the
                  reference's detail is scrambled (train_char_v2.scramble_style)
--task copy       character + style: reference = another image of the target's group, no artist tags
--task style      style only: reference = another character drawn by the target's artists, prompt names
                  the character (with_character: character, copyright, appearance, outfit, scene)

Config adapter.arch: "twoframe" (anima_env.ref_forward, reference frame of the target's size) or "asym"
(anima_env.asym_forward: the reference attends only to itself and is cached when sampling; it keeps its
~1 MP size for low and high targets). adapter.ref_lora false freezes the reference stream (base
weights, no backward; ~1.7x faster steps than the two-frame forward).

Each step draws a resolution class for the whole batch (config task.res_mix):
  base  ~1 MP: the prep's target latents and letterboxed references
  low   ~0.56 MP (x0.75 per side): target image downscaled and both re-encoded on the fly
  high  ~1.3 MP: images of an extra run rendered at that size (build_hires_manifest.py), references
        letterboxed and encoded on the fly

  python train/train_ref.py --task identity --prep runs/train/dual_r32_prep --hires-run hires1 \
      --hires-data /workspace/data/hires1 --config configs/train_char_only.json --out runs/train/char_only
  python train/train_ref.py ... --resume runs/train/char_only/state_step01500.pt
Without --hires-run the high class is skipped (base and low only); a later --resume may add it.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "datagen"))
import anima_env  # noqa: E402
from eval_char_v2 import loss_gaps  # noqa: E402
from prepare import prompt_variants, read_results  # noqa: E402
from prepare_blend import text_artist_prompt  # noqa: E402
from prepare_dual import dual_variants  # noqa: E402

TASK_PROMPT = {"identity": "with_artist", "style": "with_character"}  # copy: the prompt_mix variants
from sample import Prep, labeled_row, to_pil  # noqa: E402
from train import lr_at  # noqa: E402
from train_char_v2 import scramble_style  # noqa: E402

LOW = {(1024, 1024): (768, 768), (896, 1152): (672, 864), (832, 1216): (624, 912), (1216, 832): (912, 624)}
HIGH = {(1024, 1024): (1152, 1152), (896, 1152): (1024, 1280), (832, 1216): (960, 1408), (1216, 832): (1408, 960)}  # build_hires_manifest.HIRES
BASE_OF = {**{v: k for k, v in LOW.items()}, **{v: k for k, v in HIGH.items()}}


class MultiResSampler:
    """Per step: a resolution class, a target size within it, then targets with one reference each.
    Targets are base record ids or "h:<hires record id>"; references are base record ids."""

    def __init__(self, prep: Prep, hires: dict, task: str, res_mix: dict[str, float], batch_size: int, seed: int):
        self.rng = random.Random(seed)
        self.batch_size = batch_size
        records, groups = prep.index["records"], prep.index["groups"]
        self.refs, buckets = {}, defaultdict(list)

        def ref_pool(gid: str, exclude: str) -> list[str]:
            if task in ("identity", "style"):
                partners = groups[gid]["partners"]["character" if task == "identity" else "style"]
                return [r for g in partners for r in groups[g]["record_ids"]]
            return [r for r in groups[gid]["record_ids"] if r != exclude]

        for gid, group in groups.items():
            if group["split"] != "train":
                continue
            for target in group["record_ids"]:
                pool = ref_pool(gid, target)
                if not pool:
                    continue
                self.refs[target] = pool
                size = (records[target]["width"], records[target]["height"])
                buckets[("base", size)].append(target)
                buckets[("low", LOW[size])].append(target)
        for rid, rec in hires.items():
            group = groups.get(rec["group_id"])
            if group is None or group["split"] != "train":
                continue
            pool = [r for r in ref_pool(rec["group_id"], rid)]
            if pool:
                self.refs[f"h:{rid}"] = pool
                buckets[("high", (rec["width"], rec["height"]))].append(f"h:{rid}")
        self.buckets = dict(buckets)
        self.res_mix = {k: w for k, w in res_mix.items() if any(b[0] == k for b in self.buckets)}
        self.queues = {key: [] for key in self.buckets}

    def counts(self) -> dict[str, int]:
        out = defaultdict(int)
        for (res, _), targets in self.buckets.items():
            out[res] += len(targets)
        return dict(out)

    def next(self):
        res = self.rng.choices(list(self.res_mix), weights=list(self.res_mix.values()))[0]
        keys = [k for k in self.buckets if k[0] == res]
        key = self.rng.choices(keys, weights=[len(self.buckets[k]) for k in keys])[0]
        queue, batch = self.queues[key], []
        while len(batch) < self.batch_size:
            if not queue:
                queue.extend(self.buckets[key])
                self.rng.shuffle(queue)
            target = queue.pop()
            batch.append((target, self.rng.choice(self.refs[target])))
        return res, key[1], batch


def load_hires(run: str, data_dir: Path) -> dict[str, dict]:
    """Finished records of a build_hires_manifest.py run, with their full metadata."""
    manifest = {}
    for line in (ROOT / "runs" / run / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            manifest[row["record_id"]] = row
    results = read_results(sorted(p for p in data_dir.glob("results_gpu*.jsonl") if not p.name.endswith(".log.jsonl")))
    return {row["record_id"]: {**manifest[row["record_id"]], **row} for row in results if row["record_id"] in manifest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["identity", "copy", "style"])
    parser.add_argument("--stop-at", type=int, help="stop after this step (checkpoint + state saved); the lr schedule "
                        "still follows the configured total, so --resume continues it")
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hires-run", help="runs/<name> of a build_hires_manifest.py run")
    parser.add_argument("--hires-data", type=Path, help="its generation output dir")
    parser.add_argument("--comfy-root", default="/workspace/ComfyUI")
    parser.add_argument("--steps", type=int, help="total steps (on --resume: a new total)")
    parser.add_argument("--resume", type=Path, help="state_stepNNNNN.pt written next to the checkpoints")
    parser.add_argument("--no-sample", action="store_true")
    parser.add_argument("--eval-groups", type=int, default=60, help="held-out groups for the loss gaps")
    parser.add_argument("--eval-cases", type=int, help="eval image cases (default: all)")
    args = parser.parse_args()

    prep = Prep(args.prep)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train_cfg, adapter_cfg, task_cfg = config["train"], config["adapter"], config["task"]
    if args.steps:
        train_cfg["steps"] = args.steps
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({**config, "task_name": args.task}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    (args.out / "train.pid").write_text(str(os.getpid()))  # read by dashboard/server.py

    seed = train_cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    anima_env.setup_comfy(args.comfy_root)
    device = torch.device("cuda")
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)
    print(f"self-test ok: {anima_env.self_test(dm, device)}", flush=True)

    hires = load_hires(args.hires_run, args.hires_data) if args.hires_run else {}
    records = prep.index["records"]
    # Prompts of the hires records (the prep has only its own records' contexts).
    hires_ctx = {}
    if hires:
        clip = anima_env.load_text_encoder(config["comfy"]["clip_name"])
        for rid, rec in hires.items():
            if args.task == "identity":
                texts = {"with_artist": text_artist_prompt(rec, rec["artists"])}
            elif args.task == "style":
                texts = {"with_character": dual_variants(rec, tuple(config["data"]["partial_keep"]), seed)["with_character"]}
            else:
                texts = prompt_variants(rec, tuple(config["data"]["partial_keep"]), seed)
            hires_ctx[rid] = {k: anima_env.encode_prompt(clip, dm, t, device) for k, t in texts.items()}
        del clip
        torch.cuda.empty_cache()
        print(f"hires records={len(hires)} prompts encoded", flush=True)

    modules = anima_env.inject_lora(dm, adapter_cfg["lora_rank"], adapter_cfg["lora_alpha"])
    params = [p for m in modules.values() for p in (m.lora_down, m.lora_up)]
    optimizer = torch.optim.AdamW(params, lr=train_cfg["lr"], betas=(0.9, 0.999), weight_decay=train_cfg["weight_decay"])
    print(f"lora modules={len(modules)} params={sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    state = None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for name, module in modules.items():
                module.lora_down.copy_(state["lora"][name][0])
                module.lora_up.copy_(state["lora"][name][1])
        optimizer.load_state_dict(state["optimizer"])
        print(f"resumed from {args.resume} (step {state['step']})", flush=True)
    start_step = state["step"] if state else 0

    lf = anima_env.latent_format()

    def model_space(raw: torch.Tensor) -> torch.Tensor:
        return lf.process_in(raw[None].float())[0].to(torch.bfloat16)

    targets = {k: model_space(prep.targets.pop(k)) for k in list(prep.targets)}
    refs = {k: model_space(prep.refs.pop(k)) for k in list(prep.refs)}
    sampler = MultiResSampler(prep, hires, args.task, task_cfg["res_mix"], train_cfg["batch_size"], seed)
    print(f"targets per resolution class: {sampler.counts()}", flush=True)
    rng = random.Random(seed + 1)
    if state:
        sampler.rng.setstate(state["sampler_rng"])
        # Buckets added since the checkpoint (e.g. --hires-run given only now) start with empty queues.
        sampler.queues.update({k: v for k, v in state["sampler_queues"].items() if k in sampler.queues})
        rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
    vae = anima_env.load_vae(config["comfy"]["vae_name"])
    fill = int(config["data"].get("letterbox_fill", 255))

    def encode(image: Image.Image) -> torch.Tensor:
        return model_space(anima_env.vae_encode_image(vae, image).clone())  # out of inference mode for autograd

    def ref_latent(record_id: str, width: int, height: int, scramble: bool) -> torch.Tensor:
        key = f"{record_id}@{width}x{height}"
        if key in refs and not scramble:
            return refs[key]
        image = prep.image(record_id)
        if scramble:
            image = scramble_style(image, rng)
        return encode(anima_env.letterbox(image, width, height, fill))

    def hires_image(rid: str) -> Image.Image:
        return Image.open(args.hires_data / hires[rid]["image"]).convert("RGB")

    metadata = {"rank": adapter_cfg["lora_rank"], "alpha": adapter_cfg["lora_alpha"], "targets": list(anima_env.LORA_TARGETS),
                "slots": {"character": adapter_cfg["slots"]["character"]}, "ref_sigma": 0.0,
                "base": config["comfy"]["unet_name"], "version": 2, "task": args.task,
                "resolutions": "0.56-1.3 MP" if hires else "0.56-1 MP"}
    arch, ref_lora = adapter_cfg.get("arch", "twoframe"), adapter_cfg.get("ref_lora", True)
    if arch == "asym":
        metadata.update(arch="asym", ref_lora=ref_lora, ref_cross_attn=False, version=3)
    ref_hook = None if ref_lora else (lambda enter: anima_env.set_lora_enabled(modules, not enter))

    def ref_size(width: int, height: int) -> tuple[int, int]:
        """The asymmetric reference keeps its ~1 MP size whatever the target size; two frames must match."""
        return BASE_OF.get((width, height), (width, height)) if arch == "asym" else (width, height)

    def model_forward(xt, sigma, context, ref, jitter: int = 0, checkpoint_blocks: int = 0):
        offset = (round(adapter_cfg["slots"]["character"]["offset_h"] * (xt.shape[-2] // 2)) + jitter, 0)
        if arch == "asym":
            if ref_lora:
                return anima_env.asym_forward(dm, xt, sigma, context, ref=ref, ref_offset=offset,
                                              checkpoint_blocks=checkpoint_blocks)
            with torch.no_grad():  # frozen reference stream: base weights, no backward
                cache = anima_env.encode_ref_asym(dm, ref, offset, ref_hook=ref_hook)
            return anima_env.asym_forward(dm, xt, sigma, context, ref_kv=cache, checkpoint_blocks=checkpoint_blocks)
        return anima_env.ref_forward(dm, xt, sigma, context, refs=[(ref, 1, offset)], checkpoint_blocks=checkpoint_blocks)
    log_path = args.out / "train_log.jsonl"
    shift = train_cfg["shift"]
    mix = config["data"]["prompt_mix"]
    running, res_loss = [], defaultdict(list)
    sample_at = set(train_cfg.get("sample_at", [])) | {train_cfg["steps"]}

    @torch.inference_mode()
    def eval_rows(out_dir: Path) -> None:
        """Held-out cases at the three resolution classes (artist tags of the target for identity)."""
        rows = []
        sample_cfg = config["sample"]
        for case in prep.index["eval_cases"][:args.eval_cases]:
            target = prep.record(case["target"])
            base_size = (case["width"], case["height"])
            prompt_key = TASK_PROMPT.get(args.task, "stripped")
            ref_id = {"identity": case["character_ref"], "style": case["style_ref"]}.get(args.task) or \
                [r for r in prep.index["groups"][case["group_id"]]["record_ids"] if r != case["target"]][0]
            panels = [(f"reference ({case['split']}: {case['character']})", prep.image(ref_id))]
            if args.task == "style":  # the prompt alone already names the character: show it without the adapter
                w, h = base_size
                anima_env.set_lora_enabled(modules, False)
                x = anima_env.sample_euler(dm, prep.context(target["prompts"][prompt_key]), prep.context(target["prompts"]["negative"]),
                                           (1, 16, 1, h // 8, w // 8), case["seed"], config["sample"]["steps"], config["sample"]["cfg"],
                                           shift, device)
                anima_env.set_lora_enabled(modules, True)
                panels.append(("base model, same prompt", to_pil(anima_env.vae_decode(vae, lf.process_out(x.to(device)).float())[0])))
            for label, (w, h) in (("low", LOW[base_size]), ("base", base_size), ("high", HIGH[base_size])):
                ref = ref_latent(ref_id, *ref_size(w, h), False)[None].to(device)
                ctx_pos, ctx_neg = prep.context(target["prompts"][prompt_key]), prep.context(target["prompts"]["negative"])
                shape = (1, 16, 1, h // 8, w // 8)
                if arch == "asym":
                    offset = (round(adapter_cfg["slots"]["character"]["offset_h"] * (h // 16)), 0)
                    x = anima_env.sample_euler_asym(dm, ctx_pos, ctx_neg, shape, case["seed"], sample_cfg["steps"],
                                                    sample_cfg["cfg"], shift, device, ref, offset, ref_hook=ref_hook)
                else:
                    frames = anima_env.slot_frames({"character": ref}, h // 8, adapter_cfg)
                    x = anima_env.sample_euler(dm, ctx_pos, ctx_neg, shape, case["seed"], sample_cfg["steps"], sample_cfg["cfg"],
                                               shift, device, frames=frames)
                panels.append((f"{label} {w}x{h}", to_pil(anima_env.vae_decode(vae, lf.process_out(x.to(device)).float())[0])))
            panels.append(("generator target", prep.image(case["target"])))
            row = labeled_row(panels)
            row.save(out_dir / f"{case['split']}_{case['group_id']}.jpg", quality=90)
            rows.append(row)
            print(f"sample {case['split']}_{case['group_id']} done", flush=True)
        if rows:
            sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
            y = 0
            for row in rows:
                sheet.paste(row, (0, y))
                y += row.height
            sheet.save(out_dir / "contact_sheet.jpg", quality=85)

    def evaluate(step: int) -> None:
        t0 = time.perf_counter()
        out_dir = args.out / "samples" / f"step{step:05d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.task == "style":
            result = style_gaps()
        else:
            result = loss_gaps(dm, prep, refs, targets, torch.zeros(dm.model_channels), config, device, limit=args.eval_groups,
                               forward=lambda xt, sigma, ctx, ref: model_forward(xt, sigma, ctx, ref))
        result["step"] = step
        (out_dir / "loss_gaps.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(json.dumps({"eval_step": step, **result["gaps"]}), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"eval_step": step, **result}) + "\n")
        eval_rows(out_dir)
        print(f"eval at step {step} took {time.perf_counter() - t0:.0f}s", flush=True)

    @torch.inference_mode()
    def style_gaps() -> dict:
        """Held-out loss with the prompt naming the character (with_character), at fixed sigmas and noise:
          style_gap   ref = same character, other artist  minus  ref = other character, target's artist
                      (> 0: the adapter takes the style, not the character, from the reference)
          vs_base     base model, no reference  minus  ref = other character, target's artist
          same_group  ref = other character, target's artist  minus  ref = same group (same character and artist)
        """
        from eval_char_v2 import SIGMAS, loss_cases
        sigma = torch.tensor(SIGMAS, device=device)
        s = sigma.view(-1, 1, 1, 1, 1)
        n = len(SIGMAS)
        losses = defaultdict(list)
        for i, case in enumerate(loss_cases(prep.index, args.eval_groups)):
            record = prep.record(case["target"])
            width, height = record["width"], record["height"]
            x0 = targets[case["target"]][None].to(device).float().expand(n, -1, -1, -1, -1)
            noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(i), dtype=torch.float32).to(device)
            xt = ((1 - s) * x0 + s * noise).to(torch.bfloat16)
            ctx = anima_env.pad_contexts([prep.context(record["prompts"]["with_character"])] * n).to(device, torch.bfloat16)
            for key in ("char_partner", "style_partner", "same"):
                ref = refs[f"{case[key]}@{width}x{height}"][None].to(device).expand(n, -1, -1, -1, -1)
                losses[key].append(((model_forward(xt, sigma, ctx, ref).float() - (noise - x0)) ** 2).mean().item())
            anima_env.set_lora_enabled(modules, False)
            losses["base"].append(((anima_env.asym_forward(dm, xt, sigma, ctx).float() - (noise - x0)) ** 2).mean().item())
            anima_env.set_lora_enabled(modules, True)
        mean = {k: sum(v) / len(v) for k, v in losses.items()}
        gaps = {"style_gap": mean["char_partner"] - mean["style_partner"], "vs_base": mean["base"] - mean["style_partner"],
                "same_group": mean["style_partner"] - mean["same"]}
        return {"cases": len(losses["base"]), "gaps": {k: round(v, 6) for k, v in gaps.items()},
                "loss": {k: round(v, 6) for k, v in mean.items()}}

    def save_state(step: int) -> None:
        path = args.out / f"state_step{step:05d}.pt"
        torch.save({"step": step, "optimizer": optimizer.state_dict(),
                    "lora": {n: (m.lora_down.detach().cpu(), m.lora_up.detach().cpu()) for n, m in modules.items()},
                    "sampler_rng": sampler.rng.getstate(), "sampler_queues": sampler.queues, "rng": rng.getstate(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
        for old in args.out.glob("state_step*.pt"):
            if old != path:
                old.unlink()

    started = window_start = time.perf_counter()
    for step in range(start_step, train_cfg["steps"]):
        res, (width, height), batch = sampler.next()
        x0_list, ref_list, ctx_list = [], [], []
        for t, r in batch:
            scramble = args.task == "identity" and rng.random() < task_cfg["aug_prob"]
            if res == "high":
                rid = t[2:]
                x0_list.append(model_space(load_file(str(args.hires_data / hires[rid]["latent"]))["latent"].reshape(16, 1, height // 8, width // 8)))
                keys = list(hires_ctx[rid])
                ctx_list.append(hires_ctx[rid][TASK_PROMPT.get(args.task) or rng.choices(keys, weights=[mix[k] for k in keys])[0]])
            else:
                if res == "low":
                    x0_list.append(encode(prep.image(t).resize((width, height), Image.LANCZOS)))
                else:
                    x0_list.append(targets[t])
                prompt = TASK_PROMPT.get(args.task) or rng.choices(list(mix), weights=list(mix.values()))[0]
                ctx_list.append(prep.context(records[t]["prompts"][prompt]))
            ref_list.append(ref_latent(r, *ref_size(width, height), scramble))
        x0 = torch.stack(x0_list).to(device)
        ref = torch.stack(ref_list).to(device)
        context = anima_env.pad_contexts(ctx_list).to(device, torch.bfloat16)

        u = torch.rand(len(batch), device=device)
        sigma = (shift * u / (1 + (shift - 1) * u)).clamp(1e-3, 1.0)
        noise = torch.randn_like(x0, dtype=torch.float32)
        s = sigma.view(-1, 1, 1, 1, 1)
        xt = ((1 - s) * x0.float() + s * noise).to(torch.bfloat16)
        target_v = noise - x0.float()
        jitter = rng.randint(0, adapter_cfg["ref_offset_jitter"])
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, train_cfg)
        pred = model_forward(xt, sigma, context, ref, jitter, train_cfg["checkpoint_blocks"])
        loss = torch.nn.functional.mse_loss(pred.float(), target_v)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, train_cfg["grad_clip"])
        optimizer.step()
        running.append(loss.item())
        res_loss[res].append(loss.item())

        done = step + 1
        if done % train_cfg["log_every"] == 0 or done == train_cfg["steps"]:
            now = time.perf_counter()
            line = {"step": done, "loss": round(sum(running) / len(running), 5),
                    "mode_loss": {m: round(sum(v) / len(v), 5) for m, v in res_loss.items()},
                    "lr": optimizer.param_groups[0]["lr"], "grad_norm": round(float(grad_norm), 4),
                    "s_per_step": round((now - window_start) / len(running), 3),
                    "elapsed_h": round((now - started) / 3600, 3),
                    "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            line["eta_h"] = round(line["s_per_step"] * (train_cfg["steps"] - done) / 3600, 2)
            print(json.dumps(line), flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line) + "\n")
            running, res_loss, window_start = [], defaultdict(list), time.perf_counter()
        if done % train_cfg["save_every"] == 0 or done == train_cfg["steps"]:
            anima_env.save_lora(modules, args.out / f"lora_step{done:05d}.safetensors", {**metadata, "step": done})
            save_state(done)
        if done in sample_at and not args.no_sample:
            evaluate(done)
            window_start = time.perf_counter()
        if args.stop_at and done >= args.stop_at:
            if done % train_cfg["save_every"]:
                anima_env.save_lora(modules, args.out / f"lora_step{done:05d}.safetensors", {**metadata, "step": done})
                save_state(done)
            print(f"stopped at step {done} (--stop-at)", flush=True)
            break
    print(f"training done in {(time.perf_counter() - started) / 3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
