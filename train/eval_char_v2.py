"""Evaluation of the v2 character adapter (one character reference + a style_transfer strength).

style_transfer 0 should carry the identity only (the style comes from the prompt's artist tags);
1 should carry identity and the reference's drawing style.

Loss gaps on held-out groups (fixed sigmas and noise; a positive gap = the reference helps):
  identity_s0  artist prompt, s=0: other character (same style) - same character (other style)
  leak_s0      artist prompt, s=0: same character other style - same group (same style); ~0 if the
               style comes from the text, not the reference
  copy_s0 / copy_s1  scene prompt: same character other style - same group, at s=0 / s=1;
               copy_s1 - copy_s0 is how much the knob adds the reference's style
Images per eval case: reference | s=0 + target artist tags | s=1 scene | s=0.5 scene | generator target.

  python train/eval_char_v2.py --prep runs/train/dual_r32_prep --lora runs/train/char_v2/lora_step01000.safetensors \
      --out runs/train/char_v2/eval_step01000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402
from sample import Prep, labeled_row, to_pil  # noqa: E402

SIGMAS = (0.15, 0.35, 0.6, 0.85)


def loss_cases(index: dict, limit: int, seed: int = 0) -> list[dict]:
    """One target per held-out group with a character partner, a style partner and a second image."""
    rng = random.Random(seed)
    groups, cases = index["groups"], []
    for gid in sorted(g for g in groups if groups[g]["split"] != "train"):
        p, rids = groups[gid]["partners"], groups[gid]["record_ids"]
        if not p["character"] or not p["style"] or len(rids) < 2:
            continue
        cases.append({"target": rids[0], "same": rids[1],
                      "char_partner": rng.choice(groups[rng.choice(p["character"])]["record_ids"]),
                      "style_partner": rng.choice(groups[rng.choice(p["style"])]["record_ids"])})
    return cases[:limit]


@torch.inference_mode()
def loss_gaps(dm, prep: Prep, refs: dict, targets: dict, embed: torch.Tensor, config: dict, device, limit: int = 60,
              forward=None) -> dict:
    """refs / targets: model-space latents keyed like the prep files."""
    adapter_cfg = config["adapter"]
    setups = {  # name -> (prompt variant, strength, case key of the reference)
        "s0_artist_same": ("with_artist", 0.0, "same"),
        "s0_artist_char": ("with_artist", 0.0, "char_partner"),
        "s0_artist_style": ("with_artist", 0.0, "style_partner"),
        "s0_scene_same": ("stripped", 0.0, "same"),
        "s0_scene_char": ("stripped", 0.0, "char_partner"),
        "s1_scene_same": ("stripped", 1.0, "same"),
        "s1_scene_char": ("stripped", 1.0, "char_partner"),
    }
    losses = defaultdict(list)
    sigma = torch.tensor(SIGMAS, device=device)
    s = sigma.view(-1, 1, 1, 1, 1)
    n = len(SIGMAS)
    cases = loss_cases(prep.index, limit)
    for i, case in enumerate(cases):
        record = prep.record(case["target"])
        width, height = record["width"], record["height"]
        x0 = targets[case["target"]][None].to(device).float().expand(n, -1, -1, -1, -1)
        noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(i), dtype=torch.float32).to(device)
        xt = ((1 - s) * x0 + s * noise).to(torch.bfloat16)
        target_v = noise - x0
        for name, (prompt, strength, key) in setups.items():
            context = anima_env.pad_contexts([prep.context(record["prompts"][prompt])] * n).to(device, torch.bfloat16)
            ref = refs[f"{case[key]}@{width}x{height}"][None].to(device).expand(n, -1, -1, -1, -1)
            if forward is not None:  # another reference architecture (train_ref.py --arch asym); strength unused
                pred = forward(xt, sigma, context, ref).float()
            else:
                frames = anima_env.slot_frames({"character": ref}, x0.shape[-2], adapter_cfg)
                bias = (strength * embed.to(device)).reshape(1, -1).expand(n, -1)
                pred = anima_env.ref_forward(dm, xt, sigma, context, refs=frames, ref_bias=bias).float()
            losses[name].append(((pred - target_v) ** 2).mean().item())
    mean = {k: sum(v) / len(v) for k, v in losses.items()}
    gaps = {
        "identity_s0": mean["s0_artist_style"] - mean["s0_artist_char"],
        "leak_s0": mean["s0_artist_char"] - mean["s0_artist_same"],
        "copy_s0": mean["s0_scene_char"] - mean["s0_scene_same"],
        "copy_s1": mean["s1_scene_char"] - mean["s1_scene_same"],
    }
    gaps["knob"] = gaps["copy_s1"] - gaps["copy_s0"]
    return {"cases": len(cases), "gaps": {k: round(v, 6) for k, v in gaps.items()},
            "loss": {k: round(v, 6) for k, v in mean.items()}}


@torch.inference_mode()
def image_rows(dm, prep: Prep, refs: dict, embed: torch.Tensor, vae, out_dir: Path, config: dict, device,
               limit: int | None = None) -> None:
    lf = anima_env.latent_format()
    sample_cfg, adapter_cfg = config["sample"], config["adapter"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def generate(prompt_index, negative, shape, seed, ref, strength):
        frames = anima_env.slot_frames({"character": ref}, shape[-2], adapter_cfg)
        x = anima_env.sample_euler(dm, prep.context(prompt_index), negative, shape, seed, sample_cfg["steps"],
                                   sample_cfg["cfg"], config["train"]["shift"], device, frames=frames,
                                   ref_bias=strength * embed)
        return to_pil(anima_env.vae_decode(vae, lf.process_out(x.to(device)).float())[0])

    for case in prep.index["eval_cases"][:limit]:
        target = prep.record(case["target"])
        width, height = case["width"], case["height"]
        shape = (1, 16, 1, height // 8, width // 8)
        ref = refs[f"{case['character_ref']}@{width}x{height}"][None].to(device)
        negative = prep.context(target["prompts"]["negative"])
        panels = [(f"reference ({case['split']}: {case['character']})", prep.image(case["character_ref"])),
                  (f"s=0 + {case['artists']}", generate(target["prompts"]["with_artist"], negative, shape, case["seed"], ref, 0.0)),
                  ("s=1, scene prompt", generate(target["prompts"]["stripped"], negative, shape, case["seed"], ref, 1.0)),
                  ("s=0.5, scene prompt", generate(target["prompts"]["stripped"], negative, shape, case["seed"], ref, 0.5)),
                  (f"generator target ({case['artists']})", prep.image(case["target"]))]
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


class BlendPrep:
    """prepare_blend.py output with its latents converted by `convert` (raw VAE -> model space)."""

    def __init__(self, path: Path, convert):
        from safetensors.torch import load_file
        self.dir = Path(path)
        self.index = json.loads((self.dir / "index.json").read_text(encoding="utf-8"))
        ctx = load_file(str(self.dir / "contexts.safetensors"))
        self._data, self._offsets = ctx["data"], ctx["offsets"].tolist()
        self.targets, self.refs = {}, {}
        for kind, out in (("targets", self.targets), ("refs", self.refs)):
            for name in self.index["files"][kind]:
                raw = load_file(str(self.dir / name))
                out.update({k: convert(raw.pop(k)) for k in list(raw)})

    def context(self, i: int) -> torch.Tensor:
        return self._data[self._offsets[i]:self._offsets[i + 1]]

    def image(self, record_id: str) -> Image.Image:
        return Image.open(Path(self.index["data_dir"]) / self.index["records"][record_id]["image"]).convert("RGB")


@torch.inference_mode()
def blend_gaps(dm, prep: Prep, bprep: BlendPrep, refs: dict, targets: dict, embed: torch.Tensor, config: dict,
               device, limit: int = 60) -> dict:
    """Held-out blend targets at the right strength vs the ends, and conflict targets at s=1 vs s=0
    (prompt naming B, reference from (C, A)). Positive = the knob is used the right way:
      blend_vs_s0 / blend_vs_s1  loss at s=0 (or 1) - loss at the record's own s
      ref_wins                   conflict target: loss at s=0 - loss at s=1
    """
    adapter_cfg, groups = config["adapter"], prep.index["groups"]
    sigma = torch.tensor(SIGMAS, device=device)
    s_ = sigma.view(-1, 1, 1, 1, 1)
    n = len(SIGMAS)

    def losses(x0_raw, ref_raw, ctx, strengths, seed):
        x0 = x0_raw[None].to(device).float().expand(n, -1, -1, -1, -1)
        noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32).to(device)
        xt = ((1 - s_) * x0 + s_ * noise).to(torch.bfloat16)
        context = anima_env.pad_contexts([ctx] * n).to(device, torch.bfloat16)
        ref = ref_raw[None].to(device).expand(n, -1, -1, -1, -1)
        frames = anima_env.slot_frames({"character": ref}, x0.shape[-2], adapter_cfg)
        out = []
        for strength in strengths:
            bias = (strength * embed.to(device)).reshape(1, -1).expand(n, -1)
            pred = anima_env.ref_forward(dm, xt, sigma, context, refs=frames, ref_bias=bias).float()
            out.append(((pred - (noise - x0)) ** 2).mean().item())
        return out

    fit0, fit1, wins = [], [], []
    held = [rid for rid, r in bprep.index["records"].items() if r["split"] == "eval"][:limit]
    for i, rid in enumerate(held):
        rec = bprep.index["records"][rid]
        ref_key = f"{groups[rec['ref_group']]['record_ids'][0]}@{rec['width']}x{rec['height']}"
        l0, ls, l1 = losses(bprep.targets[rid], bprep.refs[ref_key], bprep.context(rec["prompts"]["text_artist"]),
                            (0.0, rec["s"], 1.0), 1000 + i)
        fit0.append(l0 - ls)
        fit1.append(l1 - ls)
    conflicts = [c for c in bprep.index["conflicts"] if c["split"] == "eval"][:limit]
    for i, item in enumerate(conflicts):
        target = prep.record(item["target"])
        others = [r for r in groups[item["ref_group"]]["record_ids"] if r != item["target"]]
        if not others:
            continue
        l0, l1 = losses(targets[item["target"]], refs[f"{others[0]}@{target['width']}x{target['height']}"],
                        bprep.context(item["prompt"]), (0.0, 1.0), 2000 + i)
        wins.append(l0 - l1)
    mean = lambda v: round(sum(v) / len(v), 6) if v else None  # noqa: E731
    return {"blend_vs_s0": mean(fit0), "blend_vs_s1": mean(fit1), "ref_wins": mean(wins),
            "blend_cases": len(fit0), "conflict_cases": len(wins)}


@torch.inference_mode()
def blend_rows(dm, prep: Prep, bprep: BlendPrep, embed: torch.Tensor, vae, out_dir: Path, config: dict, device) -> None:
    """Held-out blend cases: reference (C, A) | an image of (C, B) | s = 0 / 0.5 / 1 with the prompt naming B |
    the generated blend at the record's own s."""
    lf = anima_env.latent_format()
    sample_cfg, groups = config["sample"], prep.index["groups"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, rid in enumerate(bprep.index["eval_cases"]):
        rec = bprep.index["records"][rid]
        ref_id = groups[rec["ref_group"]]["record_ids"][0]
        shape = (1, 16, 1, rec["height"] // 8, rec["width"] // 8)
        ref = bprep.refs[f"{ref_id}@{rec['width']}x{rec['height']}"][None].to(device)
        frames = anima_env.slot_frames({"character": ref}, shape[-2], config["adapter"])
        ctx, neg = bprep.context(rec["prompts"]["text_artist"]), bprep.context(rec["prompts"]["negative"])
        panels = [("reference (C, A)", prep.image(ref_id)),
                  ("prompt artist (C, B)", prep.image(groups[rec["text_group"]]["record_ids"][0]))]
        for strength in (0.0, 0.5, 1.0):
            x = anima_env.sample_euler(dm, ctx, neg, shape, 777 + k, sample_cfg["steps"], sample_cfg["cfg"],
                                       config["train"]["shift"], device, frames=frames, ref_bias=strength * embed)
            panels.append((f"s={strength}, prompt names B", to_pil(anima_env.vae_decode(vae, lf.process_out(x.to(device)).float())[0])))
        panels.append((f"generated blend target (s={rec['s']})", bprep.image(rid)))
        row = labeled_row(panels)
        row.save(out_dir / f"blend_{rid}.jpg", quality=90)
        rows.append(row)
        print(f"blend sample {rid} done", flush=True)
    if rows:
        sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
        y = 0
        for row in rows:
            sheet.paste(row, (0, y))
            y += row.height
        sheet.save(out_dir / "blend_sheet.jpg", quality=85)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--max-groups", type=int, default=60, help="held-out groups for the loss gaps")
    parser.add_argument("--max-cases", type=int, help="eval cases for the images (default: all)")
    parser.add_argument("--blend-prep", type=Path, help="prepare_blend.py output: adds the blend gaps and rows")
    parser.add_argument("--comfy-root", default="/workspace/ComfyUI")
    args = parser.parse_args()
    prep = Prep(args.prep)
    config = prep.index["config"]
    anima_env.setup_comfy(args.comfy_root)
    device = torch.device("cuda")
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)
    modules = anima_env.inject_lora(dm, config["adapter"]["lora_rank"], config["adapter"]["lora_alpha"])
    anima_env.load_lora(modules, args.lora)
    embed = anima_env.load_style_embed(args.lora)
    if embed is None:  # an adapter without the knob: every strength is the same
        embed = torch.zeros(dm.model_channels)
    lf = anima_env.latent_format()
    convert = lambda raw: lf.process_in(raw[None].float())[0].to(torch.bfloat16)  # noqa: E731
    refs = {k: convert(prep.refs.pop(k)) for k in list(prep.refs)}
    targets = {k: convert(prep.targets.pop(k)) for k in list(prep.targets)}
    args.out.mkdir(parents=True, exist_ok=True)
    result = loss_gaps(dm, prep, refs, targets, embed, config, device, limit=args.max_groups)
    bprep = BlendPrep(args.blend_prep, convert) if args.blend_prep else None
    if bprep:
        result["gaps"].update(blend_gaps(dm, prep, bprep, refs, targets, embed, config, device, limit=args.max_groups))
    print(json.dumps(result), flush=True)
    (args.out / "loss_gaps.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    if not args.no_images:
        vae = anima_env.load_vae(config["comfy"]["vae_name"])
        image_rows(dm, prep, refs, embed, vae, args.out, config, device, limit=args.max_cases)
        if bprep:
            blend_rows(dm, prep, bprep, embed, vae, args.out, config, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
