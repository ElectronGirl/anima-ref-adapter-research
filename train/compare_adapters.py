"""Compare the single-reference (pilot) adapter with the two-reference adapter on held-out data.

Only characters neither adapter trained on are used (held out in the dual prep, and not a training
character of the pilot prep).

1. Held-out denoising loss at fixed sigmas and noise, per reference setup. A reference that the
   adapter uses lowers the loss against a mismatched one; the gaps say what each slot carries:
     char slot: same group (right character + style) / character partner (right character, other
     style) / style partner (other character, right style)
     style slot alone, prompt with character tags: style partner (right style) / character partner
     (same character, wrong style)
     both slots: character partner + style partner / character partner + character partner (style wrong)
2. Image rows for the eval cases: reference | base model | pilot | dual (character slot) | target.

  python train/compare_adapters.py --prep runs/train/dual_r32_prep --pilot-prep runs/train/pilot_prep \
      --pilot runs/train/pilot_r32/lora_step02000.safetensors --dual runs/train/dual_r32/lora_step04000.safetensors \
      --out runs/try/compare_adapters
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--pilot-prep", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--dual", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, default=120)
    parser.add_argument("--comfy-root", default="/workspace/ComfyUI")
    args = parser.parse_args()
    prep = Prep(args.prep)
    config, index, groups = prep.index["config"], prep.index, prep.index["groups"]
    pilot_index = json.loads((args.pilot_prep / "index.json").read_text(encoding="utf-8"))
    pilot_train_chars = {g["character"] for g in pilot_index["groups"].values() if g["split"] == "train"}
    unseen = lambda gid: groups[gid]["split"] != "train" and groups[gid]["character"] not in pilot_train_chars  # noqa: E731

    anima_env.setup_comfy(args.comfy_root)
    device = torch.device("cuda")
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)
    modules = anima_env.inject_lora(dm, config["adapter"]["lora_rank"], config["adapter"]["lora_alpha"])
    lf = anima_env.latent_format()
    adapter_cfg = config["adapter"]

    def model_latent(raw: torch.Tensor) -> torch.Tensor:
        return lf.process_in(raw[None].float().to(device)).to(torch.bfloat16)

    # Loss cases: one target per held-out group that has both partner kinds.
    rng = random.Random(0)
    cases = []
    for gid in sorted(g for g in groups if unseen(g)):
        p = groups[gid]["partners"]
        rids = groups[gid]["record_ids"]
        if not p["character"] or not p["style"] or len(rids) < 2:
            continue
        target = rids[0]
        cases.append({"target": target, "same": rids[1],
                      "char_partner": rng.choice(groups[rng.choice(p["character"])]["record_ids"]),
                      "style_partner": rng.choice(groups[rng.choice(p["style"])]["record_ids"])})
    cases = cases[:args.max_groups]
    print(f"loss cases: {len(cases)} (held-out groups unseen by both adapters)", flush=True)

    setups = {  # name -> (adapter, prompt variant, {slot: case key})
        "base, scene prompt": (None, "stripped", {}),
        "base, + character tags": (None, "with_character", {}),
        "pilot char: same group": ("pilot", "stripped", {"character": "same"}),
        "pilot char: char partner": ("pilot", "stripped", {"character": "char_partner"}),
        "pilot char: style partner": ("pilot", "stripped", {"character": "style_partner"}),
        "dual char: same group": ("dual", "stripped", {"character": "same"}),
        "dual char: char partner": ("dual", "stripped", {"character": "char_partner"}),
        "dual char: style partner": ("dual", "stripped", {"character": "style_partner"}),
        "dual style (+char tags): style partner": ("dual", "with_character", {"style": "style_partner"}),
        "dual style (+char tags): char partner": ("dual", "with_character", {"style": "char_partner"}),
        "dual both: char + style partner": ("dual", "stripped", {"character": "char_partner", "style": "style_partner"}),
        "dual both: char + char partner": ("dual", "stripped", {"character": "char_partner", "style": "char_partner"}),
    }
    losses = defaultdict(lambda: defaultdict(list))
    sigma = torch.tensor(SIGMAS, device=device)
    s = sigma.view(-1, 1, 1, 1, 1)
    for adapter, path in (("pilot", args.pilot), ("dual", args.dual), (None, None)):
        if path is not None:
            anima_env.load_lora(modules, path)
        anima_env.set_lora_enabled(modules, adapter is not None)
        names = [n for n, (a, _, _) in setups.items() if a == adapter]
        for i, case in enumerate(cases):
            record = prep.record(case["target"])
            width, height = record["width"], record["height"]
            x0 = model_latent(prep.targets[case["target"]]).float().expand(len(SIGMAS), -1, -1, -1, -1)
            noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(i), dtype=torch.float32).to(device)
            xt = ((1 - s) * x0 + s * noise).to(torch.bfloat16)
            target_v = noise - x0
            for name in names:
                _, prompt, slots = setups[name]
                ctx = prep.context(record["prompts"][prompt])
                context = anima_env.pad_contexts([ctx] * len(SIGMAS)).to(device, torch.bfloat16)
                slot_refs = {slot: model_latent(prep.refs[f"{case[key]}@{width}x{height}"]).expand(len(SIGMAS), -1, -1, -1, -1)
                             for slot, key in slots.items()}
                frames = anima_env.slot_frames(slot_refs, x0.shape[-2], adapter_cfg) if slot_refs else None
                with torch.inference_mode():
                    pred = anima_env.ref_forward(dm, xt, sigma, context, refs=frames).float()
                err = ((pred - target_v) ** 2).mean(dim=(1, 2, 3, 4))
                for j, sg in enumerate(SIGMAS):
                    losses[name][sg].append(float(err[j]))
            if (i + 1) % 20 == 0:
                print(f"{adapter or 'base'} loss {i + 1}/{len(cases)}", flush=True)

    table = {name: {str(sg): round(sum(v) / len(v), 5) for sg, v in by_sigma.items()} for name, by_sigma in losses.items()}
    for name in table:
        table[name]["mean"] = round(sum(table[name].values()) / len(SIGMAS), 5)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "losses.json").write_text(json.dumps({"cases": len(cases), "sigmas": SIGMAS, "loss": table}, indent=1),
                                          encoding="utf-8")
    header = "setup".ljust(42) + "".join(f"s={sg:<7}" for sg in SIGMAS) + "mean"
    print(header, flush=True)
    for name in setups:
        print(name.ljust(42) + "".join(f"{table[name][str(sg)]:<9}" for sg in SIGMAS) + f"{table[name]['mean']}", flush=True)

    # Images: eval cases of unseen characters, character slot only.
    vae = anima_env.load_vae(config["comfy"]["vae_name"])
    sample_cfg = config["sample"]
    image_cases = [c for c in index["eval_cases"] if unseen(c["group_id"])]
    outputs = defaultdict(dict)
    for adapter, path in (("pilot", args.pilot), ("dual", args.dual), (None, None)):
        if path is not None:
            anima_env.load_lora(modules, path)
        anima_env.set_lora_enabled(modules, adapter is not None)
        for case in image_cases:
            target = prep.record(case["target"])
            width, height = case["width"], case["height"]
            shape = (1, 16, 1, height // 8, width // 8)
            ref = model_latent(prep.refs[f"{case['character_ref']}@{width}x{height}"])
            frames = anima_env.slot_frames({"character": ref}, shape[-2], adapter_cfg) if adapter else None
            x = anima_env.sample_euler(dm, prep.context(target["prompts"]["stripped"]),
                                       prep.context(target["prompts"]["negative"]), shape, case["seed"],
                                       sample_cfg["steps"], sample_cfg["cfg"], config["train"]["shift"], device,
                                       frames=frames)
            outputs[case["group_id"]][adapter or "base"] = to_pil(anima_env.vae_decode(vae, lf.process_out(x.to(device)).float())[0])
    rows = []
    for case in image_cases:
        out = outputs[case["group_id"]]
        row = labeled_row([(f"character ref ({case['character']})", prep.image(case["character_ref"])),
                           ("base model, scene prompt", out["base"]), ("pilot adapter", out["pilot"]),
                           ("dual adapter, character slot", out["dual"]),
                           ("generator target (full tags)", prep.image(case["target"]))])
        row.save(args.out / f"{case['split']}_{case['group_id']}.jpg", quality=90)
        rows.append(row)
    if rows:
        sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
        y = 0
        for row in rows:
            sheet.paste(row, (0, y))
            y += row.height
        sheet.save(args.out / "contact_sheet.jpg", quality=85)
    print(f"images: {len(rows)} cases", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
