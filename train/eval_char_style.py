"""Character only vs character + its drawing style, with one reference image.

The two-reference adapter's intended uses: the image in the character slot alone should carry the
identity only, and the same image in both slots should carry identity and drawing style. For each
eval case of a prepare_dual.py prep, the character ref (the target's character drawn by another
artist) is used both ways with the scene prompt, next to the base model on the same prompt.

Row: reference | base model (no adapter) | character slot only | same image in both slots.

  python train/eval_char_style.py --prep runs/train/dual_r32_prep --lora runs/train/dual_r32/lora_step04000.safetensors \
      --out runs/try/char_style
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402
from prepare_dual import artist_prompt  # noqa: E402
from sample import Prep, labeled_row, to_pil  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comfy-root", default="/workspace/ComfyUI")
    args = parser.parse_args()
    prep = Prep(args.prep)
    config = prep.index["config"]
    anima_env.setup_comfy(args.comfy_root)
    device = torch.device("cuda")
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)
    modules = anima_env.inject_lora(dm, config["adapter"]["lora_rank"], config["adapter"]["lora_alpha"])
    anima_env.load_lora(modules, args.lora)
    vae = anima_env.load_vae(config["comfy"]["vae_name"])
    lf = anima_env.latent_format()
    sample_cfg = config["sample"]

    def generate(prompt_index, negative, shape, seed, slot_refs):
        anima_env.set_lora_enabled(modules, bool(slot_refs))
        frames = anima_env.slot_frames(slot_refs, shape[-2], config["adapter"]) if slot_refs else None
        x = anima_env.sample_euler(dm, prep.context(prompt_index), negative, shape, seed, sample_cfg["steps"],
                                   sample_cfg["cfg"], config["train"]["shift"], device, frames=frames)
        return to_pil(anima_env.vae_decode(vae, lf.process_out(x.to(device)).float())[0])

    args.out.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []
    for case in prep.index["eval_cases"]:
        target = prep.record(case["target"])
        width, height = case["width"], case["height"]
        shape = (1, 16, 1, height // 8, width // 8)
        ref_key = f"{case['character_ref']}@{width}x{height}"
        ref = lf.process_in(prep.refs[ref_key][None].float().to(device)).to(torch.bfloat16)
        negative = prep.context(target["prompts"]["negative"])
        scene = target["prompts"]["stripped"]
        base = generate(scene, negative, shape, case["seed"], {})
        char_only = generate(scene, negative, shape, case["seed"], {"character": ref})
        both = generate(scene, negative, shape, case["seed"], {"character": ref, "style": ref})
        ref_artists = artist_prompt(json.loads(prep.record(case["character_ref"])["style"]))
        name = f"{case['split']}_{case['group_id']}"
        row = labeled_row([(f"reference ({ref_artists})", prep.image(case["character_ref"])),
                           ("base model, no adapter", base),
                           ("character slot only", char_only),
                           ("same image in both slots", both)])
        row.save(args.out / f"{name}.jpg", quality=90)
        rows.append(row)
        summary.append({**case, "ref_artists": ref_artists, "prompt": prep.index["prompts"][scene], "image": f"{name}.jpg"})
        print(f"{name} done", flush=True)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(args.out / "contact_sheet.jpg", quality=85)
    (args.out / "samples.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
