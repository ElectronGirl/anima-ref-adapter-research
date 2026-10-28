"""Try the two-reference adapter on arbitrary images and prompts.

Each prompt is generated once per seed without the adapter (base model) and once with the given
reference images in their slots; rows are saved side by side.

  python train/try_refs.py --lora runs/train/dual_r32/lora_step04000.safetensors --prep runs/train/dual_r32_prep \
      --style ref.png --prompt "1girl, solo, ..." --prompt "..." --seeds 1 2 --out runs/try/x
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
from sample import labeled_row, to_pil  # noqa: E402

QUALITY = "masterpiece, best quality, score_7, newest, highres, absurdres, safe, "


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--prep", type=Path, required=True, help="prep dir; its index gives the config and negative")
    parser.add_argument("--character", type=Path, help="image for the character slot")
    parser.add_argument("--style", type=Path, help="image for the style slot")
    parser.add_argument("--prompt", action="append", required=True, help="tags after the quality prefix")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--size", default="832x1216")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comfy-root", default="/workspace/ComfyUI")
    args = parser.parse_args()
    width, height = map(int, args.size.split("x"))
    index = json.loads((args.prep / "index.json").read_text(encoding="utf-8"))
    config = index["config"]
    negative = index["prompts"][next(iter(index["records"].values()))["prompts"]["negative"]]

    anima_env.setup_comfy(args.comfy_root)
    device = torch.device("cuda")
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)
    modules = anima_env.inject_lora(dm, config["adapter"]["lora_rank"], config["adapter"]["lora_alpha"])
    anima_env.load_lora(modules, args.lora)
    vae = anima_env.load_vae(config["comfy"]["vae_name"])
    clip = anima_env.load_text_encoder(config["comfy"]["clip_name"])
    lf = anima_env.latent_format()
    fill = int(config["data"].get("letterbox_fill", 255))

    slot_refs, ref_panels = {}, []
    for slot, path in (("character", args.character), ("style", args.style)):
        if path is None:
            continue
        with Image.open(path) as image:
            boxed = anima_env.letterbox(image, width, height, fill)
        raw = anima_env.vae_encode_image(vae, boxed)[None].to(device)
        slot_refs[slot] = lf.process_in(raw).to(torch.bfloat16)
        ref_panels.append((f"{slot} ref", boxed))
    slots_label = " + ".join(slot_refs) + " slot" + ("s" if len(slot_refs) > 1 else "")
    shape = (1, 16, 1, height // 8, width // 8)
    ctx_neg = anima_env.encode_prompt(clip, dm, negative, device)

    def generate(ctx_pos, seed, refs, enabled):
        anima_env.set_lora_enabled(modules, enabled)
        frames = anima_env.slot_frames(refs, shape[-2], config["adapter"]) if refs else None
        x = anima_env.sample_euler(dm, ctx_pos, ctx_neg, shape, seed, config["sample"]["steps"], config["sample"]["cfg"],
                                   config["train"]["shift"], device, frames=frames)
        return to_pil(anima_env.vae_decode(vae, lf.process_out(x.to(device)).float())[0])

    args.out.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []
    for p, prompt in enumerate(args.prompt):
        ctx_pos = anima_env.encode_prompt(clip, dm, QUALITY + prompt, device)
        for seed in args.seeds:
            base = generate(ctx_pos, seed, None, False)
            adapted = generate(ctx_pos, seed, slot_refs, True)
            name = f"p{p}_s{seed}"
            base.save(args.out / f"{name}_base.png")
            adapted.save(args.out / f"{name}_adapter.png")
            row = labeled_row(ref_panels + [(f"base model (seed {seed})", base), (f"adapter, {slots_label}", adapted)])
            row.save(args.out / f"{name}.jpg", quality=90)
            rows.append(row)
            summary.append({"name": name, "prompt": QUALITY + prompt, "seed": seed})
            print(f"{name} done", flush=True)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(args.out / "contact_sheet.jpg", quality=85)
    (args.out / "samples.json").write_text(json.dumps({"lora": str(args.lora), "negative": negative, "samples": summary},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
