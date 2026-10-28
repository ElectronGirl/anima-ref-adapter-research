"""Try the adapter on reference images made from a spec (artist + described character), not the dataset.

For each character: the base model draws a reference image from the full prompt (artist +
appearance tags + reference scene). For each target scene the adapter then draws it from a prompt
without artist or appearance tags, next to the base model drawing it with the full prompt.

  python train/try_reference.py --spec configs/try_quasarcake.json --lora LORA --out runs/try/quasarcake
  # reference guidance sweep, reusing the references and full-prompt images of an earlier run
  python train/try_reference.py ... --ref-scales 1 2 3 4.5 --reuse runs/try/quasarcake --out runs/try/quasarcake_refcfg
  # guidance corrections at a reference scale: SCALE[:none|rescale<phi>|apg]
  python train/try_reference.py ... --variants 4.5 4.5:rescale0.7 4.5:apg --reuse runs/try/quasarcake --out ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402
from sample import labeled_row, to_pil  # noqa: E402


def stack_rows(rows: list[Image.Image]) -> Image.Image:
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.json")
    parser.add_argument("--comfy-root", default="/workspace/ComfyUI")
    parser.add_argument("--ref-scales", type=float, nargs="+",
                        help="reference guidance scales to compare (default: one column, reference in both CFG branches)")
    parser.add_argument("--variants", nargs="+",
                        help="SCALE[:none|rescale<phi>|apg] reference guidance with a guidance correction (overrides --ref-scales)")
    parser.add_argument("--reuse", type=Path, help="earlier output dir whose reference / full-prompt images to reuse")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sample_cfg, shift = config["sample"], config["train"]["shift"]
    args.out.mkdir(parents=True, exist_ok=True)

    anima_env.setup_comfy(args.comfy_root)
    device = torch.device("cuda")
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)
    clip = anima_env.load_text_encoder(config["comfy"]["clip_name"])
    modules = anima_env.inject_lora(dm, config["adapter"]["lora_rank"], config["adapter"]["lora_alpha"])
    anima_env.load_lora(modules, args.lora)
    vae = anima_env.load_vae(config["comfy"]["vae_name"])
    lf = anima_env.latent_format()

    def ctx(text):
        return anima_env.encode_prompt(clip, dm, text, device)

    negative = ctx(spec["negative"])

    def decode(x):
        return to_pil(anima_env.vae_decode(vae, lf.process_out(x).float())[0])

    def generate(prompt_ctx, width, height, seed, ref=None, ref_scale=None, mode="none"):
        shape = (1, 16, 1, height // 8, width // 8)
        offset = anima_env.default_ref_offset(shape[-2], config["adapter"])
        if ref_scale is None:
            anima_env.set_lora_enabled(modules, ref is not None)
            x = anima_env.sample_euler(dm, prompt_ctx, negative, shape, seed, sample_cfg["steps"], sample_cfg["cfg"],
                                       shift, device, ref=ref, ref_offset=offset)
        else:
            guidance, phi = ("rescale", float(mode[len("rescale"):])) if mode.startswith("rescale") else (mode, 0.7)
            x = anima_env.sample_euler_ref_guidance(dm, modules, prompt_ctx, negative, shape, seed, sample_cfg["steps"],
                                                    sample_cfg["cfg"], ref_scale, shift, device, ref, offset,
                                                    guidance=guidance, rescale_phi=phi)
        return decode(x)

    def reused(name):
        path = args.reuse / name if args.reuse else None
        return Image.open(path).convert("RGB") if path and path.exists() else None

    def join(*parts):
        return ", ".join(p for p in parts if p)

    if args.variants:
        variants = [(float(v.split(":")[0]), v.split(":")[1] if ":" in v else "none") for v in args.variants]
    else:
        variants = [(scale, "none") for scale in (args.ref_scales or [None])]
    report = []
    for char in spec["characters"]:
        name = char["name"]

        def full(scene):
            return join(spec["quality"], spec["artist"], char["gender"], "solo", char["description"], scene)

        def stripped(scene):
            return join(spec["quality"], char["gender"], "solo", scene)

        ref_w, ref_h = char.get("ref_size", [896, 1152])
        ref_image = reused(f"{name}_ref.png") or generate(ctx(full(char["ref_scene"])), ref_w, ref_h, char["seed"])
        ref_image.save(args.out / f"{name}_ref.png")
        rows = []
        for i, target in enumerate(spec["targets"]):
            width, height = target["size"]
            seed = char["seed"] + 100 + i
            ref_latent = anima_env.vae_encode_image(vae, anima_env.letterbox(ref_image, width, height))
            ref = lf.process_in(ref_latent[None].to(device)).to(torch.bfloat16)
            prompt_ctx = ctx(stripped(target["scene"]))
            panels = [(f"reference: {name}", ref_image)]
            for scale, mode in variants:
                image = generate(prompt_ctx, width, height, seed, ref=ref, ref_scale=scale, mode=mode)
                tag = "" if scale is None else f"_ref{scale:g}" + ("" if mode == "none" else f"_{mode}")
                label = "adapter, scene-only prompt" if scale is None else f"adapter, ref guidance {scale:g} {mode}"
                image.save(args.out / f"{name}_t{i}_adapter{tag}.png")
                panels.append((label, image))
            teacher = reused(f"{name}_t{i}_fullprompt.png") or generate(ctx(full(target["scene"])), width, height, seed)
            teacher.save(args.out / f"{name}_t{i}_fullprompt.png")
            panels.append(("no adapter, artist + description prompt", teacher))
            rows.append(labeled_row(panels))
            report.append({"character": name, "target": i, "adapter_prompt": stripped(target["scene"]),
                           "full_prompt": full(target["scene"]), "variants": variants, "text_cfg": sample_cfg["cfg"]})
            print(f"{name} target {i} done", flush=True)
        stack_rows(rows).save(args.out / f"{name}_sheet.jpg", quality=90)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
