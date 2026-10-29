"""Try an asymmetric-attention adapter (train_ref.py arch "asym") on arbitrary reference images.

Per reference: the base model on the prompt, then the adapter with reference guidance 1 (reference
in both CFG branches) and with the ComfyUI guider's default (ref_guidance 4.5 + APG).

  python train/try_asym.py --lora runs/train/copy_asym/lora_step02000.safetensors --prep runs/train/dual_r32_prep \
      --ref a.png --size 896x1152 --prompt "1girl, solo, ..." [--ref b.png --size ... --prompt ...] --out runs/try/x
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402
from sample import labeled_row, to_pil  # noqa: E402

QUALITY = "masterpiece, best quality, score_7, newest, highres, absurdres, safe, "


@torch.inference_mode()
def sample_guided(dm, ctx_pos, ctx_neg, shape, seed, steps, cfg, ref_guidance, shift, device, cache):
    """Euler with the node's guidance (AnimaRefGuiderImpl.predict_noise) in x0 space:
    base + ref_guidance * (reference effect) + cfg * (text effect), then APG. cache None = base model."""
    noise = torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32)
    x = noise.to(device)
    ctx = anima_env.pad_contexts([ctx_pos, ctx_neg]).to(device, torch.bfloat16)
    sigmas = anima_env.flow_sigmas(steps, shift)
    kv = [(k.expand(2, *k.shape[1:]), v.expand(2, *v.shape[1:])) for k, v in cache] if cache else None
    for i in range(steps):
        sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
        xb = x.to(torch.bfloat16).expand(2, *x.shape[1:])
        s = torch.full((2,), sigma, device=device)
        v = anima_env.asym_forward(dm, xb, s, ctx, ref_kv=kv).float()
        x0 = x - sigma * v  # flow matching: x = (1 - s) x0 + s noise, v = noise - x0
        pos, neg = x0[0:1], x0[1:2]
        if kv is not None and ref_guidance != 1.0:
            base = x - sigma * anima_env.asym_forward(dm, xb[1:], s[1:], ctx[1:]).float()
            guided = base + ref_guidance * (neg - base) + cfg * (pos - neg)
            update = (guided - pos).double()
            unit = F.normalize(pos.double().flatten(1), dim=1).view_as(pos)
            guided = pos + (update - (update * unit).sum() * unit).float()  # APG: drop the part along pos
        else:
            guided = neg + cfg * (pos - neg)
        v = (x - guided) / max(sigma, 1e-4)
        x = x + (sigma_next - sigma) * v
    return x


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--prep", type=Path, required=True, help="prep dir: its index gives the config and negative")
    parser.add_argument("--ref", type=Path, action="append", required=True)
    parser.add_argument("--size", action="append", required=True)
    parser.add_argument("--prompt", action="append", required=True, help="tags after the quality prefix")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comfy-root", default="/workspace/ComfyUI")
    args = parser.parse_args()
    index = json.loads((args.prep / "index.json").read_text(encoding="utf-8"))
    config = index["config"]
    negative = index["prompts"][next(iter(index["records"].values()))["prompts"]["negative"]]
    anima_env.setup_comfy(args.comfy_root)
    device = torch.device("cuda")
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)
    from safetensors import safe_open
    with safe_open(str(args.lora), "pt") as handle:
        lora_meta = json.loads((handle.metadata() or {}).get("anima_ref_adapter", "{}"))
    modules = anima_env.inject_lora(dm, lora_meta.get("rank", 32), lora_meta.get("alpha", 32))
    meta = anima_env.load_lora(modules, args.lora)
    assert meta.get("arch") == "asym", "not an asym adapter"
    vae = anima_env.load_vae(config["comfy"]["vae_name"])
    clip = anima_env.load_text_encoder(config["comfy"]["clip_name"])
    lf = anima_env.latent_format()
    ctx_neg = anima_env.encode_prompt(clip, dm, negative, device)
    steps, cfg, shift = config["sample"]["steps"], config["sample"]["cfg"], config["train"]["shift"]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (ref_path, size, prompt) in enumerate(zip(args.ref, args.size, args.prompt)):
        width, height = map(int, size.split("x"))
        shape = (1, 16, 1, height // 8, width // 8)
        with Image.open(ref_path) as image:
            boxed = anima_env.letterbox(image.convert("RGB"), width, height, 255)
        ref = lf.process_in(anima_env.vae_encode_image(vae, boxed)[None].to(device)).to(torch.bfloat16)
        ctx_pos = anima_env.encode_prompt(clip, dm, QUALITY + prompt, device)
        panels = [("reference", boxed)]
        anima_env.set_lora_enabled(modules, False)
        x = sample_guided(dm, ctx_pos, ctx_neg, shape, args.seed, steps, cfg, 1.0, shift, device, None)
        panels.append(("base model", to_pil(anima_env.vae_decode(vae, lf.process_out(x.to(device)).float())[0])))
        anima_env.set_lora_enabled(modules, True)
        with torch.inference_mode():
            cache = anima_env.encode_ref_asym(dm, ref, (round(height / 16), 0))
        for rg in (1.0, 4.5):
            x = sample_guided(dm, ctx_pos, ctx_neg, shape, args.seed, steps, cfg, rg, shift, device, cache)
            image = to_pil(anima_env.vae_decode(vae, lf.process_out(x.to(device)).float())[0])
            image.save(args.out / f"ref{i + 1}_rg{rg}.png")
            panels.append((f"adapter, ref_guidance {rg}" + (" + APG" if rg != 1 else ""), image))
        row = labeled_row(panels)
        row.save(args.out / f"ref{i + 1}.jpg", quality=92)
        rows.append(row)
        print(f"ref {i + 1} done", flush=True)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(args.out / "contact_sheet.jpg", quality=88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
