"""Turn showcase renders into web-sized gallery images and a hero banner for the README / model card.

  python deploy/make_showcase_figures.py --images runs/showcase/images --refs runs/showcase/refs \
      --out release/github/ComfyUI-Anima-Ref-Adapter/docs/showcase
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

BG = (14, 16, 21)
FG = (236, 238, 243)
MUTED = (150, 158, 172)
ACCENT = (255, 158, 196)
FONT_DIR = Path("C:/Windows/Fonts")


def font(size: int, bold: bool = False):
    for name in (["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]):
        path = FONT_DIR / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def rounded(image: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, image.width - 1, image.height - 1], radius, fill=255)
    out = Image.new("RGBA", image.size)
    out.paste(image.convert("RGB"), (0, 0), mask)
    return out


def fit_height(image: Image.Image, height: int) -> Image.Image:
    return image.resize((round(image.width * height / image.height), height), Image.LANCZOS)


def paste_card(canvas: Image.Image, image: Image.Image, xy: tuple[int, int], radius: int = 18) -> None:
    shadow = Image.new("RGBA", (image.width + 40, image.height + 40), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle([20, 24, image.width + 20, image.height + 24], radius, fill=(0, 0, 0, 140))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    canvas.alpha_composite(shadow, (xy[0] - 20, xy[1] - 20))
    canvas.alpha_composite(rounded(image, radius), xy)


def arrow(draw: ImageDraw.ImageDraw, x: int, y: int, length: int = 70) -> None:
    draw.line([x, y, x + length, y], fill=ACCENT, width=6)
    draw.polygon([(x + length + 4, y), (x + length - 18, y - 14), (x + length - 18, y + 14)], fill=ACCENT)


def banner(rows: list[tuple[Image.Image, list[Image.Image]]], path: Path, tile_h: int = 460) -> None:
    """Title, then one row per reference: reference card -> output cards."""
    pad, top, gap, row_gap = 40, 186, 22, 92
    laid = [(fit_height(ref, tile_h), [fit_height(o, tile_h) for o in outs]) for ref, outs in rows]
    width = max(pad * 2 + r.width + 120 + sum(o.width for o in outs) + gap * (len(outs) - 1) for r, outs in laid)
    height = top + len(laid) * (tile_h + row_gap) + 10
    canvas = Image.new("RGBA", (width, height), BG + (255,))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 36), "Anima Reference Adapter", font=font(64, True), fill=FG)
    draw.text((pad + 4, 112), "One reference image, any scene: character and style from the image, pose and place from the prompt.",
              font=font(26), fill=MUTED)
    y = top
    for ref_t, outs in laid:
        paste_card(canvas, ref_t, (pad, y))
        draw.text((pad + 6, y + tile_h + 18), "REFERENCE", font=font(22, True), fill=ACCENT)
        arrow(draw, pad + ref_t.width + 26, y + tile_h // 2)
        x = pad + ref_t.width + 120
        for out in outs:
            paste_card(canvas, out, (x, y))
            x += out.width + gap
        draw.text((pad + ref_t.width + 126, y + tile_h + 18), "SCENE-ONLY PROMPT  -  no character, artist or appearance tags",
                  font=font(22, True), fill=MUTED)
        y += tile_h + row_gap
    canvas.convert("RGB").save(path, quality=90)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hero", nargs="+", default=["man_night_rain", "man_cafe", "man_beach", "girl_neon", "girl_cake", "girl_umbrella"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((args.images / "showcase.json").read_text(encoding="utf-8"))

    for ref in sorted(args.refs.glob("*.png")):
        fit_height(Image.open(ref).convert("RGB"), 900).save(args.out / f"ref_{ref.stem.split('_', 1)[1]}.jpg", quality=90)
    for job in jobs:
        fit_height(Image.open(args.images / job["file"]).convert("RGB"), 900).save(args.out / f"{job['name']}.jpg", quality=88)

    man = Image.open(args.refs / "showcase_man.png").convert("RGB")
    girl = Image.open(args.refs / "showcase_girl.png").convert("RGB")
    hero = {name: Image.open(args.images / f"{name}.png").convert("RGB") for name in args.hero}
    man_outs = [hero[n] for n in args.hero if n.startswith("man_")]
    girl_outs = [hero[n] for n in args.hero if n.startswith("girl_")]
    banner([(man, man_outs), (girl, girl_outs)], args.out / "banner.jpg")
    print("banner", Image.open(args.out / "banner.jpg").size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
