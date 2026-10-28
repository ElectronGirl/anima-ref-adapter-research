"""Readable comparison figures from the training-time eval samples.

The eval rows written by train/sample.py have tiny baked-in labels. This rebuilds them as a grid with
large column headers and a character label per row; figures/heldout_prompts.json lists each row's prompt.

  python deploy/make_comparison_figures.py --samples runs/train/pilot_r32_remote/samples \
      --data /workspace/data/pilot --out docs
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_showcase_figures import ACCENT, BG, FG, MUTED, font, paste_card  # noqa: E402

PANEL_H, LABEL_H, GAP = 512, 28, 8  # layout of train/sample.py labeled_row
COLUMNS = [("ref", "Reference", "input image"),
           ("adapter", "Adapter", "scene-only prompt"),
           ("base", "No adapter", "same prompt"),
           ("truth", "With character tags", "full prompt (target)")]
FIGURES = {
    "fig2_heldout": ["held_out_g000184", "held_out_g000183", "held_out_g000056", "held_out_g000204"],
    "fig6_failures": ["held_out_g000030", "seen_g000001"],
}
CELL_W, CELL_H, NAME_W = 330, 420, 230
CARD = (26, 29, 36)


def pretty_name(tag: str) -> tuple[str, str]:
    """'tohru \\(maidragon\\)' -> ('Tohru', 'maidragon')."""
    tag = tag.replace("\\", "")
    match = re.match(r"(.*?)\s*\((.*)\)$", tag)
    name, series = (match.group(1), match.group(2)) if match else (tag, "")
    return name.title(), series


def panels(case: dict, samples: Path, data: Path) -> dict[str, Image.Image]:
    ref = Image.open(data / case["group_id"] / f"{case['ref']}.webp").convert("RGB")
    truth = Image.open(data / case["group_id"] / f"{case['target']}.webp").convert("RGB")
    name = f"{case['split']}_{case['group_id']}"
    base = Image.open(samples / "baseline" / f"{name}_baseline.png").convert("RGB")
    row = Image.open(samples / "step02000" / f"{name}.jpg").convert("RGB")
    ref_w = round(ref.width * PANEL_H / ref.height)
    out_w = round(case["width"] * PANEL_H / case["height"])
    x0 = ref_w + GAP
    adapter = row.crop((x0, LABEL_H, x0 + out_w, LABEL_H + PANEL_H))
    return {"ref": ref, "adapter": adapter, "base": base, "truth": truth}


def contain(image: Image.Image, w: int, h: int) -> Image.Image:
    scale = min(w / image.width, h / image.height)
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)


def figure(rows: list[tuple[dict, dict]], path: Path) -> None:
    pad, head_h, row_gap, col_gap = 44, 120, 28, 18
    width = pad * 2 + NAME_W + len(COLUMNS) * CELL_W + (len(COLUMNS) - 1) * col_gap
    height = pad + head_h + len(rows) * (CELL_H + row_gap)
    canvas = Image.new("RGBA", (width, height), BG + (255,))
    draw = ImageDraw.Draw(canvas)
    for c, (_, title, sub) in enumerate(COLUMNS):
        x = pad + NAME_W + c * (CELL_W + col_gap)
        color = ACCENT if c == 1 else FG
        draw.text((x + CELL_W // 2, pad + 30), title, font=font(32, True), fill=color, anchor="mm")
        draw.text((x + CELL_W // 2, pad + 74), sub, font=font(24), fill=MUTED, anchor="mm")
    for r, (case, images) in enumerate(rows):
        y = pad + head_h + r * (CELL_H + row_gap)
        name, series = pretty_name(case["character"])
        name_font = font(30, True)
        lines = []
        for word in name.split():  # wrap the name inside the label column
            if lines and draw.textlength(f"{lines[-1]} {word}", font=name_font) <= NAME_W - 24:
                lines[-1] = f"{lines[-1]} {word}"
            else:
                lines.append(word)
        ty = y + CELL_H // 2 - 22 - 19 * (len(lines) - 1)
        for line in lines:
            draw.text((pad, ty), line, font=name_font, fill=FG, anchor="lm")
            ty += 38
        draw.text((pad, ty + 2), series, font=font(22), fill=MUTED, anchor="lm")
        tag = "held out" if case["split"] == "held_out" else "seen in training"
        draw.text((pad, ty + 36), tag, font=font(20, True), fill=ACCENT, anchor="lm")
        for c, (key, _, _) in enumerate(COLUMNS):
            x = pad + NAME_W + c * (CELL_W + col_gap)
            card = Image.new("RGB", (CELL_W, CELL_H), CARD)
            img = contain(images[key], CELL_W, CELL_H)
            card.paste(img, ((CELL_W - img.width) // 2, (CELL_H - img.height) // 2))
            paste_card(canvas, card, (x, y), radius=16)
    canvas.convert("RGB").save(path, quality=90)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="docs dir: writes figures/*.jpg")
    args = parser.parse_args()
    cases = {f"{c['split']}_{c['group_id']}": c for c in
             json.loads((args.samples / "step02000" / "samples.json").read_text(encoding="utf-8"))}
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    index = []
    for fig, names in FIGURES.items():
        rows = []
        for name in names:
            case = cases[name]
            images = panels(case, args.samples, args.data)
            rows.append((case, images))
            index.append({"name": name, "figure": fig, "character": pretty_name(case["character"]),
                          "split": case["split"], "prompt": case["prompt"]})
        figure(rows, args.out / "figures" / f"{fig}.jpg")
    (args.out / "figures" / "heldout_prompts.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
