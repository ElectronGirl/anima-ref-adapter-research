"""Contact sheet for a build_blend_manifest.py --test run: per character pair, the source images of
combo A (reference style) and combo B (prompt style), then the blend row of each scheme.

  python deploy/blend_sheet.py --run blendtest --out runs/blendtest/sheet.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
H = 320


def tile(path: Path, label: str) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((round(image.width * H / image.height), H), Image.LANCZOS)
    canvas = Image.new("RGB", (image.width, H + 22), "white")
    canvas.paste(image, (0, 22))
    ImageDraw.Draw(canvas).text((4, 5), label, fill="black")
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--data-root", default="/workspace/data")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = Path(args.data_root)
    results = {}
    for path in (data / args.run).glob("results_gpu*.jsonl"):
        if path.name.endswith(".log.jsonl"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("status") == "complete":
                results[row["record_id"]] = row
    manifest = {json.loads(l)["record_id"]: json.loads(l) for l in
                (ROOT / "runs" / args.run / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l}
    source_first = {}
    rows = []
    for line in (ROOT / "runs" / args.run / "groups.jsonl").read_text(encoding="utf-8").splitlines():
        group = json.loads(line)
        blend = group["blend"]
        tiles = []
        for role in ("ref", "text"):
            run, gid = blend[f"{role}_run"], blend[f"{role}_group"]
            if run not in source_first:
                source_first[run] = {}
                for l in (ROOT / "runs" / run / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
                    r = json.loads(l)
                    source_first[run].setdefault(r["group_id"], r)
            record = source_first[run][gid]
            tiles.append(tile(data / run / record["image"] if "image" in record else
                              data / run / gid / f"{record['record_id']}.webp",
                              f"{'A ref' if role == 'ref' else 'B text'}: {record['prompt'].split(', safe, ')[1].split(', 1')[0][:40]}"))
        for rid in group["record_ids"]:
            row = results.get(rid)
            if row:
                blend_info = manifest[rid]["blend"]
                tiles.append(tile(data / args.run / row["image"], f"{blend_info['scheme']} s={blend_info['s']}"))
        width = sum(t.width for t in tiles) + 6 * (len(tiles) - 1)
        strip = Image.new("RGB", (width, H + 22), "white")
        x = 0
        for t in tiles:
            strip.paste(t, (x, 0))
            x += t.width + 6
        rows.append(strip)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 8 for r in rows)), "white")
    y = 0
    for strip in rows:
        sheet.paste(strip, (0, y))
        y += strip.height + 8
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=88)
    print(f"{len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
