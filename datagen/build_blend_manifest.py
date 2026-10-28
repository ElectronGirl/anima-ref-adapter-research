"""Style-blend targets for the style_transfer knob: one character drawn in a mix of two artist combos.

For a character C rendered in earlier runs with artist combos A (the reference's style) and B (the
style the prompt will name), a blend group renders C in new scenes with A's weights scaled by s and
B's by 1 - s. Trained with the reference from group (C, A), the prompt naming B and strength s, the
adapter learns s as "how much of the reference's style versus the prompt's artist". The pure ends
(s = 0: group (C, B); s = 1: group (C, A)) already exist in the source runs.

  python datagen/build_blend_manifest.py --run-name blend1 --sources pilot grid1 --levels 0.25 0.5 0.75 --images-per-pair 1
  python datagen/build_blend_manifest.py --run-name blendtest --sources grid1 --test 8   # weight-scheme check

--test N: N character pairs, every scheme, s in {0, 0.25, 0.5, 0.75, 1}, one shared scene and seed
per pair, so a row shows how the mix moves.

Output under runs/<run_name>/: manifest.jsonl, shard_0.jsonl, groups.jsonl, manifest_summary.json
(same layout as build_manifest.py; records carry an extra "blend" field).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_manifest import artist_prompt, sample_scene, scene_tags, write_jsonl  # noqa: E402
from catalog import anima_tag  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def blend_weights(s: float, scheme: str) -> tuple[float, float]:
    """Multipliers for (A, B) artist weights at reference share s. In the blendtest check (8 pairs)
    "linear" moved the style steadily from B to A; with "floor" the three middle levels looked alike."""
    if scheme == "linear":      # proportional, total kept at 1.2
        return 1.2 * s, 1.2 * (1 - s)
    if scheme == "floor":       # both stay present; the share moves between 0.5 and 1.1
        return 0.5 + 0.6 * s, 0.5 + 0.6 * (1 - s)
    raise ValueError(scheme)


def blend_combo(a: list[dict], b: list[dict], s: float, scheme: str, a_first: bool) -> list[dict]:
    if s >= 1.0:
        return a
    if s <= 0.0:
        return b
    wa, wb = blend_weights(s, scheme)
    part_a = [{"tag": x["tag"], "weight": round(x["weight"] * wa, 2)} for x in a]
    part_b = [{"tag": x["tag"], "weight": round(x["weight"] * wb, 2)} for x in b]
    return part_a + part_b if a_first else part_b + part_a


def load_source_groups(runs: list[str]) -> dict[str, list[dict]]:
    """character -> groups [{run, group_id, artists, record (first record: identity, rating, ...)}]."""
    by_char = defaultdict(list)
    for run in runs:
        first = {}
        for line in (ROOT / "runs" / run / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            if line:
                record = json.loads(line)
                first.setdefault(record["group_id"], record)
        for line in (ROOT / "runs" / run / "groups.jsonl").read_text(encoding="utf-8").splitlines():
            if line:
                group = json.loads(line)
                by_char[group["character"]].append({"run": run, "group_id": group["group_id"],
                                                    "artists": group["artists"], "record": first[group["group_id"]]})
    return by_char


def make_record(group_id: str, index: int, base: dict, combo: list[dict], scene: dict, seed_key: str,
                blend: dict) -> dict:
    ident = base["identity"]
    identity_tags = [ident["gender"], "solo", base["character"]] \
        + ([anima_tag(ident["copyright"])] if ident.get("copyright") else []) \
        + [anima_tag(t) for t in ident["appearance"] + ident["outfit"]]
    record_id = f"{group_id}_{index:02d}"
    return {
        "record_id": record_id, "group_id": group_id, "character": base["character"],
        "character_danbooru": base.get("character_danbooru", ""), "artists": combo, "identity": ident,
        "rating": base["rating"], "quality": base["quality"], "scene": scene,
        "prompt": ", ".join([base["quality"], base["rating"], artist_prompt(combo), *identity_tags, *scene_tags(scene)]),
        "negative": base["negative"], "width": scene["width"], "height": scene["height"],
        "seed": int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "little"),
        "blend": blend,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--sources", nargs="+", default=["pilot", "grid1"])
    parser.add_argument("--levels", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--scheme", default="linear", choices=["linear", "floor"])
    parser.add_argument("--pairs-per-character", type=int, default=1)
    parser.add_argument("--images-per-pair", type=int, help="render this many of --levels per pair, drawn at random "
                        "(default: every level)")
    parser.add_argument("--test", type=int, help="weight-scheme check with N character pairs")
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    by_char = load_source_groups(args.sources)
    chars = sorted(c for c, groups in by_char.items() if len({json.dumps(g["artists"], sort_keys=True) for g in groups}) >= 2)
    if args.test:
        rng.shuffle(chars)
        chars = chars[:args.test]

    records, groups = [], []
    for n, char in enumerate(chars):
        options = by_char[char]
        pairs = [(a, b) for a in options for b in options
                 if json.dumps(a["artists"], sort_keys=True) != json.dumps(b["artists"], sort_keys=True)]
        rng.shuffle(pairs)
        for p, (a, b) in enumerate(pairs[:args.pairs_per_character]):
            group_id = f"b{n:05d}{p}"
            a_first = rng.random() < 0.5
            base = a["record"]
            blend = {"ref_run": a["run"], "ref_group": a["group_id"], "text_run": b["run"], "text_group": b["group_id"],
                     "ref_artists": a["artists"], "text_artists": b["artists"], "a_first": a_first}
            group_records = []
            if args.test:  # one scene and seed for the whole row
                scene = sample_scene(rng)
                for scheme in ("linear", "floor"):
                    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
                        if scheme == "floor" and s in (0.0, 1.0):
                            continue  # the pure ends are the same for both schemes
                        combo = blend_combo(a["artists"], b["artists"], s, scheme, a_first)
                        group_records.append(make_record(group_id, len(group_records), base, combo, scene,
                                                         f"{args.seed}:{group_id}", {**blend, "s": s, "scheme": scheme}))
            else:
                levels = sorted(rng.sample(args.levels, args.images_per_pair)) if args.images_per_pair else args.levels
                for s in levels:
                    scene = sample_scene(rng)
                    combo = blend_combo(a["artists"], b["artists"], s, args.scheme, a_first)
                    index = len(group_records)
                    group_records.append(make_record(group_id, index, base, combo, scene,
                                                     f"{args.seed}:{group_id}_{index}", {**blend, "s": s, "scheme": args.scheme}))
            records.extend(group_records)
            groups.append({"group_id": group_id, "character": char, "artists": a["artists"], "blend": blend,
                           "record_ids": [r["record_id"] for r in group_records]})

    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    hashes = {"manifest": write_jsonl(run_dir / "manifest.jsonl", records),
              "groups": write_jsonl(run_dir / "groups.jsonl", groups),
              "shard_0": write_jsonl(run_dir / "shard_0.jsonl", records)}
    summary = {"run_name": args.run_name, "groups": len(groups), "records": len(records),
               "characters": len({g["character"] for g in groups}), "levels": args.levels, "scheme": args.scheme,
               "test": args.test, "sources": args.sources, "sha256": hashes}
    (run_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "sha256"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
