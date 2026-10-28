"""Extra higher-resolution images of existing groups (about 1.3 MP instead of 1 MP).

Each chosen group (character + artist combo of an earlier run) gets new scenes rendered at the scaled
version of its framing's resolution. Records keep the source group id, so training can pair them with
that group's character partners like any other image of the group. Lower resolutions need no new
renders: they come from downscaling existing images during training.

  python datagen/build_hires_manifest.py --run-name hires1 --sources pilot grid1 --groups 1500 \
      --index runs/train/dual_r32_prep/index.json

--index (a prepare_dual.py index) limits the choice to groups it kept, with a character partner, and
keeps its train/eval split proportions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_manifest import artist_prompt, sample_scene, scene_tags, write_jsonl  # noqa: E402
from catalog import anima_tag  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# ~1.125x per side, multiples of 64: (832, 1216) -> (960, 1408) etc.
HIRES = {(1024, 1024): (1152, 1152), (896, 1152): (1024, 1280), (832, 1216): (960, 1408), (1216, 832): (1408, 960)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--sources", nargs="+", default=["pilot", "grid1"])
    parser.add_argument("--groups", type=int, default=1500)
    parser.add_argument("--images-per-group", type=int, default=1)
    parser.add_argument("--index", type=Path, help="prepare_dual.py index.json: only groups it kept with a character partner")
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    first = {}
    for run in args.sources:
        for line in (ROOT / "runs" / run / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            if line:
                record = json.loads(line)
                first.setdefault(record["group_id"], record)
    candidates = sorted(first)
    if args.index:
        index = json.loads(args.index.read_text(encoding="utf-8"))
        candidates = [g for g in candidates if g in index["groups"] and index["groups"][g]["partners"]["character"]]
    rng.shuffle(candidates)
    chosen = candidates[:args.groups]

    records, groups = [], []
    for gid in chosen:
        base = first[gid]
        ident = base["identity"]
        identity_tags = [ident["gender"], "solo", base["character"]] \
            + ([anima_tag(ident["copyright"])] if ident.get("copyright") else []) \
            + [anima_tag(t) for t in ident["appearance"] + ident["outfit"]]
        ids = []
        for i in range(args.images_per_group):
            scene = sample_scene(rng)
            scene["width"], scene["height"] = HIRES[(scene["width"], scene["height"])]
            record_id = f"{gid}_h{i}"
            records.append({
                "record_id": record_id, "group_id": gid, "character": base["character"],
                "character_danbooru": base.get("character_danbooru", ""), "artists": base["artists"], "identity": ident,
                "rating": base["rating"], "quality": base["quality"], "scene": scene,
                "prompt": ", ".join([base["quality"], base["rating"], artist_prompt(base["artists"]), *identity_tags,
                                     *scene_tags(scene)]),
                "negative": base["negative"], "width": scene["width"], "height": scene["height"],
                "seed": int.from_bytes(hashlib.sha256(f"{args.seed}:{record_id}".encode()).digest()[:8], "little"),
                "hires_of": gid,
            })
            ids.append(record_id)
        groups.append({"group_id": gid, "character": base["character"], "artists": base["artists"], "record_ids": ids})

    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    hashes = {"manifest": write_jsonl(run_dir / "manifest.jsonl", records),
              "groups": write_jsonl(run_dir / "groups.jsonl", groups),
              "shard_0": write_jsonl(run_dir / "shard_0.jsonl", records)}
    summary = {"run_name": args.run_name, "groups": len(groups), "records": len(records), "sources": args.sources,
               "resolutions": {f"{w}x{h}": sum(1 for r in records if (r["width"], r["height"]) == (w, h))
                               for w, h in sorted({(r["width"], r["height"]) for r in records})},
               "sha256": hashes}
    (run_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "sha256"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
