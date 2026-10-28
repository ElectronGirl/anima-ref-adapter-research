"""Sample (character x artist-combo) groups and per-image scenes into a generation manifest.

Each group renders one character in one artist combination several times with independently
sampled framing, camera, pose, expression, background and lighting. Characters and artist
combinations are each reused across groups, so the dataset also contains same-character /
different-style and same-style / different-character pairs. With uses_per_character =
uses_per_combo = 3 and two images per group this is a sparse character x style grid.

Output (under runs/<run_name>/):
  manifest.jsonl            every record
  shard_<k>.jsonl           records of worker k (whole groups only)
  groups.jsonl              one line per group (character, artists, record ids)
  manifest_summary.json     counts and hashes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vocab  # noqa: E402
from catalog import anima_tag, load_artists, load_characters, load_name_set  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def weighted(rng: random.Random, items: list[tuple[str, float]]) -> str:
    return rng.choices([item for item, _ in items], weights=[w for _, w in items], k=1)[0]


def prune_generic(tags: list[str]) -> list[str]:
    """Drop a tag when a more specific kept tag ends with it (dress < white_dress)."""
    return [t for t in tags if not any(o != t and o.endswith("_" + t) for o in tags)]


def load_identity(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        out[row["character"]] = row
    return out


def eligible_characters(cfg: dict) -> list[dict]:
    chars = load_characters(ROOT / cfg["catalog"], cfg["min_post_count"])
    identity = load_identity(ROOT / cfg["identity_tags"])
    blocked = [s.casefold() for s in cfg.get("exclude_substrings", [])]
    max_q = int(cfg.get("max_parenthesized_qualifiers", 1))
    out = []
    for char in chars:
        name = char["danbooru_tag"].replace("_", " ").casefold()
        info = identity.get(char["danbooru_tag"])
        if any(s in name for s in blocked) or len(re.findall(r"\(", char["danbooru_tag"])) > max_q:
            continue
        # Anima renders a character reliably only with its real appearance tags; skip
        # characters whose single-character posts give no stable gender or too few traits.
        if info is None or info["gender"] is None or len(info["appearance"]) < cfg["min_appearance_tags"]:
            continue
        char["identity"] = {
            "gender": info["gender"],
            "copyright": info["copyright"],
            "appearance": prune_generic([t for t, _ in info["appearance"]]),
            "outfit": prune_generic([t for t, _ in info["outfit"]]),
        }
        out.append(char)
    return out


def validate_scene_vocab(tag_set: set[str]) -> None:
    pools = [list(vocab.FRAMINGS), [c for c, _ in vocab.CAMERA], vocab.POSES, vocab.FULL_BODY_POSES,
             [e for e, _ in vocab.EXPRESSIONS], [b for b, _ in vocab.BACKGROUNDS], [x for x, _ in vocab.LIGHTING]]
    unknown = sorted({t.strip() for pool in pools for item in pool for t in item.split(",")
                      if t.strip() and t.strip().replace(" ", "_") not in tag_set})
    if unknown:
        raise ValueError(f"scene vocabulary has tags that are not Danbooru tags: {unknown}")


def popularity_sample(rng: random.Random, pool: list[dict], k: int) -> list[dict]:
    """Sample k distinct items with probability ~ sqrt(post_count)."""
    keyed = sorted(pool, key=lambda item: rng.random() ** (1.0 / item["post_count"] ** 0.5), reverse=True)
    return keyed[:k]


def artist_combo(rng: random.Random, artists: list[dict], cfg: dict) -> list[dict]:
    sizes = cfg["combo_size_weights"]
    size = int(rng.choices(list(sizes), weights=list(sizes.values()), k=1)[0])
    chosen = popularity_sample(rng, artists, size)
    lo, hi = cfg["mix_weight_range"]
    combo = []
    for artist in chosen:
        weight = 1.0 if size == 1 else round(rng.uniform(lo, hi), 2)
        combo.append({"tag": artist["tag"], "weight": weight})
    return combo


def artist_prompt(combo: list[dict]) -> str:
    parts = []
    for artist in combo:
        tag = artist["tag"]
        parts.append(tag if artist["weight"] == 1.0 else f"({tag}:{artist['weight']})")
    return ", ".join(parts)


def sample_scene(rng: random.Random) -> dict:
    framing = rng.choices(list(vocab.FRAMINGS), weights=[v[0] for v in vocab.FRAMINGS.values()], k=1)[0]
    width, height = rng.choice(vocab.FRAMINGS[framing][1])
    poses = vocab.POSES + (vocab.FULL_BODY_POSES if framing.startswith(("full body", "cowboy")) else [])
    return {
        "framing": framing,
        "camera": weighted(rng, vocab.CAMERA),
        "pose": rng.choice(poses),
        "expression": weighted(rng, vocab.EXPRESSIONS),
        "background": weighted(rng, vocab.BACKGROUNDS),
        "lighting": weighted(rng, vocab.LIGHTING),
        "width": width,
        "height": height,
    }


def scene_tags(scene: dict) -> list[str]:
    keys = ("framing", "camera", "pose", "expression", "background", "lighting")
    return [scene[key] for key in keys if scene[key]]


def build(config: dict) -> tuple[list[dict], list[dict]]:
    rng = random.Random(config["seed"])
    validate_scene_vocab(set(pl.read_parquet(ROOT / config["tag_taxonomy"], columns=["tag"])["tag"].to_list()))
    chars = eligible_characters(config["characters"])
    # Characters of earlier runs stay out (their held-out eval characters must remain unseen).
    used = {json.loads(line)["character"] for run in config["characters"].get("exclude_runs", [])
            for line in (ROOT / "runs" / run / "groups.jsonl").read_text(encoding="utf-8").splitlines() if line}
    chars = [c for c in chars if c["tag"] not in used]
    artists = load_artists(ROOT / config["artists"]["catalog"], config["artists"]["min_post_count"])
    adult_allow = load_name_set(ROOT / config["rating"]["sensitive_allowlist"])
    n_groups = int(config["groups"])
    char_uses = int(config["characters"]["uses_per_character"])
    combo_uses = int(config["artists"]["uses_per_combo"])

    # Characters and artist combos are each reused, then paired so no (char, combo) repeats.
    char_slots = [c for c in popularity_sample(rng, chars, -(-n_groups // char_uses)) for _ in range(char_uses)][:n_groups]
    combos = [artist_combo(rng, artists, config["artists"]) for _ in range(-(-n_groups // combo_uses))]
    combo_slots = [c for c in combos for _ in range(combo_uses)][:n_groups]
    rng.shuffle(char_slots)
    rng.shuffle(combo_slots)
    pair_key = lambda i: (char_slots[i]["tag"], json.dumps(combo_slots[i], sort_keys=True))  # noqa: E731
    seen = set()
    for i in range(n_groups):
        for j in range(i + 1, n_groups):  # swap in a later combo until this pair is new
            if pair_key(i) not in seen:
                break
            combo_slots[i], combo_slots[j] = combo_slots[j], combo_slots[i]
        seen.add(pair_key(i))

    quality = config["prompt"]["quality"]
    negative = config["prompt"]["negative"]
    records, groups = [], []
    for g, (char, combo) in enumerate(zip(char_slots, combo_slots)):
        group_id = f"{config.get('id_prefix', 'g')}{g:06d}"
        sensitive_ok = char["danbooru_tag"] in adult_allow
        group_records = []
        for i in range(int(config["images_per_group"])):
            scene = sample_scene(rng)
            rating = "sensitive" if sensitive_ok and rng.random() < config["rating"]["sensitive_ratio"] else config["rating"]["default"]
            record_id = f"{group_id}_{i:02d}"
            ident = char["identity"]
            identity_tags = [ident["gender"], "solo", char["tag"]] + ([anima_tag(ident["copyright"])] if ident["copyright"] else []) \
                + [anima_tag(t) for t in ident["appearance"] + ident["outfit"]]
            prompt = ", ".join([quality, rating, artist_prompt(combo), *identity_tags, *scene_tags(scene)])
            seed = int.from_bytes(hashlib.sha256(f"{config['seed']}:{record_id}".encode()).digest()[:8], "little")
            group_records.append({
                "record_id": record_id,
                "group_id": group_id,
                "character": char["tag"],
                "character_danbooru": char["danbooru_tag"],
                "artists": combo,
                "identity": ident,
                "rating": rating,
                "quality": quality,
                "scene": scene,
                "prompt": prompt,
                "negative": negative,
                "width": scene["width"],
                "height": scene["height"],
                "seed": seed,
            })
        records.extend(group_records)
        groups.append({"group_id": group_id, "character": char["tag"], "artists": combo,
                       "record_ids": [r["record_id"] for r in group_records]})
    return records, groups


def write_jsonl(path: Path, rows: list[dict]) -> str:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "datagen.json")
    parser.add_argument("--groups", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--images-per-group", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("groups", "workers", "run_name", "images_per_group"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)

    records, groups = build(config)
    run_dir = ROOT / "runs" / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    hashes = {"manifest": write_jsonl(run_dir / "manifest.jsonl", records),
              "groups": write_jsonl(run_dir / "groups.jsonl", groups)}
    workers = int(config["workers"])
    by_group = {}
    for record in records:
        by_group.setdefault(record["group_id"], []).append(record)
    group_ids = sorted(by_group)
    for k in range(workers):
        shard = [r for gid in group_ids[k::workers] for r in by_group[gid]]
        hashes[f"shard_{k}"] = write_jsonl(run_dir / f"shard_{k}.jsonl", shard)
    summary = {
        "run_name": config["run_name"], "groups": len(groups), "records": len(records),
        "unique_characters": len({g["character"] for g in groups}),
        "unique_artist_combos": len({json.dumps(g["artists"], sort_keys=True) for g in groups}),
        "unique_artists": len({a["tag"] for g in groups for a in g["artists"]}),
        "resolutions": {f"{w}x{h}": sum(1 for r in records if (r["width"], r["height"]) == (w, h))
                        for w, h in sorted({(r["width"], r["height"]) for r in records})},
        "ratings": {k: sum(1 for r in records if r["rating"] == k) for k in sorted({r["rating"] for r in records})},
        "workers": workers, "sha256": hashes, "config": config,
    }
    (run_dir / "manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "sha256")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
