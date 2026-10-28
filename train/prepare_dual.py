"""Training tensors for the two-reference (character slot + style slot) adapter.

Groups render one character in one artist combination; characters and combinations are each
reused across groups (a sparse character x style grid). For a target image of group (C, S):
  character slot  an image of C in another style S' (so the slot can only carry identity)
  style slot      an image of S on another character C' (so the slot can only carry style)
  same            another image of the same group, for "both from one image"
train_dual.py draws these per step; this script stores everything they need.

Held-out evaluation: per source, eval_characters characters (by hash, as prepare.py picks them),
plus every artist combination those characters were drawn in; no group touching either is trained on.

  python train/prepare_dual.py --source pilot=/workspace/data/pilot --source grid1=/workspace/data/grid1 \
      --out runs/train/dual_prep

Writes to --out (same layout as prepare.py, so sample.Prep reads it):
  index.json, contexts.safetensors, targets_NNN.safetensors, refs_NNN.safetensors
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "datagen"))
import anima_env  # noqa: E402
from catalog import anima_tag  # noqa: E402
from prepare import SCENE_KEYS, prompt_variants, read_results, split_groups, write_shards  # noqa: E402


def style_key(record: dict) -> str:
    return json.dumps(sorted(record["artists"], key=lambda a: a["tag"]), sort_keys=True)


def artist_prompt(combo: list[dict]) -> str:  # build_manifest.artist_prompt
    return ", ".join(a["tag"] if a["weight"] == 1.0 else f"({a['tag']}:{a['weight']})" for a in combo)


def dual_variants(record: dict, keep_range: tuple[float, float], seed: int) -> dict[str, str]:
    """prepare.py's variants plus the two text fallbacks for a single filled slot:
    with_artist (style from text, character from the reference) and with_character (the reverse)."""
    out = prompt_variants(record, keep_range, seed)
    ident = record["identity"]
    prefix = [record["quality"], record["rating"]]
    scene = [record["scene"][key] for key in SCENE_KEYS if record["scene"].get(key)]
    character = [record["character"]] + ([anima_tag(ident["copyright"])] if ident.get("copyright") else []) \
        + [anima_tag(t) for t in ident["appearance"] + ident["outfit"]]
    out["with_artist"] = ", ".join(prefix + [artist_prompt(record["artists"]), ident["gender"], "solo"] + scene)
    out["with_character"] = ", ".join(prefix + [ident["gender"], "solo"] + character + scene)
    return out


def connected_subset(groups: dict[str, list[dict]], limit: int) -> dict[str, list[dict]]:
    """Debug subsets: grow from the first group through shared characters and styles, so a small
    subset still has character-slot and style-slot partners."""
    by_char, by_style = defaultdict(list), defaultdict(list)
    for gid, rows in groups.items():
        by_char[rows[0]["character"]].append(gid)
        by_style[style_key(rows[0])].append(gid)
    keep, queue = [], deque([sorted(groups)[0]])
    seen = set(queue)
    while queue and len(keep) < limit:
        gid = queue.popleft()
        keep.append(gid)
        row = groups[gid][0]
        for nxt in by_char[row["character"]] + by_style[style_key(row)]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return {gid: groups[gid] for gid in keep}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train_dual.json")
    parser.add_argument("--source", action="append", required=True, metavar="NAME=DATA_DIR",
                        help="generation output dir with results_gpu*.jsonl; repeatable")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comfy-root", default="/workspace/ComfyUI")
    parser.add_argument("--max-groups", type=int, help="debug: a connected subset of N groups per source")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    args.out.mkdir(parents=True, exist_ok=True)

    # Records of every source; group ids differ per source (g..., c...), so they share one namespace.
    groups, data_dirs, held_out = {}, {}, set()
    for spec in args.source:
        name, data_dir = spec.split("=", 1)
        data_dirs[name] = data_dir
        by_group = defaultdict(list)
        results = sorted(p for p in Path(data_dir).glob("results_gpu*.jsonl") if not p.name.endswith(".log.jsonl"))
        for record in read_results(results):
            record["source"] = name
            by_group[record["group_id"]].append(record)
        source_groups = {gid: rows for gid, rows in by_group.items() if len(rows) >= 2}
        if args.max_groups:
            source_groups = connected_subset(source_groups, args.max_groups)
        source_held, _, _ = split_groups(source_groups, data_cfg["eval_characters"], 0)
        held_out |= source_held
        print(f"{name}: groups={len(source_groups)} records={sum(map(len, source_groups.values()))} "
              f"held_out_characters={len(source_held)}", flush=True)
        groups.update(source_groups)
    held_styles = {style_key(rows[0]) for rows in groups.values() if rows[0]["character"] in held_out}
    split = {gid: "eval" if rows[0]["character"] in held_out or style_key(rows[0]) in held_styles else "train"
             for gid, rows in groups.items()}
    records = [r for rows in groups.values() for r in rows]
    print(f"total groups={len(groups)} train={sum(s == 'train' for s in split.values())} "
          f"held-out characters={len(held_out)} styles={len(held_styles)}", flush=True)

    # Partners of every group: same character in another style, same style on another character.
    by_char, by_style = defaultdict(list), defaultdict(list)
    for gid, rows in groups.items():
        by_char[rows[0]["character"]].append(gid)
        by_style[style_key(rows[0])].append(gid)

    def partners(gid: str) -> dict[str, list[str]]:
        row = groups[gid][0]
        same_split = lambda g: split[g] == split[gid]  # noqa: E731  (eval refs never come from training groups)
        return {"character": [g for g in by_char[row["character"]] if g != gid and same_split(g)
                              and style_key(groups[g][0]) != style_key(row)],
                "style": [g for g in by_style[style_key(row)] if g != gid and same_split(g)
                          and groups[g][0]["character"] != row["character"]]}

    group_partners = {gid: partners(gid) for gid in groups}

    anima_env.setup_comfy(args.comfy_root)
    device = torch.device("cuda")
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)

    prompts: dict[str, int] = {}
    index_records = {}
    for record in records:
        variants = dual_variants(record, tuple(data_cfg["partial_keep"]), config["train"]["seed"])
        entry = {key: prompts.setdefault(text, len(prompts)) for key, text in variants.items()}
        entry["negative"] = prompts.setdefault(record["negative"], len(prompts))
        index_records[record["record_id"]] = {
            "group_id": record["group_id"], "source": record["source"], "character": record["character"],
            "style": style_key(record), "width": record["width"], "height": record["height"],
            "image": record["image"], "seed": record["seed"], "prompt_original": record["prompt"], "prompts": entry,
        }
    context_path = args.out / "contexts.safetensors"
    done_prompts = 0
    if context_path.exists():  # resume: prompts are deterministic, so a file with the same count is reusable
        with safe_open(str(context_path), "pt") as handle:
            done_prompts = handle.get_slice("offsets").get_shape()[0] - 1
    if done_prompts == len(prompts):
        print(f"contexts: reusing {context_path.name} ({len(prompts)} prompts)", flush=True)
    else:
        clip = anima_env.load_text_encoder(config["comfy"]["clip_name"])
        started = time.perf_counter()
        contexts = []
        for i, text in enumerate(prompts):
            contexts.append(anima_env.encode_prompt(clip, dm, text, device))
            if (i + 1) % 2000 == 0:
                print(f"text {i + 1}/{len(prompts)} {time.perf_counter() - started:.0f}s", flush=True)
        offsets = torch.tensor([0] + [c.shape[0] for c in contexts]).cumsum(0)
        save_file({"data": torch.cat(contexts).contiguous(), "offsets": offsets}, str(context_path))
        print(f"contexts: {len(prompts)} prompts, {int(offsets[-1])} tokens, {time.perf_counter() - started:.0f}s", flush=True)
        del clip, contexts
    del dm
    torch.cuda.empty_cache()

    targets = {}
    for record in records:
        latent = load_file(str(Path(data_dirs[record["source"]]) / record["latent"]))["latent"]
        latent = latent[0] if latent.shape[0] == 1 else latent
        targets[record["record_id"]] = (latent.unsqueeze(1) if latent.ndim == 3 else latent).to(torch.float16)
    target_files = write_shards(args.out / "targets", targets)
    del targets

    # Every image is letterboxed to each target size it can be paired with (own group and partners).
    sizes_for = defaultdict(set)
    for gid, rows in groups.items():
        sizes = {(r["width"], r["height"]) for r in rows}
        for other in [gid] + group_partners[gid]["character"] + group_partners[gid]["style"]:
            for r in groups[other]:
                sizes_for[r["record_id"]] |= sizes
    jobs = [(r, size) for r in records for size in sorted(sizes_for[r["record_id"]])]
    fill = int(data_cfg.get("letterbox_fill", 255))
    vae = anima_env.load_vae(config["comfy"]["vae_name"])

    def load(job):
        from PIL import Image
        record, (width, height) = job
        with Image.open(Path(data_dirs[record["source"]]) / record["image"]) as image:
            return job, anima_env.letterbox(image, width, height, fill)

    refs = {}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        # pool.map would queue every job at once and hold all loaded images (~3 MB each) in RAM
        # while the VAE lags behind; keep only a bounded window in flight.
        pending = deque(pool.submit(load, job) for job in jobs[:64])
        for i in range(len(jobs)):
            (record, (width, height)), image = pending.popleft().result()
            if i + 64 < len(jobs):
                pending.append(pool.submit(load, jobs[i + 64]))
            refs[f"{record['record_id']}@{width}x{height}"] = anima_env.vae_encode_image(vae, image).to(torch.float16)
            if (i + 1) % 1000 == 0:
                rate = (i + 1) / (time.perf_counter() - started)
                print(f"ref {i + 1}/{len(jobs)} {rate:.1f}/s eta {(len(jobs) - i - 1) / rate / 60:.0f} min", flush=True)
    ref_files = write_shards(args.out / "refs", refs)

    # Eval cases: target of group (C, S), character ref from (C, S'), style ref from (C', S).
    rng = random.Random(config["train"]["seed"])
    eval_cases = []
    for split_name, wanted in (("held_out", "eval"), ("seen", "train")):
        limit = data_cfg["eval_cases"] if split_name == "held_out" else data_cfg["seen_eval_cases"]
        used_chars = set()
        candidates = sorted(g for g in groups if split[g] == wanted)
        if split_name == "seen":
            rng.shuffle(candidates)
        for gid in candidates:
            row = groups[gid][0]
            if split_name == "held_out" and row["character"] not in held_out:
                continue  # an eval group only because its style is held out
            p = group_partners[gid]
            if row["character"] in used_chars or not p["character"] or not p["style"]:
                continue
            used_chars.add(row["character"])
            target = sorted(groups[gid], key=lambda r: r["record_id"])[0]
            eval_cases.append({
                "split": split_name, "group_id": gid, "character": row["character"],
                "artists": artist_prompt(row["artists"]), "target": target["record_id"],
                "character_ref": sorted(r["record_id"] for r in groups[p["character"][0]])[0],
                "style_ref": sorted(r["record_id"] for r in groups[p["style"][0]])[0],
                "width": target["width"], "height": target["height"], "seed": 777 + len(eval_cases)})
            if len([c for c in eval_cases if c["split"] == split_name]) >= limit:
                break

    index = {
        "created": time.time(), "config": config, "data_dirs": data_dirs, "data_dir": next(iter(data_dirs.values())),
        "records": index_records,
        "groups": {gid: {"split": split[gid], "source": rows[0]["source"], "character": rows[0]["character"],
                         "style": style_key(rows[0]), "record_ids": sorted(r["record_id"] for r in rows),
                         "partners": group_partners[gid]}
                   for gid, rows in groups.items()},
        "held_out_characters": sorted(held_out), "held_out_styles": sorted(held_styles),
        "prompts": list(prompts), "eval_cases": eval_cases,
        "files": {"targets": target_files, "refs": ref_files},
    }
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done: refs={len(refs)} eval_cases={len(eval_cases)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
