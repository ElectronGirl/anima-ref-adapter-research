"""Turn generated groups into training tensors for the reference adapter.

Each group holds several images of one character in one artist style, each in a different scene.
Training pairs are (reference = one image of the group, target = another image of the group):
the target's prompt keeps only scene tags, so character and style must come from the reference.

Writes to --out:
  index.json             records, group split (train / held-out characters), prompts, eval cases
  contexts.safetensors   post-llm_adapter text contexts for every prompt variant (bf16, unpadded)
  targets_NNN.safetensors  raw target latents copied from the generation output
  refs_NNN.safetensors   raw VAE latents of every image letterboxed to each target size in its group
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "datagen"))
import anima_env  # noqa: E402
from catalog import anima_tag  # noqa: E402

SHARD_SIZE = 1000
SCENE_KEYS = ("framing", "camera", "pose", "expression", "background", "lighting")  # build_manifest.scene_tags


def read_results(paths: list[Path]) -> list[dict]:
    rows = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "complete" and row.get("latent"):
                for key in ("image", "latent"):  # results written on Windows use backslashes
                    row[key] = row[key].replace("\\", "/")
                rows[row["record_id"]] = row
    return sorted(rows.values(), key=lambda r: r["record_id"])


def prompt_variants(record: dict, keep_range: tuple[float, float], seed: int) -> dict[str, str]:
    prefix = [record["quality"], record["rating"], record["identity"]["gender"], "solo"]
    scene = [record["scene"][key] for key in SCENE_KEYS if record["scene"].get(key)]
    pool = [anima_tag(t) for t in record["identity"]["appearance"] + record["identity"]["outfit"]]
    rng = random.Random(f"{seed}:{record['record_id']}")
    keep = max(1, round(len(pool) * rng.uniform(*keep_range))) if pool else 0
    kept = set(rng.sample(range(len(pool)), keep))
    partial = [tag for i, tag in enumerate(pool) if i in kept]
    return {
        "stripped": ", ".join(prefix + scene),
        "partial": ", ".join(prefix + partial + scene),
        "minimal": ", ".join(prefix),
    }


def split_groups(groups: dict[str, list[dict]], eval_characters: int, seen_eval_groups: int) -> tuple[set, list, list]:
    characters = sorted({g[0]["character"] for g in groups.values()},
                        key=lambda c: hashlib.sha1(c.encode("utf-8")).hexdigest())
    held_out = set(characters[:eval_characters])
    eval_groups, seen_groups = [], []
    used_eval_chars = set()
    for gid in sorted(groups):
        character = groups[gid][0]["character"]
        if character in held_out and character not in used_eval_chars:
            eval_groups.append(gid)
            used_eval_chars.add(character)
        elif character not in held_out and len(seen_groups) < seen_eval_groups:
            seen_groups.append(gid)
    return held_out, eval_groups, seen_groups


def write_shards(prefix: Path, tensors: dict[str, torch.Tensor]) -> list[str]:
    names = []
    keys = sorted(tensors)
    for start in range(0, len(keys), SHARD_SIZE):
        name = f"{prefix.name}_{start // SHARD_SIZE:03d}.safetensors"
        save_file({k: tensors[k].contiguous() for k in keys[start:start + SHARD_SIZE]}, str(prefix.parent / name))
        names.append(name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.json")
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="results_gpu*.jsonl from the generator")
    parser.add_argument("--data-dir", type=Path, required=True, help="generation output dir (image/latent paths are relative to it)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comfy-root", default="/workspace/ComfyUI")
    parser.add_argument("--max-groups", type=int, help="debug: only use the first N groups")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    args.out.mkdir(parents=True, exist_ok=True)

    records = read_results(args.results)
    by_group = defaultdict(list)
    for record in records:
        by_group[record["group_id"]].append(record)
    groups = {gid: rows for gid, rows in by_group.items() if len(rows) >= 2}
    if args.max_groups:
        groups = {gid: groups[gid] for gid in sorted(groups)[: args.max_groups]}
    records = [r for rows in groups.values() for r in rows]
    held_out, eval_groups, seen_groups = split_groups(groups, data_cfg["eval_characters"], data_cfg["seen_eval_groups"])
    print(f"records={len(records)} groups={len(groups)} held_out_characters={len(held_out)} "
          f"eval_groups={len(eval_groups)} seen_eval_groups={len(seen_groups)}", flush=True)

    anima_env.setup_comfy(args.comfy_root)
    device = torch.device("cuda")
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)

    # Text contexts for every prompt variant (deduplicated) and the negatives.
    prompts: dict[str, int] = {}
    index_records = {}
    for record in records:
        variants = prompt_variants(record, tuple(data_cfg["partial_keep"]), config["train"]["seed"])
        entry = {key: prompts.setdefault(text, len(prompts)) for key, text in variants.items()}
        entry["negative"] = prompts.setdefault(record["negative"], len(prompts))
        index_records[record["record_id"]] = {
            "group_id": record["group_id"], "character": record["character"], "width": record["width"],
            "height": record["height"], "image": record["image"], "seed": record["seed"],
            "prompt_original": record["prompt"], "prompts": entry,
        }
    clip = anima_env.load_text_encoder(config["comfy"]["clip_name"])
    started = time.perf_counter()
    contexts = []
    for i, text in enumerate(prompts):
        contexts.append(anima_env.encode_prompt(clip, dm, text, device))
        if (i + 1) % 1000 == 0:
            print(f"text {i + 1}/{len(prompts)} {time.perf_counter() - started:.0f}s", flush=True)
    offsets = torch.tensor([0] + [c.shape[0] for c in contexts]).cumsum(0)
    save_file({"data": torch.cat(contexts).contiguous(), "offsets": offsets}, str(args.out / "contexts.safetensors"))
    print(f"contexts: {len(prompts)} prompts, {int(offsets[-1])} tokens, {time.perf_counter() - started:.0f}s", flush=True)
    del clip, contexts

    # Target latents, copied into shards so training reads one place.
    targets = {}
    for record in records:
        latent = load_file(str(args.data_dir / record["latent"]))["latent"]
        latent = latent[0] if latent.shape[0] == 1 else latent  # saved as a batch of one
        targets[record["record_id"]] = (latent.unsqueeze(1) if latent.ndim == 3 else latent).to(torch.float16)
    target_files = write_shards(args.out / "targets", targets)
    del targets

    # Reference latents: each image letterboxed to every target size it is paired with.
    vae = anima_env.load_vae(config["comfy"]["vae_name"])
    jobs = []
    for rows in groups.values():
        sizes = sorted({(r["width"], r["height"]) for r in rows})
        jobs.extend((r, size) for r in rows for size in sizes)
    fill = int(data_cfg.get("letterbox_fill", 255))

    def load(job):
        from PIL import Image
        record, (width, height) = job
        with Image.open(args.data_dir / record["image"]) as image:
            return job, anima_env.letterbox(image, width, height, fill)

    refs = {}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, ((record, (width, height)), image) in enumerate(pool.map(load, jobs)):
            refs[f"{record['record_id']}@{width}x{height}"] = anima_env.vae_encode_image(vae, image).to(torch.float16)
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.perf_counter() - started)
                print(f"ref {i + 1}/{len(jobs)} {rate:.1f}/s", flush=True)
    ref_files = write_shards(args.out / "refs", refs)

    # Eval cases: reference = first image of the group, target scene = next image with a different scene.
    eval_cases = []
    for split, gids in (("held_out", eval_groups), ("seen", seen_groups)):
        for gid in gids:
            rows = sorted(groups[gid], key=lambda r: r["record_id"])
            ref, target = rows[0], rows[1]
            eval_cases.append({"split": split, "group_id": gid, "character": ref["character"],
                               "ref": ref["record_id"], "target": target["record_id"],
                               "width": target["width"], "height": target["height"], "seed": 777 + len(eval_cases)})

    index = {
        "created": time.time(),
        "config": config,
        "data_dir": str(args.data_dir),
        "records": index_records,
        "groups": {gid: {"split": "eval" if rows[0]["character"] in held_out else "train",
                         "character": rows[0]["character"], "record_ids": sorted(r["record_id"] for r in rows)}
                   for gid, rows in groups.items()},
        "held_out_characters": sorted(held_out),
        "prompts": list(prompts),
        "eval_cases": eval_cases,
        "files": {"targets": target_files, "refs": ref_files},
    }
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    train_groups = sum(1 for g in index["groups"].values() if g["split"] == "train")
    print(f"done: train_groups={train_groups} refs={len(refs)} eval_cases={len(eval_cases)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
