"""Tensors for style-blend training, on top of a prepare_dual.py prep.

A blend record (build_blend_manifest.py) is character C drawn with artist combo A scaled by s and B
by 1 - s. Training reads it with the reference from group (C, A), the prompt naming B's artists, and
strength s. "Conflict" items are the s = 1 end without new images: a target from group (C, A), the
prompt naming B, the reference another image of (C, A) — the reference's style must win.

Output (--out): index.json, contexts.safetensors (only the new prompts), targets_NNN.safetensors
(blend latents), refs_NNN.safetensors (group-A images letterboxed to each blend record's size,
key "<record_id>@<W>x<H>"). Conflict refs and the s = 0 / s = 1 ends use the base prep's tensors.

  python train/prepare_blend.py --base runs/train/dual_r32_prep --blend-run blend1 \
      --blend-data /workspace/data/blend1 --out runs/train/blend1_prep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402
from prepare import SCENE_KEYS, read_results, write_shards  # noqa: E402
from prepare_dual import artist_prompt  # noqa: E402


def text_artist_prompt(record: dict, artists: list[dict]) -> str:
    """with_artist (prepare_dual.dual_variants) with another artist combo: quality, rating, artists, gender, solo, scene."""
    scene = [record["scene"][key] for key in SCENE_KEYS if record["scene"].get(key)]
    return ", ".join([record["quality"], record["rating"], artist_prompt(artists), record["identity"]["gender"], "solo"] + scene)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="prepare_dual.py output the blend groups refer to")
    parser.add_argument("--blend-run", required=True, help="runs/<name> with the blend manifest")
    parser.add_argument("--blend-data", type=Path, required=True, help="generation output dir of the blend run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-cases", type=int, default=8)
    parser.add_argument("--comfy-root", default="/workspace/ComfyUI")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base = json.loads((args.base / "index.json").read_text(encoding="utf-8"))
    config = base["config"]
    base_groups, base_records = base["groups"], base["records"]
    manifest = {}
    for line in (ROOT / "runs" / args.blend_run / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            manifest[row["record_id"]] = row
    results = read_results(sorted(p for p in args.blend_data.glob("results_gpu*.jsonl") if not p.name.endswith(".log.jsonl")))
    # Full metadata (scene, quality, rating) of the base records, for the conflict prompts.
    source_rows = {}
    for name, data_dir in base["data_dirs"].items():
        for row in read_results(sorted(p for p in Path(data_dir).glob("results_gpu*.jsonl") if not p.name.endswith(".log.jsonl"))):
            source_rows[row["record_id"]] = row

    prompts: dict[str, int] = {}
    blend_records, conflicts, skipped = {}, [], 0
    for row in results:
        blend = manifest[row["record_id"]]["blend"]
        ref_group, text_group = blend["ref_group"], blend["text_group"]
        if ref_group not in base_groups or text_group not in base_groups:
            skipped += 1  # a source group the base prep dropped (fewer than two finished images)
            continue
        split = "eval" if "eval" in (base_groups[ref_group]["split"], base_groups[text_group]["split"]) else "train"
        record = {**manifest[row["record_id"]], **row}
        blend_records[row["record_id"]] = {
            "group_id": row["group_id"], "character": row["character"], "split": split, "s": blend["s"],
            "width": row["width"], "height": row["height"], "image": row["image"], "latent": row["latent"],
            "ref_group": ref_group, "text_group": text_group,
            "prompts": {"text_artist": prompts.setdefault(text_artist_prompt(record, blend["text_artists"]), len(prompts)),
                        "negative": prompts.setdefault(record["negative"], len(prompts))},
        }
        for target in base_groups[ref_group]["record_ids"]:  # s = 1: the reference's own style despite B's tags
            src = source_rows[target]
            conflicts.append({"target": target, "split": split, "ref_group": ref_group, "text_group": text_group,
                              "prompt": prompts.setdefault(text_artist_prompt(src, blend["text_artists"]), len(prompts))})
    print(f"blend records={len(blend_records)} (train {sum(r['split'] == 'train' for r in blend_records.values())}) "
          f"conflict targets={len(conflicts)} skipped={skipped} new prompts={len(prompts)}", flush=True)

    anima_env.setup_comfy(args.comfy_root)
    device = torch.device("cuda")
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)
    clip = anima_env.load_text_encoder(config["comfy"]["clip_name"])
    started = time.perf_counter()
    contexts = [anima_env.encode_prompt(clip, dm, text, device) for text in prompts]
    offsets = torch.tensor([0] + [c.shape[0] for c in contexts]).cumsum(0)
    save_file({"data": torch.cat(contexts).contiguous(), "offsets": offsets}, str(args.out / "contexts.safetensors"))
    print(f"contexts: {len(prompts)} prompts in {time.perf_counter() - started:.0f}s", flush=True)
    del clip, contexts, dm
    torch.cuda.empty_cache()

    targets = {}
    for rid, rec in blend_records.items():
        latent = load_file(str(args.blend_data / rec["latent"]))["latent"]
        latent = latent[0] if latent.shape[0] == 1 else latent
        targets[rid] = (latent.unsqueeze(1) if latent.ndim == 3 else latent).to(torch.float16)
    target_files = write_shards(args.out / "targets", targets)
    del targets

    # Group-A images at every blend record's size (the base prep has them only at A's partner sizes).
    jobs = sorted({(ref, rec["width"], rec["height"]) for rec in blend_records.values()
                   for ref in base_groups[rec["ref_group"]]["record_ids"]})
    fill = int(config["data"].get("letterbox_fill", 255))
    vae = anima_env.load_vae(config["comfy"]["vae_name"])

    def load(job):
        from PIL import Image
        rid, width, height = job
        record = base_records[rid]
        with Image.open(Path(base["data_dirs"][record["source"]]) / record["image"]) as image:
            return job, anima_env.letterbox(image, width, height, fill)

    refs, started = {}, time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = deque(pool.submit(load, job) for job in jobs[:64])  # bounded, see prepare_dual.py
        for i in range(len(jobs)):
            (rid, width, height), image = pending.popleft().result()
            if i + 64 < len(jobs):
                pending.append(pool.submit(load, jobs[i + 64]))
            refs[f"{rid}@{width}x{height}"] = anima_env.vae_encode_image(vae, image).to(torch.float16)
            if (i + 1) % 1000 == 0:
                print(f"ref {i + 1}/{len(jobs)} {(i + 1) / (time.perf_counter() - started):.1f}/s", flush=True)
    ref_files = write_shards(args.out / "refs", refs)

    eval_cases = sorted((rid for rid, r in blend_records.items() if r["split"] == "eval"),
                        key=lambda rid: base_groups[blend_records[rid]["ref_group"]]["character"])[:args.eval_cases]
    index = {"created": time.time(), "base": str(args.base), "blend_run": args.blend_run, "data_dir": str(args.blend_data),
             "records": blend_records, "conflicts": conflicts, "prompts": list(prompts), "eval_cases": eval_cases,
             "files": {"targets": target_files, "refs": ref_files}}
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done: blend targets={len(blend_records)} refs={len(refs)} conflicts={len(conflicts)} eval cases={len(eval_cases)}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
