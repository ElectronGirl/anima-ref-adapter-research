"""Render showcase images through a running ComfyUI with the adapter nodes (for the README / model card).

  python deploy/showcase.py --comfy http://127.0.0.1:8199 --out runs/showcase/images

Each reference is drawn in several scenes from scene-only prompts with the basic workflow
(Euler, text cfg 4.5, reference guidance 4.5 + APG). Writes the images and showcase.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUALITY = "masterpiece, best quality, score_7, newest, highres, absurdres, safe"
REFERENCES = {
    "man": ("showcase_man.png", "1boy"),
    "girl": ("showcase_girl.png", "1girl"),
}
SCENES = {
    "man": [
        ("night_rain", "full body, walking, hands in pockets, city, street, night, rain", (896, 1152)),
        ("cafe", "upper body, sitting, holding cup, drinking, cafe, indoors, window, sunlight", (896, 1152)),
        ("beach", "cowboy shot, looking at viewer, smile, beach, ocean, sunset", (1216, 832)),
        ("sakura", "portrait, from side, looking afar, cherry blossoms, petals", (896, 1152)),
        ("library", "upper body, reading, holding book, library, bookshelf", (1024, 1024)),
        ("snow", "full body, standing, snow, forest, winter", (832, 1216)),
    ],
    "girl": [
        ("neon", "full body, walking, smile, city, street, night, neon lights", (832, 1216)),
        ("cake", "upper body, sitting, eating, cake, cafe, indoors", (1024, 1024)),
        ("beach", "cowboy shot, looking back, smile, beach, ocean, sunset", (1216, 832)),
        ("sakura", "portrait, closed eyes, smile, cherry blossoms, petals, wind", (896, 1152)),
        ("classroom", "full body, sitting, on chair, crossed legs, classroom, sunlight", (896, 1152)),
        ("umbrella", "upper body, holding umbrella, rain, street, looking at viewer", (896, 1152)),
    ],
}


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", default="http://127.0.0.1:8199")
    parser.add_argument("--comfy-output", type=Path, required=True, help="the ComfyUI instance's output directory")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "showcase" / "images")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--only", nargs="*", help="ref:scene ids to (re)render")
    parser.add_argument("--lora", action="append", default=[], metavar="NAME:STRENGTH",
                        help="extra LoRA in front of the adapter (model strength; clip strength 1), repeatable")
    parser.add_argument("--pag", metavar="SCALE,START,END", help="add Anima Safe PAG on block 18, e.g. 1.3,0.2,0.9")
    args = parser.parse_args()
    base = json.loads((ROOT / "runs" / "try" / "workflow_api.json").read_text(encoding="utf-8"))
    model, clip = ["1", 0], ["2", 0]
    for i, spec in enumerate(args.lora):
        name, strength = spec.rsplit(":", 1)
        base[f"lora{i}"] = {"class_type": "LoraLoader", "inputs": {
            "lora_name": name, "strength_model": float(strength), "strength_clip": 1.0, "model": model, "clip": clip}}
        model, clip = [f"lora{i}", 0], [f"lora{i}", 1]
    base["4"]["inputs"]["model"] = model
    base["9"]["inputs"]["clip"] = base["10"]["inputs"]["clip"] = clip
    if args.pag:
        scale, start, end = (float(v) for v in args.pag.split(","))
        base["20"] = {"class_type": "AnimaSafePAG", "inputs": {
            "scale": scale, "block_indices": "18", "perturbation_strength": 0.75, "head_indices": "",
            "start_percent": start, "end_percent": end, "rescale": 0.2, "rescale_mode": "full", "model": ["6", 0]}}
        base["11"]["inputs"]["model"] = base["13"]["inputs"]["model"] = ["20", 0]
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for ref_id, scenes in SCENES.items():
        image, gender = REFERENCES[ref_id]
        for index, (scene_id, scene, (width, height)) in enumerate(scenes):
            name = f"{ref_id}_{scene_id}"
            if args.only and name not in args.only:
                continue
            api = json.loads(json.dumps(base))
            api["7"]["inputs"]["image"] = image
            api["8"]["inputs"].update({"width": width, "height": height})
            api["9"]["inputs"]["text"] = f"{QUALITY}, {gender}, solo, {scene}"
            api["13"]["inputs"]["steps"] = args.steps
            api["14"]["inputs"]["noise_seed"] = args.seed + index
            api["17"]["inputs"]["filename_prefix"] = f"showcase/{name}"
            prompt_id = post(f"{args.comfy}/prompt", {"prompt": api})["prompt_id"]
            jobs.append({"name": name, "ref": ref_id, "prompt": api["9"]["inputs"]["text"], "scene": scene,
                         "width": width, "height": height, "seed": args.seed + index, "prompt_id": prompt_id,
                         "loras": args.lora, "pag": args.pag, "steps": args.steps})
    print(f"queued {len(jobs)}", flush=True)
    for job in jobs:
        while True:
            entry = get(f"{args.comfy}/history/{job['prompt_id']}").get(job["prompt_id"])
            if entry and entry.get("status", {}).get("completed") or entry and entry["status"].get("status_str") == "error":
                break
            time.sleep(3)
        files = [i for o in entry.get("outputs", {}).values() for i in o.get("images", []) if i.get("type") == "output"]
        if not files:
            print(f"{job['name']}: failed", flush=True)
            continue
        src = args.comfy_output / files[0]["subfolder"] / files[0]["filename"]
        shutil.copy(src, args.out / f"{job['name']}.png")
        job["file"] = f"{job['name']}.png"
        print(f"{job['name']}: done", flush=True)
    index_path = args.out / "showcase.json"
    previous = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    merged = {j["name"]: j for j in previous}
    merged.update({j["name"]: j for j in jobs if "file" in j})
    index_path.write_text(json.dumps(list(merged.values()), ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
