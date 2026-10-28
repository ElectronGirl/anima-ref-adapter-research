"""ComfyUI API workflow for one heterogeneous batch (same resolution, per-item prompt and seed).

Acceleration ("block_compile_sage2_speed" profile):
SageAttention (KJNodes) -> AnimaBlockCompile -> SamplerSPEED. Each part can be switched off
for machines that lack the node (local test).
"""
from __future__ import annotations

import json

PROFILES = {
    "baseline": dict(sage=False, block_compile=False, speed=False),
    "sage2": dict(sage=True, block_compile=False, speed=False),
    "sage2_speed": dict(sage=True, block_compile=False, speed=True),
    "block_compile_sage2": dict(sage=True, block_compile=True, speed=False),
    "block_compile_sage2_speed": dict(sage=True, block_compile=True, speed=True),
}


def build_batch_workflow(records: list[dict], config: dict, out_dir: str, results_path: str) -> dict:
    width, height = records[0]["width"], records[0]["height"]
    if any((r["width"], r["height"]) != (width, height) for r in records):
        raise ValueError("a batch must share one resolution")
    if len({r["negative"] for r in records}) != 1:
        raise ValueError("a batch must share one negative prompt")
    comfy, sampling, accel = config["comfy"], config["sampling"], config["acceleration"]
    flags = PROFILES[accel["profile"]]
    out = config["output"]

    wf: dict = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": comfy["unet_name"], "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": comfy["clip_name"],
                                                        "type": comfy.get("clip_type", "stable_diffusion"), "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": comfy["vae_name"]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": records[0]["negative"], "clip": ["clip", 0]}},
        "latent": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": len(records)}},
    }
    model = ["unet", 0]
    if flags["sage"]:
        sage = accel.get("sage_attention", {})
        wf["sage"] = {"class_type": "PathchSageAttentionKJ", "inputs": {
            "model": model, "sage_attention": sage.get("mode", "auto"), "allow_compile": bool(sage.get("allow_compile", True))}}
        model = ["sage", 0]
    if flags["block_compile"]:
        wf["compile"] = {"class_type": "AnimaBlockCompile", "inputs": {"model": model}}
        model = ["compile", 0]

    cond_inputs = {}
    for i, record in enumerate(records, start=1):
        wf[f"pos{i}"] = {"class_type": "CLIPTextEncode", "inputs": {"text": record["prompt"], "clip": ["clip", 0]}}
        cond_inputs[f"conditioning_{i}"] = [f"pos{i}", 0]
    wf["posbatch"] = {"class_type": "AnimaRefHeteroConditioningBatch", "inputs": {"model": model, **cond_inputs}}
    wf["noise"] = {"class_type": "AnimaRefHeteroSeedNoise",
                   "inputs": {"seeds": ",".join(str(r["seed"] % 2**63) for r in records)}}

    if flags["speed"]:
        speed = accel["speed"]
        wf["sampler"] = {"class_type": "SamplerSPEED", "inputs": {
            "base_sampler": speed["base_sampler"], "transform": speed["transform"], "mode": speed["mode"],
            "model_preset": speed["model_preset"], "scales": speed["scales"], "delta": float(speed["delta"]),
            "manual_sigmas": speed["manual_sigmas"], "spectrum_A": float(speed["spectrum_A"]),
            "spectrum_beta": float(speed["spectrum_beta"]), "seed": int(records[0]["seed"] % 2**31)}}
    else:
        wf["sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampling["sampler_name"]}}
    wf["sigmas"] = {"class_type": "BasicScheduler", "inputs": {
        "model": model, "scheduler": sampling["scheduler"], "steps": int(sampling["steps"]), "denoise": 1.0}}
    wf["guider"] = {"class_type": "CFGGuider", "inputs": {
        "model": model, "positive": ["posbatch", 0], "negative": ["neg", 0], "cfg": float(sampling["cfg"])}}
    wf["sample"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
        "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}}
    wf["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}

    keep = ("record_id", "group_id", "character", "character_danbooru", "artists", "identity", "rating",
            "quality", "scene", "prompt", "negative", "width", "height", "seed")
    save_inputs = {
        "images": ["decode", 0],
        "records_json": json.dumps([{k: r[k] for k in keep} for r in records], ensure_ascii=False),
        "out_dir": out_dir, "results_path": results_path,
        "image_format": out["image_format"], "webp_quality": int(out.get("webp_quality", 95)),
    }
    if out.get("save_latent"):
        save_inputs["samples"] = ["sample", 0]
    wf["save"] = {"class_type": "AnimaRefDatagenSave", "inputs": save_inputs}
    return wf


def build_flush_workflow(token: str) -> dict:
    return {"flush": {"class_type": "AnimaRefDatagenFlush", "inputs": {"token": token}}}
