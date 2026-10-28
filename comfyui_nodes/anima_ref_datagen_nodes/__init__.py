"""ComfyUI nodes for reference-adapter data generation.

AnimaRefHeteroConditioningBatch / AnimaRefHeteroSeedNoise: one sampler call renders a batch whose
items each have their own prompt and diffusion seed.

AnimaRefDatagenSave writes images (and optionally latents) under caller-chosen record ids on a
background thread, so encoding and disk writes overlap the next batch on the GPU. A record is
appended to the results file only after its files are fully written.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

MAX_BATCH = 64
_SAVE_WORKERS = int(os.environ.get("DATAGEN_SAVE_WORKERS", "8"))
_executor = ThreadPoolExecutor(max_workers=_SAVE_WORKERS, thread_name_prefix="datagen-save")
_pending_lock = threading.Lock()
_pending: list = []
_results_lock = threading.Lock()
_MAX_PENDING_BATCHES = int(os.environ.get("DATAGEN_MAX_PENDING_BATCHES", "3"))


class AnimaHeterogeneousConditioningBatch:
    """Batch independently encoded Anima prompts into one post-adapter context batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",), "conditioning_1": ("CONDITIONING",)},
            "optional": {f"conditioning_{i}": ("CONDITIONING",) for i in range(2, MAX_BATCH + 1)},
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "batch"
    CATEGORY = "Anima Ref Datagen"

    def batch(self, model, conditioning_1, **optional):
        import comfy.model_management as model_management

        conditionings = [conditioning_1] + [
            optional[key] for key in sorted(optional, key=lambda k: int(k.rsplit("_", 1)[1]))
            if optional[key] is not None
        ]
        model_management.load_models_gpu([model])
        device = model.load_device
        dtype = model.model.get_dtype_inference()
        diffusion_model = model.model.diffusion_model
        contexts, first_metadata = [], None
        # Outside sampling ComfyUI does not cast stored (bf16) weights per op, so run the
        # adapter under autocast at the model's inference dtype, as extra_conds would.
        with torch.inference_mode(), torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            for conditioning in conditionings:
                if len(conditioning) != 1:
                    raise ValueError("each prompt must be a single conditioning item")
                cross_attn, metadata = conditioning[0]
                ids, weights = metadata.get("t5xxl_ids"), metadata.get("t5xxl_weights")
                if ids is None or weights is None:
                    raise ValueError("expected raw Anima conditioning with t5xxl ids and weights")
                contexts.append(diffusion_model.preprocess_text_embeds(
                    cross_attn.to(device=device, dtype=dtype),
                    ids.flatten().unsqueeze(0).to(device=device),
                    t5xxl_weights=weights.flatten().unsqueeze(0).unsqueeze(-1).to(device=device, dtype=dtype),
                ))
                if first_metadata is None:
                    first_metadata = metadata.copy()
        first_metadata.pop("t5xxl_ids", None)
        first_metadata.pop("t5xxl_weights", None)
        return ([[torch.cat(contexts, dim=0).detach(), first_metadata]],)


class _SeedBatchNoise:
    def __init__(self, seeds):
        self.seeds = seeds
        self.seed = seeds[0]

    def generate_noise(self, input_latent):
        latent = input_latent["samples"]
        if latent.shape[0] != len(self.seeds):
            raise ValueError(f"latent batch {latent.shape[0]} != {len(self.seeds)} seeds")
        shape = (1, *latent.shape[1:])
        return torch.cat([
            torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(seed),
                        dtype=torch.float32, device="cpu").to(dtype=latent.dtype)
            for seed in self.seeds
        ], dim=0)


class AnimaHeterogeneousSeedNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seeds": ("STRING", {"default": "41,31415"})}}

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "build"
    CATEGORY = "Anima Ref Datagen"

    def build(self, seeds):
        parsed = [int(v) for v in seeds.split(",") if v.strip()]
        if not 1 <= len(parsed) <= MAX_BATCH:
            raise ValueError(f"need 1..{MAX_BATCH} seeds")
        return (_SeedBatchNoise(parsed),)


def _write_item(image: np.ndarray, latent, rec: dict, out_dir: Path, fmt: str, quality: int, results: Path):
    group_dir = out_dir / rec["group_id"]
    group_dir.mkdir(parents=True, exist_ok=True)
    image_path = group_dir / f"{rec['record_id']}.{fmt}"
    tmp = image_path.with_name(image_path.name + ".tmp")
    pil = Image.fromarray(image)
    if fmt == "webp":
        pil.save(tmp, format="WEBP", quality=quality, method=4)
    else:
        pil.save(tmp, format="PNG", compress_level=1)
    tmp.replace(image_path)
    entry = {**rec, "status": "complete", "image": str(image_path.relative_to(out_dir)),
             "finished_at": time.time()}
    if latent is not None:
        latent_path = group_dir / f"{rec['record_id']}.latent.safetensors"
        tmp = latent_path.with_name(latent_path.name + ".tmp")
        save_file({"latent": latent}, str(tmp), metadata={"space": "vae_raw", "record_id": rec["record_id"]})
        tmp.replace(latent_path)
        entry["latent"] = str(latent_path.relative_to(out_dir))
    with _results_lock:
        with results.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _drain(max_pending: int):
    with _pending_lock:
        while len(_pending) > max_pending:
            futures = _pending.pop(0)
            for future in futures:
                future.result()


class AnimaRefDatagenSave:
    """Save a generated batch under record ids; files are written asynchronously."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "records_json": ("STRING", {"multiline": True}),
                "out_dir": ("STRING",),
                "results_path": ("STRING",),
                "image_format": (["webp", "png"],),
                "webp_quality": ("INT", {"default": 95, "min": 50, "max": 100}),
            },
            "optional": {"samples": ("LATENT",)},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "Anima Ref Datagen"

    def save(self, images, records_json, out_dir, results_path, image_format, webp_quality, samples=None):
        records = json.loads(records_json)
        if len(records) != images.shape[0]:
            raise ValueError(f"{images.shape[0]} images for {len(records)} records")
        if not torch.isfinite(images).all():
            raise ValueError("non-finite pixels in batch (NaN); refusing to save")
        pixels = (images.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        latents = None
        if samples is not None:
            latents = samples["samples"].to(torch.float16).cpu()
            if latents.shape[0] != len(records):
                raise ValueError("latent batch does not match records")
        out = Path(out_dir)
        results = Path(results_path)
        results.parent.mkdir(parents=True, exist_ok=True)
        futures = [
            _executor.submit(_write_item, pixels[i], None if latents is None else latents[i:i + 1].clone(),
                             rec, out, image_format, int(webp_quality), results)
            for i, rec in enumerate(records)
        ]
        with _pending_lock:
            _pending.append(futures)
        _drain(_MAX_PENDING_BATCHES)  # bound memory; surfaces write errors of older batches
        return {"ui": {"datagen_saved": [len(records)]}}


class AnimaRefDatagenFlush:
    """Wait for all queued writes (run once at the end of a worker)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"token": ("STRING", {"default": ""})}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "flush"
    CATEGORY = "Anima Ref Datagen"

    def flush(self, token):
        _drain(0)
        return {"ui": {"datagen_flushed": [token]}}


NODE_CLASS_MAPPINGS = {
    "AnimaRefHeteroConditioningBatch": AnimaHeterogeneousConditioningBatch,
    "AnimaRefHeteroSeedNoise": AnimaHeterogeneousSeedNoise,
    "AnimaRefDatagenSave": AnimaRefDatagenSave,
    "AnimaRefDatagenFlush": AnimaRefDatagenFlush,
}
