"""Comparison samples for the reference adapter.

For each eval case (a held-out or seen character): the reference image, the adapter's output for
another scene of the group (prompt without character / artist / appearance tags), the base model
on the same stripped prompt without the adapter, and the image the generator made with full tags.

  python train/sample.py --prep PREP --lora runs/train/x/lora_step02000.safetensors --out runs/train/x/samples
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402

PANEL_HEIGHT = 512


class Prep:
    """index.json plus contexts and latents of a prepare.py output, held in RAM."""

    def __init__(self, prep_dir: Path):
        self.dir = Path(prep_dir)
        self.index = json.loads((self.dir / "index.json").read_text(encoding="utf-8"))
        ctx = load_file(str(self.dir / "contexts.safetensors"))
        self._ctx_data, self._ctx_offsets = ctx["data"], ctx["offsets"].tolist()
        self.targets = self._load(self.index["files"]["targets"])
        self.refs = self._load(self.index["files"]["refs"])

    def _load(self, names: list[str]) -> dict[str, torch.Tensor]:
        tensors = {}
        for name in names:
            tensors.update(load_file(str(self.dir / name)))
        return tensors

    def context(self, prompt_index: int) -> torch.Tensor:
        return self._ctx_data[self._ctx_offsets[prompt_index]:self._ctx_offsets[prompt_index + 1]]

    def record(self, record_id: str) -> dict:
        return self.index["records"][record_id]

    def image(self, record_id: str) -> Image.Image | None:
        record = self.record(record_id)
        data_dir = self.index.get("data_dirs", {}).get(record.get("source"), self.index["data_dir"])
        path = Path(data_dir) / record["image"]
        return Image.open(path).convert("RGB") if path.exists() else None


def to_pil(pixels: torch.Tensor) -> Image.Image:
    return Image.fromarray((pixels.numpy() * 255).round().astype("uint8"))


def labeled_row(panels: list[tuple[str, Image.Image]]) -> Image.Image:
    resized = [(label, im.resize((round(im.width * PANEL_HEIGHT / im.height), PANEL_HEIGHT), Image.LANCZOS))
               for label, im in panels]
    row = Image.new("RGB", (sum(im.width for _, im in resized) + 8 * (len(resized) - 1), PANEL_HEIGHT + 28), "white")
    draw = ImageDraw.Draw(row)
    x = 0
    for label, im in resized:
        row.paste(im, (x, 28))
        draw.text((x + 4, 6), label, fill="black")
        x += im.width + 8
    return row


def run_eval(dm, lora_modules, prep: Prep, vae, out_dir: Path, config: dict, device, baseline_dir: Path | None = None) -> list[dict]:
    lf = anima_env.latent_format()
    sample_cfg = config["sample"]
    adapter_cfg = config["adapter"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []

    def generate(ctx_pos, ctx_neg, shape, seed, ref=None, offset=(0, 0)):
        x = anima_env.sample_euler(dm, ctx_pos, ctx_neg, shape, seed, sample_cfg["steps"], sample_cfg["cfg"],
                                   config["train"]["shift"], device, ref=ref, ref_offset=offset)
        return to_pil(anima_env.vae_decode(vae, lf.process_out(x.to(device)).float())[0])

    for case in prep.index["eval_cases"]:
        target = prep.record(case["target"])
        width, height = case["width"], case["height"]
        raw_ref = prep.refs[f"{case['ref']}@{width}x{height}"][None].float()
        ref = lf.process_in(raw_ref.to(device)).to(torch.bfloat16)
        shape = (1, 16, 1, height // 8, width // 8)
        ctx_pos = prep.context(target["prompts"]["stripped"])
        ctx_neg = prep.context(target["prompts"]["negative"])
        offset = anima_env.default_ref_offset(shape[-2], adapter_cfg)

        anima_env.set_lora_enabled(lora_modules, True)
        adapted = generate(ctx_pos, ctx_neg, shape, case["seed"], ref=ref, offset=offset)

        name = f"{case['split']}_{case['group_id']}"
        baseline_path = (baseline_dir or out_dir) / f"{name}_baseline.png"
        if baseline_path.exists():
            baseline = Image.open(baseline_path).convert("RGB")
        else:
            anima_env.set_lora_enabled(lora_modules, False)
            baseline = generate(ctx_pos, ctx_neg, shape, case["seed"])
            anima_env.set_lora_enabled(lora_modules, True)
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline.save(baseline_path)

        ref_image = prep.image(case["ref"]) or to_pil(anima_env.vae_decode(vae, raw_ref)[0])
        truth = prep.image(case["target"]) or to_pil(
            anima_env.vae_decode(vae, prep.targets[case["target"]][None].float())[0])
        row = labeled_row([(f"reference ({case['split']}: {case['character']})", ref_image),
                           ("adapter, stripped prompt", adapted),
                           ("no adapter, stripped prompt", baseline),
                           ("generator target (full tags)", truth)])
        row.save(out_dir / f"{name}.jpg", quality=90)
        rows.append(row)
        summary.append({**case, "prompt": prep.index["prompts"][target["prompts"]["stripped"]],
                        "original_prompt": target["prompt_original"], "image": f"{name}.jpg"})
        print(f"sample {name} done", flush=True)

    if rows:
        width = max(r.width for r in rows)
        sheet = Image.new("RGB", (width, sum(r.height for r in rows)), "white")
        y = 0
        for row in rows:
            sheet.paste(row, (0, y))
            y += row.height
        sheet.save(out_dir / "contact_sheet.jpg", quality=85)
    (out_dir / "samples.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def run_eval_dual(dm, lora_modules, prep: Prep, vae, out_dir: Path, config: dict, device,
                  refs: dict | None = None, targets: dict | None = None) -> list[dict]:
    """Two-reference eval (prepare_dual.py cases): for a target of group (character C, style S), the
    character slot holds C in another style and the style slot S on another character.

    Row: character ref | style ref | both slots | character slot only | style slot only (+ character
    tags in the prompt) | generator target. refs / targets are model-space latents if already loaded.
    """
    lf = anima_env.latent_format()
    sample_cfg, adapter_cfg = config["sample"], config["adapter"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []

    def latent(key: str) -> torch.Tensor:
        if refs is not None:
            return refs[key][None].to(device)
        return lf.process_in(prep.refs[key][None].float().to(device)).to(torch.bfloat16)

    def generate(ctx_pos, ctx_neg, shape, seed, slot_refs):
        frames = anima_env.slot_frames(slot_refs, shape[-2], adapter_cfg)
        x = anima_env.sample_euler(dm, ctx_pos, ctx_neg, shape, seed, sample_cfg["steps"], sample_cfg["cfg"],
                                   config["train"]["shift"], device, frames=frames)
        return to_pil(anima_env.vae_decode(vae, lf.process_out(x.to(device)).float())[0])

    def image_or_decode(record_id: str, width: int, height: int) -> Image.Image:
        image = prep.image(record_id)
        if image is not None:
            return image
        key = f"{record_id}@{width}x{height}"
        raw = lf.process_out(latent(key).float()) if refs is not None else prep.refs[key][None].float()
        return to_pil(anima_env.vae_decode(vae, raw.to(device))[0])

    anima_env.set_lora_enabled(lora_modules, True)
    for case in prep.index["eval_cases"]:
        target = prep.record(case["target"])
        width, height = case["width"], case["height"]
        shape = (1, 16, 1, height // 8, width // 8)
        char_ref = latent(f"{case['character_ref']}@{width}x{height}")
        style_ref = latent(f"{case['style_ref']}@{width}x{height}")
        stripped = prep.context(target["prompts"]["stripped"])
        negative = prep.context(target["prompts"]["negative"])
        both = generate(stripped, negative, shape, case["seed"], {"character": char_ref, "style": style_ref})
        char_only = generate(stripped, negative, shape, case["seed"], {"character": char_ref})
        style_only = generate(prep.context(target["prompts"]["with_character"]), negative, shape, case["seed"],
                              {"style": style_ref})
        truth = image_or_decode(case["target"], width, height)
        name = f"{case['split']}_{case['group_id']}"
        row = labeled_row([(f"character ref ({case['split']}: {case['character']})",
                            image_or_decode(case["character_ref"], width, height)),
                           (f"style ref ({case['artists']})", image_or_decode(case["style_ref"], width, height)),
                           ("both slots, scene prompt", both),
                           ("character slot only, scene prompt", char_only),
                           ("style slot only, + character tags", style_only),
                           ("generator target (full tags)", truth)])
        row.save(out_dir / f"{name}.jpg", quality=90)
        rows.append(row)
        summary.append({**case, "prompt": prep.index["prompts"][target["prompts"]["stripped"]],
                        "style_only_prompt": prep.index["prompts"][target["prompts"]["with_character"]],
                        "original_prompt": target["prompt_original"], "image": f"{name}.jpg"})
        print(f"sample {name} done", flush=True)

    if rows:
        sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
        y = 0
        for row in rows:
            sheet.paste(row, (0, y))
            y += row.height
        sheet.save(out_dir / "contact_sheet.jpg", quality=85)
    (out_dir / "samples.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="defaults to the config stored in the prep index")
    parser.add_argument("--comfy-root", default="/workspace/ComfyUI")
    args = parser.parse_args()
    prep = Prep(args.prep)
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else prep.index["config"]
    anima_env.setup_comfy(args.comfy_root)
    device = torch.device("cuda")
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)
    modules = anima_env.inject_lora(dm, config["adapter"]["lora_rank"], config["adapter"]["lora_alpha"])
    anima_env.load_lora(modules, args.lora)
    vae = anima_env.load_vae(config["comfy"]["vae_name"])
    if "slots" in config["adapter"]:
        run_eval_dual(dm, modules, prep, vae, args.out, config, device)
    else:
        run_eval(dm, modules, prep, vae, args.out, config, device, baseline_dir=args.out.parent / "baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
