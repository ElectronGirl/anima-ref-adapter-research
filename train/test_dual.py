"""Checks for the two-reference path: the training forward with frames given as refs= matches the
single-reference forward, and the ComfyUI node's forward matches the training forward for any
combination of slots.

  python train/test_dual.py --comfy-root /workspace/ComfyUI
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402

SLOTS = {"character": {"t": 1, "offset_h": 1.0, "offset_w": 0}, "style": {"t": 2, "offset_h": 1.0, "offset_w": 0}}


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", default="/workspace/ComfyUI")
    parser.add_argument("--unet", default="anima_baseV10.safetensors")
    args = parser.parse_args()
    anima_env.setup_comfy(args.comfy_root)
    device = torch.device("cuda")
    _, dm = anima_env.load_diffusion_model(args.unet, device)
    spec = importlib.util.spec_from_file_location("anima_ref_nodes", ROOT / "comfyui_nodes" / "anima_ref_adapter_nodes" / "__init__.py")
    nodes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nodes)

    gen = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(2, 16, 1, 16, 24, generator=gen).to(device, torch.bfloat16)
    char = torch.randn(2, 16, 1, 16, 24, generator=gen).to(device, torch.bfloat16)
    style = torch.randn(2, 16, 1, 16, 24, generator=gen).to(device, torch.bfloat16)
    ctx = torch.randn(2, 512, 1024, generator=gen).to(device, torch.bfloat16) * 0.1
    sigma = torch.tensor([0.9, 0.3], device=device)
    cfg = {"slots": SLOTS}
    tol = lambda ref: 1e-3 * max(1.0, ref.abs().max().item())  # noqa: E731

    single = anima_env.ref_forward(dm, x, sigma, ctx, ref=char, ref_offset=(8, 0)).float()
    frames = anima_env.slot_frames({"character": char}, x.shape[-2], cfg)
    diff = (anima_env.ref_forward(dm, x, sigma, ctx, refs=frames).float() - single).abs().max().item()
    assert diff == 0.0, f"refs= differs from ref=: {diff}"
    print(f"single-slot refs= matches ref=: max diff {diff}")

    outputs = {}
    for name, slot_refs in (("character", {"character": char}), ("style", {"style": style}),
                            ("both", {"character": char, "style": style})):
        ours = anima_env.ref_forward(dm, x, sigma, ctx, refs=anima_env.slot_frames(slot_refs, x.shape[-2], cfg)).float()
        node_refs = {slot: [slot_refs[slot]] if slot in slot_refs else [] for slot in SLOTS}
        theirs = nodes._ref_forward(dm, x, sigma, ctx, node_refs, SLOTS, 1.0, {}).float()
        diff = (ours - theirs).abs().max().item()
        assert diff <= tol(ours), f"node forward differs for {name}: {diff}"
        outputs[name] = ours
        print(f"node forward matches for {name}: max diff {diff}")
    # The style slot must not be read as the character slot (different t index -> different output).
    diff = (outputs["style"] - anima_env.ref_forward(dm, x, sigma, ctx, ref=style, ref_offset=(8, 0)).float()).abs().max().item()
    assert diff > 0, "style slot frame is indistinguishable from a character slot frame"
    print(f"style slot differs from character slot with the same image: max diff {diff:.4f}")
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
