"""The ComfyUI node's asymmetric-reference forward (LoRA by forward hooks, cached reference) against
the training forward (anima_env.asym_forward with LoRALinear), with a trained asym adapter.

  python train/test_node_asym.py --lora runs/train/copy_asym/lora_step01000.safetensors
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--comfy-root", default="/workspace/ComfyUI")
    args = parser.parse_args()
    anima_env.setup_comfy(args.comfy_root)
    device = torch.device("cuda")
    _, dm = anima_env.load_diffusion_model("anima_baseV10.safetensors", device)
    spec = importlib.util.spec_from_file_location("anima_ref_nodes", ROOT / "comfyui_nodes" / "anima_ref_adapter_nodes" / "__init__.py")
    nodes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nodes)

    gen = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(2, 16, 1, 32, 24, generator=gen).to(device, torch.bfloat16)
    ref = torch.randn(1, 16, 1, 40, 28, generator=gen).to(device, torch.bfloat16)  # another size than the target
    ctx = (torch.randn(2, 512, 1024, generator=gen) * 0.1).to(device, torch.bfloat16)
    sigma = torch.tensor([0.9, 0.3], device=device)

    adapter = nodes._Adapter(str(args.lora), 1.0)
    assert adapter.asym, "not an asym adapter"
    adapter.install(dm)
    with torch.inference_mode():
        kv = nodes._asym_ref_kv(adapter, dm, [ref[:, :, 0]], x, 1.0)  # 4D, as the conditioning passes it
        kv_again = nodes._asym_ref_kv(adapter, dm, [ref[:, :, 0]], x, 1.0)
        assert kv_again is kv and len(adapter.ref_cache) == 1, "reference cache missed"
        nodes._active["key"] = adapter.key
        node_out = nodes._asym_forward(dm, x, sigma, ctx, kv, {}).float()
        # Another node replacing a block's attention (as Anima Safe PAG does): a pass-through replacement
        # must give the same output, and a PAG-like one (blend with each query's own value) must run.
        attn = dm.blocks[18].self_attn
        original = attn.compute_attention
        attn.compute_attention = lambda q, k, v, transformer_options=None: original(q, k, v, transformer_options=transformer_options or {})
        passthrough = nodes._asym_forward(dm, x, sigma, ctx, kv, {}).float()
        print(f"patched attention (pass-through) vs unpatched: max diff {(passthrough - node_out).abs().max().item():.3g}")

        def pag_like(q, k, v, transformer_options=None):
            q, k, v = (t.take() if not torch.is_tensor(t) else t for t in (q, k, v))
            normal = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
            weak = normal.lerp(v, 0.75)
            return attn.output_proj(weak.reshape(*weak.shape[:2], -1))
        attn.compute_attention = pag_like
        pag_out = nodes._asym_forward(dm, x, sigma, ctx, kv, {}).float()
        print(f"PAG-like attention runs: mean change {(pag_out - node_out).abs().mean().item():.3g}")
        del attn.compute_attention
        nodes._active["key"] = None
        node_plain = dm(x, sigma, ctx).float()  # hooks inactive: the base model

    meta = nodes._read_metadata(str(args.lora))
    modules = anima_env.inject_lora(dm, meta.get("rank", 32), meta.get("alpha", 32))
    anima_env.load_lora(modules, args.lora)
    offset = (x.shape[-2] // 2, 0)
    with torch.inference_mode():
        train_out = anima_env.asym_forward(dm, x, sigma, ctx, ref=ref.expand(2, -1, -1, -1, -1), ref_offset=offset).float()
        anima_env.set_lora_enabled(modules, False)
        base_out = anima_env.asym_forward(dm, x, sigma, ctx).float()
    diff = (node_out - train_out).abs().max().item()
    scale = train_out.abs().max().item()
    print(f"node (hooks, cached reference) vs training forward (LoRALinear): max diff {diff:.4g} (scale {scale:.3g}), "
          f"mean diff {(node_out - train_out).abs().mean().item():.3g}")
    print(f"adapter effect: mean |with - base| {(train_out - base_out).abs().mean().item():.3g}; "
          f"node base vs training base {(node_plain - base_out).abs().max().item():.3g}")
    assert diff <= 2e-2 * max(1.0, scale), "node forward differs from the training forward"
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
