"""Checks and timings for the asymmetric-attention reference forward (anima_env.asym_forward).

1. no reference: asym_forward == ComfyUI's model forward
2. reference computed alongside == cached reference keys/values (encode_ref_asym)
3. gradients reach the LoRA through the target and the reference stream
4. timings at 832x1216: training step (batch 4) and sampling step (batch 2, CFG) for the two-frame
   ref_forward, asym with the reference trained, asym with a frozen reference (no LoRA, no grad);
   sampling with a cached reference

  python train/test_asym.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402


def timed(fn, repeat: int = 5) -> float:
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat


def main() -> int:
    anima_env.setup_comfy("/workspace/ComfyUI")
    device = torch.device("cuda")
    _, dm = anima_env.load_diffusion_model("anima_baseV10.safetensors", device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(2, 16, 1, 32, 24, generator=gen).to(device, torch.bfloat16)
    ref = torch.randn(2, 16, 1, 32, 24, generator=gen).to(device, torch.bfloat16)
    ctx = (torch.randn(2, 512, 1024, generator=gen) * 0.1).to(device, torch.bfloat16)
    sigma = torch.tensor([0.9, 0.3], device=device)
    offset = (16, 0)

    with torch.inference_mode():
        ours, theirs = anima_env.asym_forward(dm, x, sigma, ctx).float(), dm(x, sigma, ctx).float()
        print(f"1. no reference vs ComfyUI forward: max diff {(ours - theirs).abs().max().item():.3g} "
              f"(scale {theirs.abs().max().item():.3g})", flush=True)
        joint = anima_env.asym_forward(dm, x, sigma, ctx, ref=ref, ref_offset=offset).float()
        cache = anima_env.encode_ref_asym(dm, ref, offset)
        cached = anima_env.asym_forward(dm, x, sigma, ctx, ref_kv=cache).float()
        print(f"2. joint vs cached reference: max diff {(joint - cached).abs().max().item():.3g}; "
              f"reference changes the output by {(joint - ours).abs().mean().item():.3g} (mean abs)", flush=True)

    modules = anima_env.inject_lora(dm, 32, 32)
    for m in modules.values():
        torch.nn.init.normal_(m.lora_up, std=1e-3)
    out = anima_env.asym_forward(dm, x, sigma, ctx, ref=ref, ref_offset=offset)
    out.float().pow(2).mean().backward()
    probe = modules["blocks.27.self_attn.q_proj"]
    ref_only = modules["blocks.27.cross_attn.q_proj"]  # target stream only (the reference skips cross-attention)
    print(f"3. grad norms: self_attn.q_proj {probe.lora_up.grad.norm().item():.3g}, "
          f"cross_attn.q_proj {ref_only.lora_up.grad.norm().item():.3g}", flush=True)
    for m in modules.values():
        m.lora_down.grad = m.lora_up.grad = None

    def set_lora(enabled):
        anima_env.set_lora_enabled(modules, enabled)

    B = 4
    xt = torch.randn(B, 16, 1, 152, 104, device=device, dtype=torch.bfloat16)
    rt = torch.randn(B, 16, 1, 152, 104, device=device, dtype=torch.bfloat16)
    ct = (torch.randn(B, 512, 1024, device=device) * 0.1).to(torch.bfloat16)
    st = torch.full((B,), 0.7, device=device)
    off = (76, 0)

    def step_old():
        anima_env.ref_forward(dm, xt, st, ct, ref=rt, ref_offset=off, checkpoint_blocks=16).float().pow(2).mean().backward()

    def step_asym():
        anima_env.asym_forward(dm, xt, st, ct, ref=rt, ref_offset=off, checkpoint_blocks=16).float().pow(2).mean().backward()

    def step_frozen():
        with torch.no_grad():
            cache = anima_env.encode_ref_asym(dm, rt, off, ref_hook=lambda enter: set_lora(not enter))
        anima_env.asym_forward(dm, xt, st, ct, ref_kv=cache, checkpoint_blocks=16).float().pow(2).mean().backward()

    def step_plain():
        anima_env.asym_forward(dm, xt, st, ct, checkpoint_blocks=16).float().pow(2).mean().backward()

    results = {}
    for name, fn in (("two-frame ref_forward", step_old), ("asym, reference trained", step_asym),
                     ("asym, reference frozen", step_frozen), ("no reference (floor)", step_plain)):
        torch.cuda.reset_peak_memory_stats()
        results[name] = timed(fn, 3)
        print(f"4a. train step B={B}: {name}: {results[name]:.2f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
              flush=True)

    xs, rs, cs, ss = xt[:2], rt[:2], ct[:2], st[:2]
    with torch.inference_mode():
        cache = anima_env.encode_ref_asym(dm, rs[:1].expand(2, -1, -1, -1, -1), off)
        t_old = timed(lambda: anima_env.ref_forward(dm, xs, ss, cs, ref=rs, ref_offset=off))
        t_asym = timed(lambda: anima_env.asym_forward(dm, xs, ss, cs, ref=rs, ref_offset=off))
        t_cached = timed(lambda: anima_env.asym_forward(dm, xs, ss, cs, ref_kv=cache))
        t_plain = timed(lambda: anima_env.asym_forward(dm, xs, ss, cs))
        t_encode = timed(lambda: anima_env.encode_ref_asym(dm, rs[:1], off))
    print(f"4b. sampling step (CFG batch 2): two-frame {t_old:.3f} s, asym uncached {t_asym:.3f} s, "
          f"asym cached {t_cached:.3f} s, no reference {t_plain:.3f} s; reference encode once {t_encode:.3f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
