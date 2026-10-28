"""Anima reference adapter: model plumbing shared by prepare.py, train.py and sample.py.

The reference image is appended to the target latent as a second frame on the T axis, as
ComfyUI-Cosmos-Reference does, with two changes:
  * the reference frame gets timestep 0 (it is a clean image) while the target frame gets sigma;
  * the reference frame's RoPE (h, w) coordinates are shifted past the target grid. With aligned
    coordinates the video prior reads "same (h, w) in the next frame = same content" and copies
    the pose; shifted, the frame can only be used as a source of identity and style.
LoRA on every block's attention and MLP learns to read the reference frame.

Everything runs on ComfyUI's own Anima modules (loaded from its checkout) so the trained weights
plug straight back into ComfyUI. self_test() checks that the forward used here matches ComfyUI's.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

LORA_TARGETS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj",
    "cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.output_proj",
    "mlp.layer1", "mlp.layer2",
)


def setup_comfy(comfy_root: str | Path) -> None:
    """Import ComfyUI as a library with settings that allow backprop through its Anima modules."""
    root = str(Path(comfy_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import comfy.cli_args as cli_args
    # Dynamic VRAM (comfy-aimdo) virtualises parameters; training needs plain resident weights.
    cli_args.args.disable_dynamic_vram = True
    import comfy.model_management  # noqa: F401  (reads the args above at import)
    import comfy.quant_ops
    from comfy_kitchen.backends.eager import rope as eager_rope
    # The dispatching comfy_kitchen op has no autograd formula; its eager version is plain torch.
    comfy.quant_ops.ck.rms_rope_split_half = eager_rope.rms_rope_split_half


def model_path(folder: str, name: str) -> str:
    import folder_paths
    return folder_paths.get_full_path_or_raise(folder, name)


def load_diffusion_model(unet_name: str, device: torch.device):
    """Return (base_model, diffusion_model): comfy.model_base.Anima and its MiniTrainDIT, in bf16."""
    import comfy.sd
    patcher = comfy.sd.load_diffusion_model(model_path("diffusion_models", unet_name),
                                            model_options={"dtype": torch.bfloat16})
    base = patcher.model
    dm = base.diffusion_model.to(device=device, dtype=torch.bfloat16)
    dm.requires_grad_(False)
    dm.eval()
    if getattr(dm, "extra_per_block_abs_pos_emb", False):
        raise NotImplementedError("absolute per-block position embeddings would leak the target grid into the reference")
    return base, dm


def load_text_encoder(clip_name: str):
    import comfy.sd
    return comfy.sd.load_clip(ckpt_paths=[model_path("text_encoders", clip_name)], embedding_directory=None,
                              clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)


def load_vae(vae_name: str):
    import comfy.sd
    import comfy.utils
    return comfy.sd.VAE(sd=comfy.utils.load_torch_file(model_path("vae", vae_name)))


def latent_format():
    import comfy.latent_formats
    return comfy.latent_formats.Wan21()


@torch.inference_mode()
def encode_prompt(clip, dm, text: str, device: torch.device) -> torch.Tensor:
    """Post-llm_adapter context [tokens, 1024] (bf16, unpadded), as Anima.extra_conds builds it."""
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))
    cross_attn, meta = cond[0]
    ids, weights = meta["t5xxl_ids"], meta["t5xxl_weights"]
    ctx = dm.preprocess_text_embeds(
        cross_attn.to(device=device, dtype=torch.bfloat16),
        ids.flatten().unsqueeze(0).to(device=device),
        t5xxl_weights=weights.flatten().unsqueeze(0).unsqueeze(-1).to(device=device, dtype=torch.bfloat16),
    )
    return ctx[0, : ids.numel()].float().cpu().to(torch.bfloat16)


def pad_contexts(contexts: list[torch.Tensor], min_len: int = 512) -> torch.Tensor:
    """Stack contexts, zero-padded like preprocess_text_embeds (to 512, or the longest)."""
    length = max(min_len, max(c.shape[0] for c in contexts))
    return torch.stack([F.pad(c, (0, 0, 0, length - c.shape[0])) for c in contexts])


def letterbox(image, width: int, height: int, fill: int = 255):
    """Fit a PIL image inside width x height, keeping aspect, and pad with a flat colour."""
    from PIL import Image
    scale = min(width / image.width, height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    canvas = Image.new("RGB", (width, height), (fill, fill, fill))
    canvas.paste(image.convert("RGB").resize(size, Image.LANCZOS), ((width - size[0]) // 2, (height - size[1]) // 2))
    return canvas


@torch.inference_mode()
def vae_encode_image(vae, image) -> torch.Tensor:
    """Raw VAE latent [16, 1, h, w] of one PIL image (one image per call: Wan VAE reads a batch as video)."""
    import numpy as np
    pixels = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).unsqueeze(0)
    latent = vae.encode(pixels)
    if latent.ndim == 4:
        latent = latent.unsqueeze(2)
    return latent[0].float().cpu()


@torch.inference_mode()
def vae_decode(vae, raw_latent: torch.Tensor) -> torch.Tensor:
    """[B, 16, 1, h, w] raw latent -> [B, H, W, 3] float pixels in 0..1."""
    images = vae.decode(raw_latent)
    return images.reshape(-1, *images.shape[-3:]).clamp(0, 1).float().cpu()


def frame_positions(t_index: int, height: int, width: int, h0: int, w0: int, device) -> tuple[torch.Tensor, ...]:
    hh, ww = torch.meshgrid(torch.arange(height, device=device) + h0, torch.arange(width, device=device) + w0,
                            indexing="ij")
    return torch.full_like(hh, t_index).flatten(), hh.flatten(), ww.flatten()


def rope_from_positions(pe, t: torch.Tensor, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """VideoRopePosition3DEmb.generate_embeddings for explicit per-token (t, h, w) positions."""
    device = t.device
    h_freqs = 1.0 / ((10000.0 * pe.h_ntk_factor) ** pe.dim_spatial_range.to(device))
    w_freqs = 1.0 / ((10000.0 * pe.w_ntk_factor) ** pe.dim_spatial_range.to(device))
    t_freqs = 1.0 / ((10000.0 * pe.t_ntk_factor) ** pe.dim_temporal_range.to(device))

    def rotations(pos, freqs):
        angle = torch.outer(pos.float(), freqs)
        return torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], dim=-1)

    emb = torch.cat([rotations(t, t_freqs), rotations(h, h_freqs), rotations(w, w_freqs)], dim=-2)
    return emb.reshape(*emb.shape[:-1], 2, 2).float()


def ref_forward(dm, x: torch.Tensor, sigma: torch.Tensor, context: torch.Tensor, ref: torch.Tensor | None = None,
                ref_offset: tuple[int, int] = (0, 0), ref_sigma: float = 0.0, checkpoint_blocks: int = 0,
                refs: list[tuple[torch.Tensor, int, tuple[int, int]]] | None = None,
                ref_bias: torch.Tensor | None = None) -> torch.Tensor:
    """Velocity prediction for x [B, C, 1, H, W] (model-space latents), optionally reading reference frames.

    refs lists (latent [B, C, 1, H, W], RoPE t index, (h, w) RoPE shift in patch units) per reference
    frame; ref / ref_offset is the single-reference shorthand (t index 1). ref_bias [B, D] is added to
    every reference token after patch embedding (the v2 adapter's style-transfer embedding). The first
    checkpoint_blocks blocks recompute their activations in backward (gradient checkpointing).
    """
    if ref is not None:
        refs = [(ref, 1, ref_offset)] + list(refs or [])
    refs = refs or []
    batch = x.shape[0]
    frames = torch.cat([x] + [r.to(x) for r, _, _ in refs], dim=2)
    x_emb, _, extra = dm.prepare_embedded_sequence(frames, fps=None, padding_mask=None)
    if extra is not None:
        raise NotImplementedError("extra per-block position embeddings are not supported")
    if ref_bias is not None and refs:
        bias = ref_bias.to(x_emb.dtype)[:, None, None, None, :]
        x_emb = torch.cat([x_emb[:, :1], x_emb[:, 1:] + bias], dim=1)
    _, _, hp, wp, _ = x_emb.shape
    positions = [frame_positions(0, hp, wp, 0, 0, x.device)]
    for _, t_index, (h0, w0) in refs:
        positions.append(frame_positions(t_index, hp, wp, h0, w0, x.device))
    rope = rope_from_positions(dm.pos_embedder, *(torch.cat(axis) for axis in zip(*positions)))

    sigma = sigma.to(device=x.device, dtype=torch.float32).reshape(batch)
    timesteps = torch.stack([sigma] + [torch.full_like(sigma, ref_sigma)] * len(refs), dim=1)
    t_emb, adaln = dm.t_embedder[1](dm.t_embedder[0](timesteps).to(x_emb.dtype))
    t_emb = dm.t_embedding_norm(t_emb)
    kwargs = {"rope_emb_L_1_1_D": rope.unsqueeze(1).unsqueeze(0), "adaln_lora_B_T_3D": adaln,
              "extra_per_block_pos_emb": None}
    for index, block in enumerate(dm.blocks):
        if index < checkpoint_blocks:
            x_emb = torch.utils.checkpoint.checkpoint(block, x_emb, t_emb, context, use_reentrant=False,
                                                      transformer_options={}, **kwargs)
        else:
            x_emb = block(x_emb, t_emb, context, transformer_options={}, **kwargs)
    out = dm.final_layer(x_emb.to(context.dtype), t_emb, adaln_lora_B_T_3D=adaln)
    return dm.unpatchify(out)[:, :, :1]


def default_ref_offset(latent_height: int, cfg: dict, patch: int = 2) -> tuple[int, int]:
    hp = latent_height // patch
    return round(cfg.get("ref_offset_h", 1.0) * hp), int(cfg.get("ref_offset_w", 0))


SLOTS = ("character", "style")  # two-reference adapter: frame order after the target


def slot_frames(slot_refs: dict[str, torch.Tensor | None], latent_height: int, cfg: dict, jitter: list[int] | None = None,
                patch: int = 2) -> list[tuple[torch.Tensor, int, tuple[int, int]]]:
    """ref_forward frames for the filled slots of a two-reference adapter (cfg = config["adapter"]).

    Each slot has its own RoPE t index (cfg["slots"][slot]["t"]); its (h, w) shift is offset_h target
    heights (+ an optional per-slot jitter in patches) and offset_w patches, as for the single reference.
    """
    hp = latent_height // patch
    frames = []
    for i, slot in enumerate(SLOTS):
        latent = slot_refs.get(slot)
        if latent is None:
            continue
        spec = cfg["slots"][slot]
        offset = (round(spec["offset_h"] * hp) + (jitter[i] if jitter else 0), int(spec.get("offset_w", 0)))
        frames.append((latent, int(spec["t"]), offset))
    return frames


@torch.inference_mode()
def self_test(dm, device: torch.device) -> dict:
    """Check that ref_forward reproduces ComfyUI's MiniTrainDIT forward and that gradients flow."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(2, 16, 1, 16, 24, generator=gen).to(device, torch.bfloat16)
    ctx = torch.randn(2, 512, 1024, generator=gen).to(device, torch.bfloat16) * 0.1
    sigma = torch.tensor([0.9, 0.3], device=device)
    ours = ref_forward(dm, x, sigma, ctx).float()
    theirs = dm(x, sigma, ctx).float()
    diff = (ours - theirs).abs().max().item()
    scale = theirs.abs().max().item()
    if diff > 1e-3 * max(1.0, scale):
        raise RuntimeError(f"ref_forward differs from ComfyUI forward: max diff {diff} (scale {scale})")
    return {"max_abs_diff": diff, "output_scale": scale}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.lora_down = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_up = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))
        self.scale = alpha / rank
        self.alpha = alpha
        self.enabled = True

    def forward(self, x):
        out = self.base(x)
        if not self.enabled:
            return out
        return out + F.linear(F.linear(x, self.lora_down.to(x.dtype)), self.lora_up.to(x.dtype)) * self.scale


def inject_lora(dm, rank: int, alpha: float, targets=LORA_TARGETS) -> dict[str, LoRALinear]:
    modules = {}
    for index, block in enumerate(dm.blocks):
        for target in targets:
            parent_name, attr = target.rsplit(".", 1)
            parent = block.get_submodule(parent_name)
            lora = LoRALinear(getattr(parent, attr), rank, alpha)
            setattr(parent, attr, lora)
            modules[f"blocks.{index}.{target}"] = lora
    return modules


def set_lora_enabled(modules: dict[str, LoRALinear], enabled: bool) -> None:
    for module in modules.values():
        module.enabled = enabled


STYLE_EMBED_KEY = "anima_ref_adapter.style_embed"  # v2: [D] vector, scaled by style_transfer


def save_lora(modules: dict[str, LoRALinear], path: Path, metadata: dict, extra: dict[str, torch.Tensor] | None = None) -> None:
    """ComfyUI-loadable LoRA (diffusion_model.<name>.lora_down/lora_up/alpha) plus adapter metadata
    and extra adapter tensors (keys outside diffusion_model.*, which LoRA loaders ignore)."""
    tensors = {k: v.detach().float().cpu().contiguous() for k, v in (extra or {}).items()}
    for name, module in modules.items():
        key = f"diffusion_model.{name}"
        tensors[f"{key}.lora_down.weight"] = module.lora_down.detach().to(torch.bfloat16).cpu().contiguous()
        tensors[f"{key}.lora_up.weight"] = module.lora_up.detach().to(torch.bfloat16).cpu().contiguous()
        tensors[f"{key}.alpha"] = torch.tensor(float(module.alpha))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={"anima_ref_adapter": json.dumps(metadata)})


def load_lora(modules: dict[str, LoRALinear], path: Path) -> dict:
    tensors = load_file(str(path))
    with torch.no_grad():
        for name, module in modules.items():
            key = f"diffusion_model.{name}"
            module.lora_down.copy_(tensors[f"{key}.lora_down.weight"].float())
            module.lora_up.copy_(tensors[f"{key}.lora_up.weight"].float())
    from safetensors import safe_open
    with safe_open(str(path), "pt") as handle:
        return json.loads((handle.metadata() or {}).get("anima_ref_adapter", "{}"))


def load_style_embed(path: Path) -> torch.Tensor | None:
    """The v2 adapter's style-transfer embedding [D], or None for adapters without one."""
    from safetensors import safe_open
    with safe_open(str(path), "pt") as handle:
        return handle.get_tensor(STYLE_EMBED_KEY).float() if STYLE_EMBED_KEY in handle.keys() else None


def flow_sigmas(steps: int, shift: float) -> torch.Tensor:
    """Anima's shift-3 flow schedule, uniform in timestep (ComfyUI simple/sgm_uniform), ending at 0."""
    t = torch.linspace(1.0, 0.0, steps + 1)
    return shift * t / (1 + (shift - 1) * t)


@torch.inference_mode()
def sample_euler(dm, ctx_pos: torch.Tensor, ctx_neg: torch.Tensor, shape: tuple, seed: int, steps: int, cfg: float,
                 shift: float, device, ref: torch.Tensor | None = None, ref_offset=(0, 0), frames=None,
                 ref_bias: torch.Tensor | None = None) -> torch.Tensor:
    """Euler flow sampling with CFG; the reference (if any) is given to both branches. Returns model-space x0.

    frames: ref_forward reference frames (see slot_frames) instead of the single ref.
    ref_bias: [D] added to the reference tokens (v2 style-transfer embedding times its strength)."""
    noise = torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32)
    x = noise.to(device)
    context = pad_contexts([ctx_pos, ctx_neg]).to(device, torch.bfloat16)
    if ref is not None:
        frames = [(ref, 1, ref_offset)] + list(frames or [])
    frames = [(r.to(device, torch.bfloat16).expand(2, *r.shape[1:]), t, o) for r, t, o in frames or []]
    sigmas = flow_sigmas(steps, shift)
    bias = ref_bias.to(device).reshape(1, -1).expand(2, -1) if ref_bias is not None else None
    for i in range(steps):
        sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
        v = ref_forward(dm, x.to(torch.bfloat16).expand(2, *x.shape[1:]), torch.full((2,), sigma, device=device),
                        context, refs=frames, ref_bias=bias).float()
        v = v[1:2] + cfg * (v[0:1] - v[1:2])
        x = x + (sigma_next - sigma) * v
    return x


def _flat_dims(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(range(1, x.ndim))


def rescale_guidance(x0_guided: torch.Tensor, x0_cond: torch.Tensor, phi: float) -> torch.Tensor:
    """CFG rescale (Lin et al. 2024) in x0 space: match the guided std to the conditional std, blend by phi."""
    std_ratio = x0_cond.std(dim=_flat_dims(x0_cond), keepdim=True) / x0_guided.std(dim=_flat_dims(x0_guided), keepdim=True)
    return phi * x0_guided * std_ratio + (1 - phi) * x0_guided


def apg_guidance(x0_guided: torch.Tensor, x0_cond: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
    """Adaptive projected guidance (Sadat et al. 2024) in x0 space, applied to the total guidance update:
    the component of (guided - cond) parallel to cond, which mostly raises saturation, is scaled by eta."""
    update = x0_guided - x0_cond
    dims = _flat_dims(x0_cond)
    unit = torch.nn.functional.normalize(x0_cond.double().flatten(1), dim=1).view_as(x0_cond)
    parallel = (update.double() * unit).sum(dim=dims, keepdim=True) * unit
    orthogonal = update.double() - parallel
    return x0_cond + (orthogonal + eta * parallel).to(x0_cond.dtype)


@torch.inference_mode()
def sample_euler_ref_guidance(dm, lora_modules, ctx_pos: torch.Tensor, ctx_neg: torch.Tensor, shape: tuple, seed: int,
                              steps: int, cfg_text: float, cfg_ref: float, shift: float, device, ref: torch.Tensor,
                              ref_offset=(0, 0), guidance: str = "none", rescale_phi: float = 0.7,
                              apg_eta: float = 0.0) -> torch.Tensor:
    """Euler flow sampling with separate text and reference guidance (InstructPix2Pix-style):

        v = v(neg) + cfg_ref * (v(neg, ref) - v(neg)) + cfg_text * (v(pos, ref) - v(neg, ref))

    v(neg) is the base model (LoRA off, no reference). cfg_ref = 1 is sample_euler with the
    reference in both branches; cfg_ref = cfg_text drops the reference from the uncond branch.
    guidance "rescale" / "apg" correct the combined result against the conditional branch
    v(pos, ref), in x0 space (x0 = x - sigma * v).
    """
    noise = torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32)
    x = noise.to(device)
    context = pad_contexts([ctx_pos, ctx_neg]).to(device, torch.bfloat16)
    refs = ref.to(device, torch.bfloat16).expand(2, *ref.shape[1:])
    sigmas = flow_sigmas(steps, shift)
    for i in range(steps):
        sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
        xb = x.to(torch.bfloat16)
        set_lora_enabled(lora_modules, True)
        v_ref = ref_forward(dm, xb.expand(2, *x.shape[1:]), torch.full((2,), sigma, device=device), context,
                            ref=refs, ref_offset=ref_offset).float()
        set_lora_enabled(lora_modules, False)
        v_base = ref_forward(dm, xb, torch.full((1,), sigma, device=device), context[1:2]).float()
        v = v_base + cfg_ref * (v_ref[1:2] - v_base) + cfg_text * (v_ref[0:1] - v_ref[1:2])
        if guidance != "none":
            x0_guided, x0_cond = x - sigma * v, x - sigma * v_ref[0:1]
            if guidance == "rescale":
                x0 = rescale_guidance(x0_guided, x0_cond, rescale_phi)
            elif guidance == "apg":
                x0 = apg_guidance(x0_guided, x0_cond, apg_eta)
            else:
                raise ValueError(f"unknown guidance {guidance}")
            v = (x - x0) / sigma
        x = x + (sigma_next - sigma) * v
    set_lora_enabled(lora_modules, True)
    return x
