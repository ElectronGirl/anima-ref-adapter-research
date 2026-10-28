"""Write the ComfyUI test workflow for the reference adapter (UI format) and its API-format twin.

  python deploy/make_test_workflow.py --ui "<ComfyUI>/user/default/workflows/Anima Ref Adapter 테스트.json" \
      --api runs/try/workflow_api.json
  # with Anima Safe PAG (no turbo LoRA: the adapter was trained on the base model; no SamplerSPEED)
  python deploy/make_test_workflow.py --accelerated --ui ".../Anima Ref Adapter 테스트 (가속).json"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NEGATIVE = ("worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, chromatic aberration, "
            "bad anatomy, bad hands, extra arms, extra fingers, missing fingers, text, watermark, signature, "
            "artist name, multiple views, split screen, comic, 2koma, 4koma")
POSITIVE = ("masterpiece, best quality, score_7, newest, highres, absurdres, safe, 1girl, solo, "
            "full body, walking, waving, open mouth, smile, city, street, night")
NOTE = """## Anima Ref Adapter 테스트

참조 이미지의 **캐릭터와 그림체**를 가져오고, **자세·배경·구도는 프롬프트**를 따릅니다.

- **프롬프트**: 캐릭터 이름, 작가 태그, 외모·복장 태그 없이 퀄리티 + 성별 + `solo` + 장면 태그(구도, 자세, 표정, 배경, 조명)만 씁니다. 외모 태그를 일부 넣어도 됩니다.
- **Anima Ref Latent**: 참조 이미지를 생성 크기(width/height)에 맞춰 비율 유지 + 흰 여백으로 넣습니다. 생성 크기는 여기서 정합니다(빈 latent도 여기서 나옵니다).
- **Anima Ref Guider**
  - `cfg`: 텍스트 가이던스 (4.5)
  - `ref_guidance`: 참조 가이던스. 1 = 참조를 CFG 양쪽에 넣음(약함), 4.5~6 권장. 높을수록 캐릭터 세부와 그림체가 강해지고 채도가 오릅니다.
  - `correction`: `apg`(기본) 또는 `rescale`(phi 0.5)로 높은 가이던스의 과포화를 줄입니다.
- **Anima Ref Adapter Loader**: 어댑터 LoRA. `ref_offset_h`는 학습값 1.0 그대로 두세요.
- 모델: anima base v1.0 (어댑터는 turbo LoRA 없이 학습), bf16 연산, SageAttention, euler 30 steps.
- 참조 가이던스가 1이 아니면 스텝마다 모델을 3번 호출합니다(일반 CFG의 1.5배 시간).

파일럿 어댑터(캐릭터 300명, 2,000 steps)라서 작은 무늬나 피부색 같은 세부는 약합니다.
"""

# (id, type, title, widgets, {input: (src_id, src_slot, type)}, [(output, type)], pos, size)
NODES = [
    (1, "UNETLoader", "Base model", ["anima_baseV10.safetensors", "default"], {}, [("MODEL", "MODEL")], [40, 40], [300, 90]),
    (2, "CLIPLoader", None, ["qwen_3_06b_base.safetensors", "stable_diffusion", "default"], {}, [("CLIP", "CLIP")],
     [40, 170], [300, 110]),
    (3, "VAELoader", None, ["qwenImageVAE_qwenImageVAE.safetensors"], {}, [("VAE", "VAE")], [40, 320], [300, 60]),
    (4, "AnimaRefAdapterLoader", "Ref adapter LoRA", ["anima_ref_adapter_pilot_r32_s2000.safetensors", 1.0, 1.0],
     {"model": (1, 0, "MODEL")}, [("MODEL", "MODEL")], [380, 40], [330, 110]),
    (5, "ModelComputeDtype", None, ["bf16"], {"model": (4, 0, "MODEL")}, [("MODEL", "MODEL")], [380, 190], [330, 60]),
    (6, "PathchSageAttentionKJ", None, ["auto", False], {"model": (5, 0, "MODEL")}, [("MODEL", "MODEL")],
     [380, 290], [330, 80]),
    (7, "LoadImage", "Reference image", ["anima_ref_test_sailor.png", "image"], {},
     [("IMAGE", "IMAGE"), ("MASK", "MASK")], [40, 460], [320, 380]),
    (8, "AnimaRefLatent", "Reference + output size", [896, 1152, 255], {"image": (7, 0, "IMAGE"), "vae": (3, 0, "VAE")},
     [("ref_latent", "LATENT"), ("empty_latent", "LATENT"), ("ref_image", "IMAGE")], [380, 460], [330, 150]),
    (9, "CLIPTextEncode", "Positive (scene only)", [POSITIVE], {"clip": (2, 0, "CLIP")},
     [("CONDITIONING", "CONDITIONING")], [750, 40], [420, 200]),
    (10, "CLIPTextEncode", "Negative", [NEGATIVE], {"clip": (2, 0, "CLIP")}, [("CONDITIONING", "CONDITIONING")],
     [750, 280], [420, 160]),
    (11, "AnimaRefGuider", None, [4.5, 4.5, "apg", 0.5],
     {"model": (6, 0, "MODEL"), "positive": (9, 0, "CONDITIONING"), "negative": (10, 0, "CONDITIONING"),
      "ref_latent": (8, 0, "LATENT")}, [("GUIDER", "GUIDER")], [1210, 40], [320, 170]),
    (12, "KSamplerSelect", None, ["euler"], {}, [("SAMPLER", "SAMPLER")], [1210, 250], [320, 60]),
    (13, "BasicScheduler", None, ["simple", 30, 1.0], {"model": (6, 0, "MODEL")}, [("SIGMAS", "SIGMAS")],
     [1210, 350], [320, 110]),
    (14, "RandomNoise", None, [20261027, "randomize"], {}, [("NOISE", "NOISE")], [1210, 500], [320, 90]),
    (15, "SamplerCustomAdvanced", None, [],
     {"noise": (14, 0, "NOISE"), "guider": (11, 0, "GUIDER"), "sampler": (12, 0, "SAMPLER"),
      "sigmas": (13, 0, "SIGMAS"), "latent_image": (8, 1, "LATENT")},
     [("output", "LATENT"), ("denoised_output", "LATENT")], [1570, 40], [280, 130]),
    (16, "VAEDecode", None, [], {"samples": (15, 0, "LATENT"), "vae": (3, 0, "VAE")}, [("IMAGE", "IMAGE")],
     [1570, 220], [280, 50]),
    (17, "SaveImage", None, ["anima_ref_adapter/test"], {"images": (16, 0, "IMAGE")}, [], [1890, 40], [520, 640]),
    (18, "PreviewImage", "Reference as fed (letterboxed)", [], {"images": (8, 2, "IMAGE")}, [], [750, 480], [420, 520]),
]

ACCEL_NOTE = """
### PAG
- **Anima Safe PAG**: 블록 18 self-attention을 약하게 흐린 예측과의 차이로 디테일을 보강합니다(scale 1.3, 0.3~0.8 구간). 활성 구간에서 스텝당 모델 호출이 1번 더 늡니다.
- **SamplerSPEED는 쓰지 마세요.** 초반 저해상도 단계에서 참조를 줄여 넣어야 하는데 어댑터가 그걸 읽지 못해, 참조 가이던스가 높으면 그림이 깨지고 낮으면 참조가 거의 반영되지 않습니다.
- turbo LoRA는 쓰지 않습니다(어댑터가 base 모델로 학습됨). torch compile은 3060 Ti에서 이득이 작고 첫 실행이 오래 걸려 넣지 않았습니다.
"""


NOTE_EN = """## Anima Reference Adapter

Takes the **character and drawing style** from the reference image; **pose, background and framing** come from the prompt.

- **Prompt**: no character name, artist tag or appearance/outfit tags. Write quality tags, gender, `solo` and scene tags (framing, pose, expression, background, lighting). Adding a few appearance tags is fine.
- **Anima Ref Latent**: fits the reference into the output size (aspect kept, white padding). The output size is set here (it also returns the empty latent).
- **Anima Ref Guider**
  - `cfg`: text guidance (4.5)
  - `ref_guidance`: reference guidance. 1 = reference in both CFG branches (weak); 4.5-6 recommended. Higher gives more character detail and style, and more saturation.
  - `correction`: `apg` (default) or `rescale` (phi 0.5) tames over-saturation at high guidance.
- **Anima Ref Adapter Loader**: the adapter LoRA. Keep `ref_offset_h` at 1.0 (training value).
- Base: Anima base v1.0 without the turbo LoRA (the adapter was trained on the base model), bf16 compute, SageAttention, euler 30 steps.
- With ref_guidance != 1 each step runs the model 3 times (1.5x a normal CFG step).

This is a pilot adapter (300 characters, 2,000 steps): small patterns, scars and skin tones are weak.
"""
ACCEL_NOTE_EN = """
### PAG
- **Anima Safe PAG** adds detail from the difference to a prediction with block 18 self-attention blurred (scale 1.3, 0.3-0.8 of the schedule). One extra model call per step while active.
- **Avoid SamplerSPEED for now**: its half-resolution stage needs a half-size reference, which this adapter never saw in training; results are unreliable.
"""


def accelerated(nodes):
    """Put Anima Safe PAG in front of the guider. (SamplerSPEED breaks the reference: its half-resolution
    stage needs a downscaled reference latent, which the adapter never saw in training.)"""
    out = []
    for nid, ntype, title, widgets, inputs, outputs, pos, size in nodes:
        if ntype in ("AnimaRefGuider", "BasicScheduler"):
            inputs = {**inputs, "model": (20, 0, "MODEL")}
        out.append((nid, ntype, title, widgets, inputs, outputs, pos, size))
    out.append((20, "AnimaSafePAG", "Anima Safe PAG", [1.3, "18", 0.75, "", 0.3, 0.8, 0.2, "full"],
                {"model": (6, 0, "MODEL")}, [("MODEL", "MODEL")], [380, 650], [330, 230]))
    return out


WIDGET_NAMES = {
    "SamplerSPEED": ["base_sampler", "transform", "mode", "model_preset", "scales", "delta", "manual_sigmas",
                     "spectrum_A", "spectrum_beta", "seed", None],
    "AnimaSafePAG": ["scale", "block_indices", "perturbation_strength", "head_indices", "start_percent",
                     "end_percent", "rescale", "rescale_mode"],
    "UNETLoader": ["unet_name", "weight_dtype"], "CLIPLoader": ["clip_name", "type", "device"], "VAELoader": ["vae_name"],
    "AnimaRefAdapterLoader": ["lora_name", "strength", "ref_offset_h"], "ModelComputeDtype": ["dtype"],
    "PathchSageAttentionKJ": ["sage_attention", "allow_compile"], "LoadImage": ["image", None],
    "AnimaRefLatent": ["width", "height", "fill"], "CLIPTextEncode": ["text"],
    "AnimaRefGuider": ["cfg", "ref_guidance", "correction", "rescale_phi"], "KSamplerSelect": ["sampler_name"],
    "BasicScheduler": ["scheduler", "steps", "denoise"], "RandomNoise": ["noise_seed", None],
    "SamplerCustomAdvanced": [], "VAEDecode": [], "SaveImage": ["filename_prefix"], "PreviewImage": [],
}


def to_ui(nodes, note_text=NOTE) -> dict:
    links, ui_nodes, outgoing = [], [], {}
    for order, (nid, ntype, title, widgets, inputs, outputs, pos, size) in enumerate(nodes):
        ui_inputs = []
        for name, (src, slot, typ) in inputs.items():
            link_id = len(links) + 1
            links.append([link_id, src, slot, nid, len(ui_inputs), typ])
            outgoing.setdefault((src, slot), []).append(link_id)
            ui_inputs.append({"name": name, "type": typ, "link": link_id})
        node = {"id": nid, "type": ntype, "pos": pos, "size": size, "flags": {}, "order": order, "mode": 0,
                "inputs": ui_inputs, "outputs": [], "properties": {"Node name for S&R": ntype}, "widgets_values": widgets}
        if title:
            node["title"] = title
        ui_nodes.append((node, outputs))
    for node, outputs in ui_nodes:
        node["outputs"] = [{"name": name, "type": typ, "slot_index": i, "links": outgoing.get((node["id"], i))}
                           for i, (name, typ) in enumerate(outputs)]
    note = {"id": 19, "type": "MarkdownNote", "pos": [40, 880], "size": [670, 620], "flags": {}, "order": len(nodes),
            "mode": 0, "inputs": [], "outputs": [], "title": "사용법" if note_text.startswith(NOTE[:20]) else "How to use",
            "properties": {}, "widgets_values": [note_text],
            "color": "#432", "bgcolor": "#653"}
    return {"last_node_id": max([19] + [n["id"] for n, _ in ui_nodes]), "last_link_id": len(links), "nodes": [n for n, _ in ui_nodes] + [note], "links": links,
            "groups": [], "config": {}, "extra": {"ds": {"scale": 0.75, "offset": [0, 0]}}, "version": 0.4}


def to_api(nodes) -> dict:
    api = {}
    for nid, ntype, _, widgets, inputs, _, _, _ in nodes:
        values = {name: value for name, value in zip(WIDGET_NAMES[ntype], widgets) if name}
        values.update({name: [str(src), slot] for name, (src, slot, _) in inputs.items()})
        api[str(nid)] = {"class_type": ntype, "inputs": values}
    return api


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui", type=Path, required=True)
    parser.add_argument("--api", type=Path)
    parser.add_argument("--accelerated", action="store_true", help="add Anima Safe PAG")
    parser.add_argument("--lang", choices=["ko", "en"], default="ko", help="language of the usage note")
    parser.add_argument("--lora", help="adapter LoRA file name (default: the pilot file name)")
    parser.add_argument("--ref-image", help="LoadImage file name")
    args = parser.parse_args()
    note, accel_note = (NOTE, ACCEL_NOTE) if args.lang == "ko" else (NOTE_EN, ACCEL_NOTE_EN)
    nodes, note = (accelerated(NODES), note + accel_note) if args.accelerated else (NODES, note)
    first_widget = {"AnimaRefAdapterLoader": args.lora, "LoadImage": args.ref_image}
    nodes = [(nid, ntype, title, [first_widget[ntype], *widgets[1:]] if first_widget.get(ntype) else widgets,
              inputs, outputs, pos, size)
             for nid, ntype, title, widgets, inputs, outputs, pos, size in nodes]
    args.ui.write_text(json.dumps(to_ui(nodes, note), ensure_ascii=False, indent=1), encoding="utf-8")
    if args.api:
        args.api.parent.mkdir(parents=True, exist_ok=True)
        args.api.write_text(json.dumps(to_api(nodes), ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
