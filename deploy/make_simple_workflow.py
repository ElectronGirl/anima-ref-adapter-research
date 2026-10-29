"""Write the release workflows: the Anima default text-to-image graph plus one Anima Ref Adapter node.

  python deploy/make_simple_workflow.py --out workflows
  python deploy/make_simple_workflow.py --out "<ComfyUI>/user/default/workflows" --lang ko --lora "anima_ref_adapter\\copy_asym_r64_step01000.safetensors"

Writes anima_ref_adapter.json (one reference) and anima_ref_adapter_two_characters.json (two references
stitched side by side). Only core ComfyUI nodes besides the adapter node.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_test_workflow import to_ui

POSITIVE = "masterpiece, best quality, score_7, safe, 1girl, solo, cowboy shot, sitting, cafe, holding cup, smile, looking at viewer, indoors, window, day"
POSITIVE_TWO = ("masterpiece, best quality, score_7, safe, 2girls, the girl on the left is a policewoman with black hair over one eye, "
                "the girl on the right is a mecha girl with aqua hair and mechanical wings, standing side by side, peace sign, smile, city, street, day")
NEGATIVE = "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration"

NOTE_EN = """## Anima Reference Adapter

The Anima default workflow plus one node. The reference gives **who** is drawn and much of **how**; the prompt gives pose, scene and framing.

- **Prompt**: quality tags, `1girl` / `1boy`, `solo`, then scene tags. No character name, artist tag or outfit tags needed.
- **Anima Ref Adapter**: put it after the model loader and any LoRAs.
  - `ref_guidance` 4.5: how hard to push toward the reference (1 = no extra model call, weaker).
  - `correction` apg: keeps colors in check.
  - `ref_end` 1.0: set ~0.6 to save time with nearly the same image.
- Any sampler works. Defaults here: er_sde, simple, 30 steps, cfg 4.5.
- **Faster**: a turbo LoRA in front of the adapter with 10 steps, cfg 1, ref_guidance 1; SageAttention; EasyCache.
"""
NOTE_TWO_EN = """
### Two characters
Stitch the two references side by side (core **Image Stitch** node) and write in the prompt who is on the left and who is on the right, with a few words each. Use `2girls` / `1boy, 1girl` instead of `solo`, and a landscape output.
"""
NOTE_KO = """## Anima Ref Adapter

Anima 기본 워크플로에 노드 하나를 더한 것입니다. 참조 이미지에서 **캐릭터와 그림체**를, 프롬프트에서 자세·장면·구도를 가져옵니다.

- **프롬프트**: 퀄리티 태그, `1girl` / `1boy`, `solo`, 장면 태그. 캐릭터 이름·작가·복장 태그는 필요 없습니다.
- **Anima Ref Adapter**: 모델 로더와 다른 LoRA 뒤에 둡니다.
  - `ref_guidance` 4.5: 참조 쪽으로 미는 세기(1이면 추가 호출 없음, 약함)
  - `correction` apg: 과포화를 줄입니다.
  - `ref_end` 1.0: 0.6쯤으로 낮추면 거의 같은 그림으로 빨라집니다.
- 샘플러는 무엇이든 됩니다. 기본값: er_sde, simple, 30 steps, cfg 4.5.
- **가속**: turbo LoRA를 어댑터 앞에 넣고 10 steps, cfg 1, ref_guidance 1. SageAttention, EasyCache도 됩니다.
"""
NOTE_TWO_KO = """
### 두 캐릭터
참조 두 장을 좌우로 붙이고(기본 **Image Stitch** 노드), 프롬프트에 왼쪽·오른쪽이 누구인지 짧게 씁니다. `solo` 대신 `2girls` / `1boy, 1girl`, 가로 크기를 쓰세요.
"""


def nodes(lora, two):
    image = (7, 0, "IMAGE")
    out = [
        (1, "UNETLoader", None, ["anima-base-v1.0.safetensors", "default"], {}, [("MODEL", "MODEL")], [40, 40], [320, 90]),
        (2, "CLIPLoader", None, ["qwen_3_06b_base.safetensors", "stable_diffusion", "default"], {}, [("CLIP", "CLIP")], [40, 170], [320, 110]),
        (3, "VAELoader", None, ["qwen_image_vae.safetensors"], {}, [("VAE", "VAE")], [40, 320], [320, 60]),
        (7, "LoadImage", "Reference" + (" (left)" if two else ""), ["ref_police.webp" if two else "reference.png", "image"], {},
         [("IMAGE", "IMAGE"), ("MASK", "MASK")], [40, 420], [320, 400]),
    ]
    if two:
        out += [
            (21, "LoadImage", "Reference (right)", ["ref_mecha.webp", "image"], {}, [("IMAGE", "IMAGE"), ("MASK", "MASK")], [40, 860], [320, 400]),
            (22, "ImageStitch", None, ["right", True, 0, "white"], {"image1": (7, 0, "IMAGE"), "image2": (21, 0, "IMAGE")},
             [("IMAGE", "IMAGE")], [400, 420], [320, 130]),
        ]
        image = (22, 0, "IMAGE")
    out += [
        (4, "AnimaRefAdapter", None, [lora, 4.5, "apg", 1.0, 1.0, 1.0],
         {"model": (1, 0, "MODEL"), "vae": (3, 0, "VAE"), "image": image}, [("MODEL", "MODEL")], [400, 40], [320, 220]),
        (9, "CLIPTextEncode", "Positive", [POSITIVE_TWO if two else POSITIVE], {"clip": (2, 0, "CLIP")},
         [("CONDITIONING", "CONDITIONING")], [760, 40], [420, 200]),
        (10, "CLIPTextEncode", "Negative", [NEGATIVE], {"clip": (2, 0, "CLIP")}, [("CONDITIONING", "CONDITIONING")], [760, 280], [420, 140]),
        (20, "EmptyLatentImage", None, [1216, 832, 1] if two else [832, 1216, 1], {}, [("LATENT", "LATENT")], [760, 460], [420, 110]),
        (15, "KSampler", None, [20261029, "randomize", 30, 4.5, "er_sde", "simple", 1.0],
         {"model": (4, 0, "MODEL"), "positive": (9, 0, "CONDITIONING"), "negative": (10, 0, "CONDITIONING"),
          "latent_image": (20, 0, "LATENT")}, [("LATENT", "LATENT")], [1220, 40], [320, 270]),
        (16, "VAEDecode", None, [], {"samples": (15, 0, "LATENT"), "vae": (3, 0, "VAE")}, [("IMAGE", "IMAGE")], [1220, 350], [320, 50]),
        (17, "SaveImage", None, ["anima_ref_adapter/image"], {"images": (16, 0, "IMAGE")}, [], [1580, 40], [560, 760]),
    ]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="folder for the two workflow files")
    parser.add_argument("--lang", choices=["en", "ko"], default="en")
    parser.add_argument("--lora", default="anima_ref_adapter_v2_r64.safetensors")
    parser.add_argument("--names", nargs=2, default=["anima_ref_adapter.json", "anima_ref_adapter_two_characters.json"])
    args = parser.parse_args()
    note, note_two = (NOTE_EN, NOTE_TWO_EN) if args.lang == "en" else (NOTE_KO, NOTE_TWO_KO)
    for two, name in zip((False, True), args.names):
        workflow = to_ui(nodes(args.lora, two), note + note_two if two else note)
        workflow["nodes"][-1].update(title="How to use" if args.lang == "en" else "사용법", pos=[400, 600], size=[780, 520])
        (args.out / name).write_text(json.dumps(workflow, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
