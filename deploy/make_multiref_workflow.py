"""ComfyUI test workflow (UI format) that runs several reference images side by side with one adapter.

Shared: model, adapter loader, CLIP, VAE, negative prompt, sampler, scheduler, seed. One chain per
reference: LoadImage -> Anima Ref Latent -> positive prompt -> Anima Ref Guider -> sampler -> decode -> save.

  python deploy/make_multiref_workflow.py --lora anima_ref_adapter/copy_asym_step01500.safetensors \
      --ui "<ComfyUI>/user/default/workflows/Anima Ref 비대칭 테스트.json"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_test_workflow import NEGATIVE, NODES, to_ui

QUALITY = "masterpiece, best quality, score_7, newest, highres, absurdres, safe, "
# (image in ComfyUI/input, output width, height, scene prompt after the quality tags)
REFS = [
    ("anima_ref_test_1_bunny.webp", 896, 1152, "1girl, solo, cowboy shot, standing, looking at viewer, smile, holding cup, cafe, indoors, day"),
    ("anima_ref_test_2_angel.webp", 1024, 1024, "1girl, solo, upper body, sitting, reading book, library, indoors, soft lighting"),
    ("anima_ref_test_3_throne.webp", 832, 1216, "1boy, solo, full body, running, outdoors, park, sunset"),
    ("anima_ref_test_4_chibi.jpg", 832, 1216, "1girl, solo, full body, standing, waving, open mouth, smile, beach, day"),
]
NOTE = """## Anima Ref 어댑터 테스트 (비대칭 attention, 캐릭터 + 그림체)

참조 이미지 4장을 나란히 생성합니다. 각 줄 = 참조 이미지 → 생성 크기 → 장면 프롬프트 → 가이더 → 저장.

- **어댑터**: `copy_asym` 파일럿. 참조의 **캐릭터와 그림체**를 함께 가져옵니다. 자세·배경·구도는 프롬프트를 따릅니다.
- **구조**: 참조 토큰은 참조끼리만 보고 프롬프트도 읽지 않습니다. 그래서 참조는 생성 시작 때 한 번만 계산되어 모든 스텝과 CFG 양쪽에서 재사용됩니다(스텝당 약 2배 빠름). 참조 크기는 생성 크기와 달라도 됩니다.
- **프롬프트**: 캐릭터 이름, 작가 태그, 외모·복장 태그 없이 퀄리티 + 성별 + `solo` + 장면 태그만 씁니다.
- **Anima Ref Guider**: `cfg` 4.5(텍스트), `ref_guidance` 4.5(참조; 1이면 약하고 빠름, 높을수록 참조를 강하게 따르고 채도가 오름), `correction` apg.
- **학습 범위**: 0.56~1MP에서 학습했습니다. 1MP보다 큰 크기는 학습하지 않았습니다.
- 생성 순서: 큐에 넣으면 4장이 차례로 생성됩니다. 원하지 않는 줄은 노드를 선택하고 Ctrl+B(bypass)로 끄세요.
"""

# Style-only adapter: the reference gives the drawing style; the prompt names the character to draw.
STYLE_REFS = [
    ("anima_ref_test_1_bunny.webp", 896, 1152, "1girl, solo, hatsune miku, vocaloid, aqua hair, twintails, aqua eyes, detached sleeves, necktie, cowboy shot, standing, smile, city, night"),
    ("anima_ref_test_2_angel.webp", 1024, 1024, "1girl, solo, hakurei reimu, touhou, brown hair, red bow, hair tubes, red skirt, detached sleeves, upper body, smile, shrine, day"),
    ("anima_ref_test_3_throne.webp", 832, 1216, "1boy, solo, kamado tanjirou, kimetsu no yaiba, red hair, scar on forehead, checkered haori, full body, standing, forest"),
    ("anima_ref_test_4_chibi.jpg", 832, 1216, "1girl, solo, frieren, sousou no frieren, white hair, twintails, green eyes, white capelet, full body, standing, flower field"),
]
STYLE_NOTE = """## Anima Ref 어댑터 테스트 (그림체만)

참조 이미지 4장의 **그림체만** 가져오고, **캐릭터는 프롬프트**로 정합니다. 각 줄 = 참조 이미지 → 생성 크기 → 프롬프트 → 가이더 → 저장.

- **어댑터**: `style_asym_r64`. 같은 작가의 다른 캐릭터를 참조로, 프롬프트에 캐릭터·외모·의상 태그를 넣고 작가 태그만 빼서 학습했습니다.
- **프롬프트**: 퀄리티 + 성별 + `solo` + 캐릭터 이름·작품 + 외모·의상 태그 + 장면 태그. 작가 태그는 넣지 않습니다.
- **비교**: 같은 프롬프트를 어댑터 없이 생성해 보면 그림체가 얼마나 바뀌었는지 보입니다(어댑터 로더 strength 0).
- **Anima Ref Guider**: `cfg` 4.5, `ref_guidance` 4.5, `correction` apg.
- 아직 짧게 학습한 버전입니다(500/1000 step). 참조의 캐릭터 요소(의상, 소품)가 섞여 들어오면 ref_guidance를 낮춰 보세요.
"""

SHARED_IDS = {1, 2, 3, 4, 5, 6, 10, 12, 13, 14}
CHAIN_IDS = {7, 8, 9, 11, 15, 16, 17}


def build(lora: str, refs=REFS) -> list:
    base = {n[0]: n for n in NODES}
    nodes = []
    for nid in sorted(SHARED_IDS):
        _, ntype, title, widgets, inputs, outputs, pos, size = base[nid]
        if ntype == "AnimaRefAdapterLoader":
            widgets = [lora, *widgets[1:]]
        if ntype == "CLIPTextEncode":
            widgets = [NEGATIVE]
        nodes.append((nid, ntype, title, widgets, inputs, outputs, pos, size))
    for k, (image, width, height, scene) in enumerate(refs):
        offset = 100 * (k + 1)
        remap = lambda src: src + offset if src in CHAIN_IDS else src  # noqa: E731
        for nid in sorted(CHAIN_IDS):
            _, ntype, title, widgets, inputs, outputs, pos, size = base[nid]
            if ntype == "LoadImage":
                widgets, title = [image, "image"], f"Reference {k + 1}"
            elif ntype == "AnimaRefLatent":
                widgets = [width, height, 255]
            elif ntype == "CLIPTextEncode":
                widgets, title = [QUALITY + scene], f"Positive {k + 1} (scene only)"
            elif ntype == "SaveImage":
                widgets, size = [f"anima_ref_adapter/asym_test_{k + 1}"], [420, 560]
            inputs = {name: (remap(src), slot, typ) for name, (src, slot, typ) in inputs.items()}
            # chains stacked below the shared nodes, one row each
            row_pos = [pos[0], 900 + k * 700 + (pos[1] - 40) % 600] if nid != 17 else [1890, 900 + k * 700]
            nodes.append((nid + offset, ntype, title, widgets, inputs, outputs, row_pos, size))
    return nodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", required=True, help="adapter file name under models/loras")
    parser.add_argument("--ui", type=Path, required=True)
    parser.add_argument("--style", action="store_true", help="style-only adapter: prompts name the character")
    args = parser.parse_args()
    workflow = to_ui(build(args.lora, STYLE_REFS if args.style else REFS), STYLE_NOTE if args.style else NOTE)
    for node in workflow["nodes"]:
        if node["type"] == "MarkdownNote":
            node["pos"], node["size"] = [750, 480], [800, 400]
    args.ui.parent.mkdir(parents=True, exist_ok=True)
    args.ui.write_text(json.dumps(workflow, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(workflow['nodes'])} nodes, {len(workflow['links'])} links -> {args.ui}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
