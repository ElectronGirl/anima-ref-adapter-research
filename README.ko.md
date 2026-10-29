# Anima Reference Adapter — 연구 코드

[English](README.md) · **한국어** · [日本語](README.ja.md)

[Anima](https://huggingface.co/circlestone-labs/Anima)(Cosmos-Predict2 MiniTrainDIT, 2B)용 레퍼런스 어댑터의 학습·평가
코드입니다. 모델에 레퍼런스 이미지 한 장을 주면, 프롬프트가 묘사하는 새로운 장면에 그 캐릭터를 그리고, 경우에
따라 그 그림체로도 그립니다.

이 저장소에는 공개된 어댑터와 그 뒤의 실험에 쓰인 모든 것이 들어 있습니다. Anima로 직접 만드는 합성 데이터 생성,
텐서 준비, 학습, 평가와 분석 스크립트가 있고, 실패했거나 진행 중이던 실험과 그 수치는
[docs/experiments.md](docs/experiments.md)에 정리되어 있습니다(영어).

- ComfyUI 추론 노드(배포판): [ElectronGirl/ComfyUI-Anima-Ref-Adapter](https://github.com/ElectronGirl/ComfyUI-Anima-Ref-Adapter)
- 공개 가중치와 기술 보고서: [ElectronGirl/anima-ref-adapter](https://huggingface.co/ElectronGirl/anima-ref-adapter)

## 방법 요약

파일럿에서는 레퍼런스 이미지를 출력 크기에 맞춰 레터박스 처리하고 VAE로 인코딩한 뒤, 시간 축의 추가 프레임으로
타깃 latent 뒤에 붙였습니다(ComfyUI-Cosmos-Reference와 같은 방식). 두 가지 변경으로 이 프레임을 이어서 그릴 레이아웃이
아니라 외형의 출처로 만듭니다. 레퍼런스 프레임은 깨끗한 상태(timestep 0)로 넣고, RoPE의 h 좌표를 타깃 격자 바깥으로
타깃 높이 하나만큼 밀어서 공간적으로 대응하는 위치가 없게 합니다. 모든 블록의 attention과 MLP 투영에 붙인 rank-32
LoRA(파라미터 45.9M)가 이 프레임을 읽도록 학습합니다. 학습 쌍은 Anima로 생성합니다. 같은 캐릭터(정확한 Danbooru
외모·의상 태그로 고정)를 서로 다른 장면과 작가 그림체로 그리고, 학습 프롬프트에서는 레퍼런스가 제공해야 할
정보(캐릭터, 외모, 의상, 작가)를 뺍니다.

v2(`copy_asym_r64`)는 이 결합 프레임을 비대칭 attention으로 바꿨습니다. 레퍼런스 토큰은 자기 자신만 보고 텍스트
cross-attention을 건너뛰며, 타깃 토큰은 자기 자신과 레퍼런스를 함께 봅니다. 그래서 추론할 때 레퍼런스의 블록별
key/value를 이미지당 한 번만 계산해 캐시할 수 있습니다(`train/anima_env.py`의 `asym_forward`).

## 현황

| 어댑터 | 하는 일 | 상태 |
|---|---|---|
| `pilot_r32` | 레퍼런스 한 장 → 캐릭터(+ 일부 그림체) | 공개됨. 이미지 3,600장, 2,000 steps |
| `dual_r32` | 캐릭터 슬롯 + 그림체 슬롯 | 학습함(이미지 15,600장, 4,000 steps). 캐릭터 슬롯은 파일럿보다 낫지만 **그림체 슬롯은 동작하지 않음** |
| `char_v2` / `char_v3` | 레퍼런스 한 장 + `style_transfer` 세기(0 = 캐릭터만, 1 = 캐릭터와 그림체) | 중단: 세기 값이 무시됨 |
| `copy_asym_r64` | 캐릭터 + 그림체, 비대칭 attention(레퍼런스 캐시), rank 64 | **v2로 공개됨**. 이미지 15,600장, 계획한 3,000 steps 중 1,000에서 중단 |
| `style_asym_r64` | 다른 캐릭터의 이미지에서 그림체만 | 1,000 steps까지 학습. **레퍼런스의 그림체를 따라가지 않음** |
| 캐릭터 전용 어댑터 | 캐릭터는 레퍼런스에서, 그림체는 작가 태그에서; 0.56–1.27 MP | 시작하지 않음 |

**자금 부족으로 개발을 중단했습니다(2026-10-29).** 추가 학습을 후원하려면 [Ko-fi](https://ko-fi.com/electrongirl)를 이용해 주세요.

설정, 홀드아웃 지표, 그림체 슬롯이 실패한 과정과 이유는 [docs/experiments.md](docs/experiments.md)에 있습니다.

## 구성

```
datagen/         manifest sampling (characters x artist combos x scenes), ComfyUI batch workflow, GPU worker
comfyui_nodes/   anima_ref_datagen_nodes (batched heterogeneous generation + async saving)
                 anima_ref_adapter_nodes (inference: loader, reference latent, guider; dev version)
train/           anima_env.py   model plumbing: two-frame forward, offset RoPE, LoRA, sampler, self-test
                 prepare*.py    text contexts, target latents, letterboxed reference latents -> safetensors
                 train*.py      pilot (single ref), dual (two slots), char_v2 (single ref + style knob),
                                train_ref.py (multi-resolution; two-frame or asymmetric; copy / identity / style)
                 sample.py, eval_*.py, compare_adapters.py, try_refs.py   evaluation and probing
configs/         datagen and training configs used for the runs above
deploy/          setup and run scripts for a rented GPU box, dataset packing (Parquet), figures
dashboard/       local web page: training curves, eval loss gaps, eval images (synced over SSH)
data/            tag catalogs (not committed; see data/README.md)
```

## 필요한 것

- 배치 4로 학습하려면 VRAM 약 70GB인 NVIDIA GPU가 필요합니다(RTX PRO 6000 96GB를 썼습니다. `checkpoint_blocks`로
  속도를 희생해 메모리를 줄일 수 있습니다). 데이터 생성은 ComfyUI에서 Anima를 돌릴 수 있는 GPU라면 됩니다.
- ComfyUI 설치본: `train/anima_env.py`가 ComfyUI의 Anima 모듈을 그대로 가져다 쓰므로, 학습한 가중치를 바로 ComfyUI에서
  쓸 수 있습니다. `anima_env.self_test`가 학습용 forward와 ComfyUI의 forward가 같은지 확인합니다.
- `ComfyUI/models`의 모델: `anima_baseV10.safetensors`, `qwen_3_06b_base.safetensors`,
  `qwenImageVAE_qwenImageVAE.safetensors`(이름은 `configs/*.json`에 있습니다).
- ComfyUI의 requirements 위에 `pip install -r requirements.txt`.

`deploy/bootstrap.sh`는 새 PyTorch/CUDA 인스턴스에서 이 모든 것(ComfyUI, 커스텀 노드, SageAttention, 모델 다운로드)을
설치합니다.

## 실행

```bash
# 0. tag catalogs (data/README.md)
python datagen/fetch_character_catalog.py
python datagen/fetch_character_catalog.py --category artist --min-post-count 100 --output data/danbooru_artists_ge100.csv
python datagen/build_character_tags.py

# 1. data: sample a manifest, render it with ComfyUI (one worker per GPU)
python datagen/build_manifest.py --config configs/datagen_grid1.json
deploy/launch.sh grid1 production          # deploy/status.sh grid1 for progress

# 2. two-reference adapter (prepares tensors once, then trains; waits for generation to finish)
deploy/train_dual.sh dual_r32 pilot grid1

# 3. character adapter v2 with the style_transfer knob, on the same prepared tensors
python train/train_char_v2.py --prep runs/train/dual_r32_prep --config configs/train_char_v2.json --out runs/train/char_v2
python train/eval_char_v2.py --prep runs/train/dual_r32_prep --lora runs/train/char_v2/lora_step03000.safetensors --out runs/eval/char_v2

# 4. v2: asymmetric attention, character + style (and the style-only variant with --task style)
python train/train_ref.py --task copy --prep runs/train/dual_r32_prep --config configs/train_copy_asym_r64.json --out runs/train/copy_asym_r64
python train/train_ref.py ... --resume runs/train/copy_asym_r64/state_step01000.pt   # continue a run
python train/try_asym.py --lora runs/train/copy_asym_r64/lora_step01000.safetensors --prep runs/train/dual_r32_prep \
    --ref my_ref.png --size 832x1216 --prompt "1girl, solo, ..." --out runs/try/mine

# probing
python train/compare_adapters.py ...      # held-out loss per reference setup, two adapters side by side
python train/try_refs.py --lora ... --prep ... --style my_style.png --prompt "1girl, solo, ..."
python dashboard/server.py --host my-instance --run runs/train/char_v2
```

모든 스크립트의 docstring에 정확한 사용법이 있습니다. 데이터셋은 `deploy/pack_dataset.py pack`으로 Parquet 샤드(행마다
WebP 이미지 + 원본 latent + 메타데이터)로 묶을 수 있고, `pack_dataset.py unpack`으로 prepare 스크립트가 읽는 구조로
되돌릴 수 있습니다.

## 평가

이미지만으로 판단했다가 틀린 적이 여러 번 있어서, 모든 어댑터를 **홀드아웃 loss 차이**로도 측정합니다. 학습에서
뺀 캐릭터에 대해 고정된 sigma(0.15, 0.35, 0.6, 0.85)와 고정된 노이즈로 디노이징 loss를 구하고, 맞는 레퍼런스와
맞지 않는 레퍼런스를 비교합니다. 이 차이가 레퍼런스가 실제로 무엇을 전달하는지 알려 줍니다. 예를 들어 캐릭터 차이는
"다른 캐릭터의 이미지(같은 그림체)를 줬을 때의 loss − 맞는 캐릭터(다른 그림체)를 줬을 때의 loss"입니다.
`train/compare_adapters.py`와 `train/eval_char_v2.py`가 이 값을 계산하며, 정의는 각 docstring에 있습니다.

## 라이선스

코드는 Apache-2.0입니다(`LICENSE` 참고). Anima의 가중치는 CircleStone Labs Non-Commercial License를 따르며, Anima 위에서
학습한 어댑터와 Anima로 생성한 이미지에도 이 라이선스가 적용됩니다.
