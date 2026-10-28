# Anima Reference Adapter — research code

Training and evaluation code for reference adapters on [Anima](https://huggingface.co/circlestone-labs/Anima)
(Cosmos-Predict2 MiniTrainDIT, 2B): give the model one reference image and it draws that character, and
optionally that drawing style, in a new scene described by the prompt.

This repository holds everything behind the released adapter and the experiments since: synthetic data
generation with Anima itself, tensor preparation, training, evaluation and analysis scripts, plus the
failed and in-progress experiments with their numbers ([docs/experiments.md](docs/experiments.md)).

- Inference nodes for ComfyUI (release): [ElectronGirl/ComfyUI-Anima-Ref-Adapter](https://github.com/ElectronGirl/ComfyUI-Anima-Ref-Adapter)
- Released weights and technical report: [ElectronGirl/anima-ref-adapter](https://huggingface.co/ElectronGirl/anima-ref-adapter)

## Method in one paragraph

The reference image is letterboxed to the output size, VAE-encoded and appended to the target latent as
an extra frame on the time axis (as ComfyUI-Cosmos-Reference does). Two changes make it a source of
appearance instead of a layout to continue: the reference frame is fed clean (timestep 0), and its RoPE
h coordinates are shifted one target height past the target grid, so there is no spatial
correspondence to copy. A rank-32 LoRA on every block's attention and MLP projections (45.9M
parameters) learns to read the frame. Training pairs are generated with Anima: the same character
(fixed by exact Danbooru appearance/outfit tags) in different scenes and artist styles; the training
prompt omits what the reference should supply (character, appearance, outfit, artist).

## Status

| Adapter | What it does | State |
|---|---|---|
| `pilot_r32` | one reference → identity (+ some style) | released; 3,600 images, 2,000 steps |
| `dual_r32` | character slot + style slot | trained (15,600 images, 4,000 steps); character slot better than pilot, **style slot does not work** |
| `char_v2` / `char_v3` | one reference + `style_transfer` strength (0 = identity only, 1 = identity and style) | stopped: the strength was ignored |
| character-only adapter | identity from the reference, style from artist tags; 0.56–1.27 MP | **next** |
| character + style adapter | identity and style from the reference | `pilot_r32`; may be retrained |
| style-only adapter | style only, from another character's image | planned |

See [docs/experiments.md](docs/experiments.md) for the setups, held-out metrics, what went wrong with
the style slot and why.

## Layout

```
datagen/         manifest sampling (characters x artist combos x scenes), ComfyUI batch workflow, GPU worker
comfyui_nodes/   anima_ref_datagen_nodes (batched heterogeneous generation + async saving)
                 anima_ref_adapter_nodes (inference: loader, reference latent, guider; dev version)
train/           anima_env.py   model plumbing: two-frame forward, offset RoPE, LoRA, sampler, self-test
                 prepare*.py    text contexts, target latents, letterboxed reference latents -> safetensors
                 train*.py      pilot (single ref), dual (two slots), char_v2 (single ref + style knob)
                 sample.py, eval_*.py, compare_adapters.py, try_refs.py   evaluation and probing
configs/         datagen and training configs used for the runs above
deploy/          setup and run scripts for a rented GPU box, dataset packing (Parquet), figures
dashboard/       local web page: training curves, eval loss gaps, eval images (synced over SSH)
data/            tag catalogs (not committed; see data/README.md)
```

## Requirements

- An NVIDIA GPU with ~70 GB for training at batch 4 (an RTX PRO 6000 96 GB was used; `checkpoint_blocks`
  trades speed for memory). Generation runs on any GPU that runs Anima in ComfyUI.
- A ComfyUI checkout: `train/anima_env.py` imports ComfyUI's own Anima modules, so trained weights plug
  straight back into ComfyUI. `anima_env.self_test` checks the training forward against ComfyUI's.
- Models in `ComfyUI/models`: `anima_baseV10.safetensors`, `qwen_3_06b_base.safetensors`,
  `qwenImageVAE_qwenImageVAE.safetensors` (names in `configs/*.json`).
- `pip install -r requirements.txt` on top of ComfyUI's requirements.

`deploy/bootstrap.sh` sets all of this up on a fresh PyTorch/CUDA instance (ComfyUI, custom nodes,
SageAttention, model downloads).

## Running it

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

# probing
python train/compare_adapters.py ...      # held-out loss per reference setup, two adapters side by side
python train/try_refs.py --lora ... --prep ... --style my_style.png --prompt "1girl, solo, ..."
python dashboard/server.py --host my-instance --run runs/train/char_v2
```

Every script has a docstring with its exact usage. Datasets can be packed into Parquet shards (WebP
image + raw latent + metadata per row) with `deploy/pack_dataset.py pack`, and restored to the layout the
prepare scripts read with `pack_dataset.py unpack`.

## Evaluation

Images alone were misleading more than once, so every adapter is also measured by **held-out loss gaps**:
denoising loss on held-out characters at fixed sigmas (0.15, 0.35, 0.6, 0.85) and fixed noise, with a
matching versus a mismatched reference. A gap says what the reference actually carries — e.g. identity
gap = loss with another character's image (same style) minus loss with the right character (other style).
`train/compare_adapters.py` and `train/eval_char_v2.py` compute them; definitions are in their docstrings.

## License

Code: Apache-2.0 (see `LICENSE`). Anima's weights are under the CircleStone Labs Non-Commercial License;
adapters trained on top of Anima, and images generated with it, are subject to that license as well.
