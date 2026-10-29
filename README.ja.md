# Anima Reference Adapter — 研究コード

[English](README.md) · [한국어](README.ko.md) · **日本語**

[Anima](https://huggingface.co/circlestone-labs/Anima)（Cosmos-Predict2 MiniTrainDIT、2B）向け参照アダプターの学習・評価
コードです。モデルに参照画像を1枚渡すと、プロンプトが描写する新しいシーンにそのキャラクターを描き、場合によっては
その画風でも描きます。

このリポジトリには、公開したアダプターとその後の実験に使ったものがすべて入っています。Anima 自身による合成データの
生成、テンソルの準備、学習、評価と分析のスクリプトがあり、失敗した実験や途中の実験とその数値は
[docs/experiments.md](docs/experiments.md) にまとめています（英語）。

- ComfyUI 推論ノード（配布版）: [ElectronGirl/ComfyUI-Anima-Ref-Adapter](https://github.com/ElectronGirl/ComfyUI-Anima-Ref-Adapter)
- 公開している重みと技術レポート: [ElectronGirl/anima-ref-adapter](https://huggingface.co/ElectronGirl/anima-ref-adapter)

## 手法の概要

パイロット版では、参照画像を出力サイズに合わせてレターボックス化し、VAE でエンコードしてから、時間軸の追加フレーム
としてターゲットの latent の後ろにつなげました（ComfyUI-Cosmos-Reference と同じ方式）。2つの変更により、このフレームを
続けて描くレイアウトではなく、外見の情報源にしています。参照フレームはノイズのない状態（timestep 0）で入力し、RoPE の
h 座標をターゲットの格子の外へターゲットの高さ1つ分ずらすことで、空間的に対応する位置をなくしています。すべての
ブロックの attention と MLP の射影に付けた rank-32 の LoRA（パラメーター 45.9M）が、このフレームを読むように学習します。
学習ペアは Anima で生成します。同じキャラクター（Danbooru の正確な外見・衣装タグで固定）を異なるシーンと絵師の画風で
描き、学習用プロンプトからは参照画像が担うべき情報（キャラクター、外見、衣装、絵師）を取り除きます。

v2（`copy_asym_r64`）では、この結合フレームを非対称 attention に置き換えました。参照トークンは自分自身だけを見てテキストの
cross-attention を飛ばし、ターゲットのトークンは自分自身と参照の両方を見ます。そのため推論時には、参照のブロックごとの
key/value を画像1枚につき1回だけ計算してキャッシュできます（`train/anima_env.py` の `asym_forward`）。

## 現状

| アダプター | 役割 | 状態 |
|---|---|---|
| `pilot_r32` | 参照画像1枚 → キャラクター（+ 画風の一部） | 公開済み。画像 3,600 枚、2,000 steps |
| `dual_r32` | キャラクタースロット + 画風スロット | 学習済み（画像 15,600 枚、4,000 steps）。キャラクタースロットはパイロット版より良いが、**画風スロットは機能しない** |
| `char_v2` / `char_v3` | 参照画像1枚 + `style_transfer` の強さ（0 = キャラクターのみ、1 = キャラクターと画風） | 中止: 強さの値が無視された |
| `copy_asym_r64` | キャラクター + 画風、非対称 attention（参照のキャッシュ）、rank 64 | **v2 として公開済み**。画像 15,600 枚、予定 3,000 steps のうち 1,000 で中断 |
| `style_asym_r64` | 別のキャラクターの画像から画風のみ | 1,000 steps まで学習。**参照画像の画風を再現しない** |
| キャラクター専用アダプター | キャラクターは参照画像から、画風は絵師タグから。0.56–1.27 MP | 未着手 |

**資金不足のため開発を休止しています（2026-10-29）。** 追加学習の支援は [Ko-fi](https://ko-fi.com/electrongirl) からお願いします。

設定、ホールドアウト指標、画風スロットが失敗した経緯と理由は [docs/experiments.md](docs/experiments.md) にあります。

## 構成

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

## 必要なもの

- バッチ 4 で学習するには VRAM 約 70GB の NVIDIA GPU が必要です（RTX PRO 6000 96GB を使いました。`checkpoint_blocks` で
  速度と引き換えにメモリを減らせます）。データ生成は、ComfyUI で Anima を動かせる GPU なら十分です。
- ComfyUI のチェックアウト: `train/anima_env.py` は ComfyUI の Anima モジュールをそのまま使うので、学習した重みをそのまま
  ComfyUI で使えます。`anima_env.self_test` が学習用の forward と ComfyUI の forward が一致するか確認します。
- `ComfyUI/models` のモデル: `anima_baseV10.safetensors`、`qwen_3_06b_base.safetensors`、
  `qwenImageVAE_qwenImageVAE.safetensors`（名前は `configs/*.json` にあります）。
- ComfyUI の requirements に加えて `pip install -r requirements.txt`。

`deploy/bootstrap.sh` は、新しい PyTorch/CUDA インスタンスにこれらすべて（ComfyUI、カスタムノード、SageAttention、
モデルのダウンロード）を用意します。

## 実行

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

各スクリプトの docstring に正確な使い方が書いてあります。データセットは `deploy/pack_dataset.py pack` で Parquet の
シャード（各行に WebP 画像 + 生の latent + メタデータ）にまとめられ、`pack_dataset.py unpack` で prepare スクリプトが
読む構成に戻せます。

## 評価

画像だけで判断して誤ったことが何度もあったため、すべてのアダプターを**ホールドアウトの loss 差**でも測定します。
学習から除外したキャラクターについて、固定した sigma（0.15、0.35、0.6、0.85）と固定したノイズでデノイズの loss を求め、
正しい参照画像と正しくない参照画像を比べます。この差が、参照画像が実際に何を伝えているかを示します。たとえば
キャラクターの差は「別のキャラクターの画像（同じ画風）を与えたときの loss − 正しいキャラクター（別の画風）を与えたときの
loss」です。`train/compare_adapters.py` と `train/eval_char_v2.py` がこの値を計算し、定義はそれぞれの docstring にあります。

## ライセンス

コードは Apache-2.0 です（`LICENSE` を参照）。Anima の重みは CircleStone Labs Non-Commercial License に従い、Anima の上で
学習したアダプターと、Anima で生成した画像にもこのライセンスが適用されます。
