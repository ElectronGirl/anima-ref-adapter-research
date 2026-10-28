# Experiments

Chronological log of the adapter runs, with the numbers that decided the next step. All training ran on
one RTX PRO 6000 Max-Q (96 GB), batch 4, AdamW, lr 1e-4 with warmup and cosine decay to 0.1×, flow
matching with shift 3 (σ sampled as `3u / (1 + 2u)`, u uniform), LoRA rank 32 / alpha 32 on all 28 blocks'
attention and MLP projections.

## Data

| Run | Groups × images | Characters | Artist combos | Notes |
|---|---|---|---|---|
| `pilot` | 600 × 6 = 3,600 | 300 (2 styles each) | 292 | same character + same style within a group |
| `grid1` | 6,000 × 2 = 12,000 | 2,000 | 1,726 | sparse character × style grid: each character in ~3 styles, each combo on ~3 characters |

Combined (`dual_r32_prep`): 6,600 groups, 2,300 characters, 1,952 artist combos; per combo a median of
3 characters, per character a median of 3 combos. Images are 832×1216, 896×1152, 1024×1024 or
1216×832, rendered by Anima base v1.0 (bf16, er_sde, 24 steps, cfg 4.5) with exact Danbooru tags. Held out for
evaluation: 12 characters per source chosen by hash, plus every group that shares a held-out
character or a held-out artist combo (209 groups); no training group touches either.

Preparing tensors (`prepare_dual.py`): 62,346 distinct prompts (text contexts, 7.9 GB), 15,600 target
latents, 55,950 letterboxed reference latents (each image encoded at every target size it can be paired
with). About 21 min of text encoding and 45 min of VAE encoding at ~21 images/s.

## 1. `pilot_r32` — single reference

Pairs: two images of the same group (same character, same artist, different scene). Prompt without
character, appearance, outfit and artist tags (70%), with some appearance/outfit tags (20%), or minimal
(10%). 2,000 steps.

Result: identity transfers to held-out characters and to original characters in unseen styles; pose and
background follow the prompt. Because every pair shares the artist, the adapter also learned to copy the
reference's drawing style — useful, but not controllable. Report: `docs/report.md` in the release repo.

## 2. `dual_r32` — character slot + style slot (style slot failed)

Two reference frames: character slot (RoPE t = 1) and style slot (t = 2), both shifted one target height.
For a target of group (character C, style S): character ref = C drawn in another style; style ref = S on
another character. Modes per batch: split (both slots) 45%, character only 20%, style only 20%, same group
in both slots 15%. With one slot empty the prompt gained the missing information 60% of the time
(artist tags for character-only, character tags for style-only). Initialised from `pilot_r32`; 4,000 steps,
7.15 h at 5.8 s/step, 68.8 GB peak.

Per-mode training loss stayed flat around 0.115 for all four modes the whole run — two references were
no better than one.

**Held-out loss** (66 groups whose characters neither adapter trained on; mean over 4 sigmas):

| Reference setup | pilot | dual |
|---|---|---|
| no reference (base model) | 0.0880 | 0.0880 |
| character slot: same character, other style | 0.0861 | **0.0851** |
| character slot: other character, same style | 0.0872 | 0.0866 |
| identity gap (row 3 − row 2) | 0.0011 | **0.0015** |
| both slots, style slot right / wrong | — | 0.08499 / 0.08548 |
| character slot only | — | 0.08508 |
| style slot only (+ character tags), right / wrong style | — | 0.08345 / 0.08387 |

So the dual adapter is a slightly better *character* adapter than the pilot (lower loss at every sigma,
40% larger identity gap; images about even), but the style slot adds almost nothing: filling it with the
right style lowers the loss by 0.0001 over the character slot alone. In images, both slots ≈ character
slot only, the output takes the character reference's style, and style-only transfer fails for unseen
artists.

### Why it failed

1. **Contradictory supervision for the character slot.** In some batches the right answer was to copy the
   character reference's style (`same` mode, and character-only batches without artist tags, where nothing
   else tells the model the target style); in others it was wrong. With nothing telling the model which
   case it is in, it learned an average. Initialising from the pilot, which had learned "copy everything",
   pulled the average towards copying.
2. **The loss barely sees style.** Reference effects on the v-prediction MSE are concentrated at high noise
   (σ = 0.85, where colour and large shapes are decided); at σ = 0.15 every setup — with or without a
   reference — scores the same within 0.001. Line work and texture, much of what "style" means, hardly
   move the loss, so the gradient towards using the style slot is tiny. (Weighting training towards low σ
   therefore would not help.)
3. **Many artist combos look alike in Anima**, so copying the character reference's style is often close
   to right anyway.

## 3. `char_v2` — one reference + `style_transfer` strength (in progress)

Fixes cause 1 by making the task unambiguous. A learned vector times a strength s ∈ [0, 1] is added to
every reference token after patch embedding (`ref_bias` in `anima_env.ref_forward`). Two modes, 50/50:

| s | Reference | Prompt | Target |
|---|---|---|---|
| 0 (identity only) | same character, another artist; 50% re-encoded with detail scrambled (2–4× down/up-sampling, blur, tone jitter) | scene + target's artist tags | the character in the prompt's style |
| 1 (identity + style) | another image of the target's group | scene, no artist | same character, same artist |

Trained from scratch (no LoRA init, to avoid inheriting the copying habit), 3,000 steps, ~4.2 s/step.
Evaluated at steps 1,000 / 2,000 / 3,000 with `eval_char_v2.py` gaps:

- `identity_s0` — artist prompt, s = 0: other character − right character (should stay high)
- `leak_s0` — artist prompt, s = 0: right character in another style − same group (style leaking from the
  reference; should go to ~0)
- `copy_s0`, `copy_s1` — scene prompt: right character in another style − same group, at s = 0 and 1
- `knob` = `copy_s1 − copy_s0` (should grow)

Reference point (`dual_r32` with s ignored): identity 0.00197, leak 0.00026, copy 0.00060, knob 0.

Results will be added here.

## 4. Style adapter (planned)

Style ref = same artist, different character; prompt = everything except the artist; trained from scratch.
Cause 2 remains: candidate fixes are selecting pairs whose styles differ clearly (style-embedding
distance), a style loss on the predicted x0 (style-encoder features), or conditioning on a style-encoder
embedding as extra tokens (IP-Adapter style) instead of a reference frame.

## Engineering notes

- `ThreadPoolExecutor.map` submits every job up front; with image loading faster than VAE encoding it
  held all ~56k letterboxed images (~3 MB each) in RAM and was OOM-killed at the 128 GB container limit.
  `prepare_dual.py` keeps a bounded window of 64 in flight and reuses finished text contexts on restart.
- `ref_forward` reproduces ComfyUI's forward exactly (max abs diff 0.0); `test_dual.py` checks that the
  ComfyUI node and the training forward agree for every slot combination.
- Process checks with `pgrep -f <pattern>` from a shell whose own command line contains the pattern match
  themselves; use `[p]attern` or a pid file.
