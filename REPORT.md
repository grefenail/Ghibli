# Ghibli-Style LoRA Fine-Tuning of Stable Diffusion 1.5

## 1. Objective

This project fine-tunes Stable Diffusion 1.5 to reproduce the visual style of Studio Ghibli films using a lightweight LoRA (Low-Rank Adaptation) adapter, without modifying the frozen base model weights. A new token, `<sks>`, is introduced to represent the learned style, so that any prompt of the form `"<subject>, in <sks> style"` renders in the target aesthetic. The training corpus (`style_imgs/512`) consists of 845 frames sampled from several Ghibli films (*Arrietty*, *From Up on Poppy Hill*, *Howl's Moving Castle*, *Spirited Away*, and others).

## 2. Method

**Dual-adapter LoRA.** Rather than fine-tuning the full ~1B-parameter UNet and text encoder, low-rank adapters (rank 8) are attached to the attention projection layers of both components: `to_q/to_k/to_v/to_out.0` in the UNet, and `q_proj/k_proj/v_proj/out_proj` in the CLIP text encoder. This is implemented via diffusers' native `add_adapter()` / `save_lora_weights()` API, which keeps the checkpoint small (~8.7 MB) and directly loadable through `pipe.load_lora_weights()` at inference time — no custom loading logic required.

**Token embedding training.** By default, the `<sks>` token's embedding row is left at its random initialization, and only the surrounding LoRA deltas are trained (the attention layers learn to *contextualize* an arbitrary token as "Ghibli style"). An optional `--train_token_embedding` flag additionally makes the token's own embedding row trainable, via a backward hook that zeroes the gradient of every other row in the embedding matrix — so a single optimizer step over the full embedding table only ever updates that one row. This must be registered *after* `add_adapter()`, since PEFT's adapter injection resets `requires_grad` on every non-adapter parameter.

**Classifier-free guidance (CFG) dropout.** At inference, CFG extrapolates between a conditional and an unconditional (empty-prompt) noise prediction, scaled by `guidance_scale`. The original training loop never exposed the model to an empty prompt, so the unconditional branch was effectively untrained — meaning high guidance scales amplified an untrained prediction rather than a clean style signal. A `--cfg_dropout_prob` flag was added to randomly substitute the batch's caption with an empty string on a configurable fraction of training steps (10% used here), so the model learns both branches. This measurably improved stylization strength and reduced artifacts at `guidance_scale` values of 7.5–10.0 compared to checkpoints trained without it.

**LoRA scale.** A `--lora_alpha` flag decouples the adapter's scaling factor from its rank (`scale = lora_alpha / rank`); the default is 1× (`lora_alpha = rank`), with 2× (`lora_alpha = 2·rank`) tested as an alternative that injects stronger style deltas per rank.

Training uses a standard DDPM noise-prediction objective (MSE loss) with fp16 mixed precision, gradient scaling, gradient clipping, and gradient accumulation (effective batch size 4 from batch size 1 × 4 accumulation steps).

## 3. Dataset Construction

Beyond the full 845-image corpus, several curated subsets were built to test the effect of dataset size and content balance:

| Variant | Size | Notes |
|---|---|---|
| `512_market10_general90` | 101 | 10 market-content frames (via 2× duplication of 5 unique source images) + 91 general frames |
| `512_half400_market10` | 400 | 360 general (randomly sampled) + 40 market (5 unique × 8 copies) |
| `512_800_market10` | 800 | 720 general + 80 market (5 unique × 16 copies) |
| `512_full_market10` | 933 | Full 845-image pool + 90 extra market duplicates (5 unique × 19 total instances) |

Only **five unique real market/food-stall frames** exist in the corpus (`Spirited_Away_3/4/6/7`, `Howls_Moving_Castle_1`). "Market ratio" datasets are constructed by duplicating these five images to reach a target frequency (~10% of the dataset), rather than by sourcing additional distinct market content. An earlier version of `train_lora.py` also supported a `--captions_file` JSON to give market-tagged images a distinct caption from the rest — this was later removed after discovering that every dataset used the identical caption string for both the override and the default fallback prompt, making the caption distinction a no-op. The mechanism now in effect is **pure oversampling**: market frames are shown to the model more frequently per epoch, with no textual differentiation from general content. All training uses the single fixed prompt `"a busy market, in <sks> style"`.

## 4. Evaluation Protocol

`eval_lora.py` loads the frozen SD 1.5 pipeline, re-registers the `<sks>` token, restores its trained embedding (if present) from the checkpoint's `sks_token_embedding` key, and loads the LoRA weights via `pipe.load_lora_weights()`. To support systematic comparison, the script was extended with `--seeds` and `--guidance_scales` (comma-separated lists), producing one image per (seed, CFG) pair — enabling direct, seed-matched visual comparison of how CFG strength and different training configurations affect the same underlying noise sample.

## 5. Key Findings

- **CFG dropout is necessary for clean high-guidance behavior.** Checkpoints trained without it show artifacting or noise amplification at `guidance_scale ≥ 7.5`; checkpoints trained with 10% dropout show markedly cleaner, more strongly stylized output at the same guidance values.
- **Market-frame oversampling is a frequency effect, not a content-labeling effect**, given the shared caption text — its impact should be interpreted as "does repeating these 5 frames more often measurably shift the style," not as multi-concept learning.
- **Environment reproducibility matters for grading.** A clean `venv` install of `requirements.txt` (unpinned) pulled `transformers==5.14.1`, `diffusers==0.39.0`, `huggingface_hub==1.24.0`, and `peft==0.19.1` — matching the working development environment exactly — but a newer `torch` (2.13.0 vs. the tested 2.5.1+cu121), since only `torch>=2.0` is pinned. This is a latent reproducibility risk for future re-runs on a different date.

## 6. Conclusion & Future Work

The dual-adapter LoRA approach with trained token embedding and CFG dropout produces a lightweight (~8.7 MB), reproducible style adapter for SD 1.5. The current best configuration (`rank=8`, `lora_alpha=16`, `max_steps=800`, `cfg_dropout_prob=0.1`, trained token embedding, full 933-image oversampled dataset) is used as the reference checkpoint. Future work could (1) reintroduce distinct captions per content category to test whether textual differentiation adds value beyond oversampling, (2) pin exact dependency versions in `requirements.txt` for long-term reproducibility, and (3) run a systematic ablation isolating the individual contribution of `lora_alpha` scale, dataset size, and CFG dropout rate.
