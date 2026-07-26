# Generative Model — LoRA Style Fine-Tuning with Stable Diffusion 1.5
*"Ghibli Market"*

## Team

- Prabin Sharma Poudel
- Azariah Asafo Agyei
- Adam Lo Jen Khai

Technical University of Nuremberg (UTN)

## Dataset

- `style_imgs/original/` : full-resolution reference frames sampled from several Studio Ghibli films.
- `style_imgs/512/` : 512x512 center-cropped versions of those frames, ready to train (845 images).
- `style_imgs/512_800_market10/` : a curated 800-image variant, with the 5 known market/food-stall frames oversampled to make up ~10% of the dataset (used for the run documented in this README).

## Task Overview

### Tokenizer Setup

Add a new style token `<sks>` to the tokenizer and resize the text encoder's embedding table to match.

- Implemented inline in `code/train_lora.py` (see `tokenizer.add_tokens(...)` and `text_encoder.resize_token_embeddings(...)`), not a separate script.
- By default, only the surrounding LoRA attention weights learn to *use* `<sks>`; the token's own embedding row stays at its random initialization. Pass `--train_token_embedding` to also make that row trainable (via a gradient mask so only that one row updates).

### LoRA Fine-Tuning

Fine-tune both the UNet and the text encoder with a dual-adapter LoRA setup, so that the prompt

```
a busy market, in <sks> style
```

renders in the learned Ghibli style. Adapters attach to the attention projections of both models (`to_q/to_k/to_v/to_out.0` in the UNet, `q_proj/k_proj/v_proj/out_proj` in the CLIP text encoder) via `add_adapter()`, and only those adapter weights are trained — the base SD 1.5 weights stay frozen throughout.

### What is Tokenization Here?

Stable Diffusion uses a text tokenizer + encoder (CLIP) to understand prompts. By default, it has no concept of a custom style like "Ghibli" — there's no token for it.

We add a new token (`<sks>`) and give it its own embedding row, so that during LoRA training the model can learn to associate `<sks>` with the training dataset's visual style.

After this step, prompts like

```
a busy market, in <sks> style
```

are understood as "market scene + learned style token," and the attached LoRA adapters do the work of rendering that combination in the target aesthetic.

## Repository Structure

```
Ghibli_final/
|-- README.md
|-- report.pdf
|-- requirements.txt
|-- code/
|   |-- train_lora.py   # tokenizer + dual-adapter
|   |                   #   LoRA training loop
|   \-- eval_lora.py    # load checkpoint, render
|-- style_imgs/
|   |-- original/          # full-res source frames
|   |-- 512/               # 512 crops, full 845 set
|   \-- 512_800_market10/  # 800 imgs, 10% market
|-- lora_out/
|   \-- pytorch_lora_weights.safetensors
\-- samples/            # rendered eval images
```

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train

```bash
python code/train_lora.py \
    --data_dir style_imgs/512_800_market10 \
    --instance_token "<sks>" \
    --output_dir lora_out \
    --rank 8 \
    --learning_rate 1e-4 \
    --max_steps 800 \
    --overwrite
```

This writes exactly one file: `lora_out/pytorch_lora_weights.safetensors`. Omit `--overwrite` to refuse re-running if that file already exists.

Two flags apply automatically even though they aren't passed above — both default to values chosen from experimentation, not off:
- `--lora_alpha` defaults to `16` regardless of `--rank` (so with `--rank 8` above, the effective LoRA scale is `16/8 = 2x`). Pass `--lora_alpha 8` to get the unscaled `1x` behavior instead.
- `--cfg_dropout_prob` defaults to `0.1` (10% of training steps use an empty prompt instead of the caption), which improves classifier-free guidance quality at inference. Pass `--cfg_dropout_prob 0.0` to disable it.

Add `--train_token_embedding` to also learn the `<sks>` token's own embedding row (off by default).

### 3. Evaluate

```bash
python code/eval_lora.py \
    --weights lora_out/pytorch_lora_weights.safetensors \
    --prompt "a busy market, in <sks> style" \
    --guidance_scales 7.5 \
    --outdir samples
```

`--negative_prompt` and `--seeds` also default to fixed values (a standard artifact-suppressing negative prompt, and a list of 10 fixed seeds) so every run without those flags is reproducible; pass either explicitly to override. This renders one image per seed (10 by default) at each comma-separated CFG value in `--guidance_scales`, named `sample_cfg<value>_seed<value>.png`.

## Expected GPU / runtime

- GPU: single NVIDIA GPU with >= 16 GB VRAM (e.g. A40/A100/V100/RTX 3090), fp16 mixed precision.
- Training: 800 steps, batch size 1 with gradient accumulation 4, ~11 minutes measured on an NVIDIA A40.
- Evaluation: ~1-2 minutes per image at 30 inference steps (so ~10-15 minutes for the default 10-seed sweep at one CFG value).
- CPU-only is possible for smoke testing but training 800 steps is impractically slow.
