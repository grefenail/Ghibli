# Ghibli Market: LoRA Style-Tuning with Stable Diffusion 1.5

## Team

- Prabin Sharma Poudel
- Jayb
- Adam Lo Jen Khai

Technical University of Nuremberg (UTN)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Train the dual-adapter (UNet + text encoder) LoRA:

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

This writes exactly one file: `lora_out/pytorch_lora_weights.safetensors`.
Omit `--overwrite` to refuse re-running if that file already exists.

Two flags apply automatically even though they aren't passed above — both default to values chosen from experimentation, not off:
- `--lora_alpha` defaults to `16` regardless of `--rank` (so with `--rank 8` above, the effective LoRA scale is `16/8 = 2x`). Pass `--lora_alpha 8` to get the unscaled `1x` behavior instead.
- `--cfg_dropout_prob` defaults to `0.1` (10% of training steps use an empty prompt instead of the caption), which improves classifier-free guidance quality at inference. Pass `--cfg_dropout_prob 0.0` to disable it.

Add `--train_token_embedding` to also learn the `<sks>` token's own embedding row (off by default).

Evaluate the trained adapter:

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
