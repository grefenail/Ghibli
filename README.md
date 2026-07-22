# Ghibli Market: LoRA Style-Tuning with Stable Diffusion 1.5

## Team

- Hesham Abdalla (hesham.abdalla@utn.de)
- Jan Kobiolka (jan.kobiolka@utn.de)

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
    --data_dir style_imgs/512 \
    --instance_token "<sks>" \
    --output_dir lora_out \
    --rank 8 \
    --learning_rate 1e-4 \
    --max_steps 800 \
    --overwrite
```

This writes exactly one file: `lora_out/pytorch_lora_weights.safetensors`.
Omit `--overwrite` to refuse re-running if that file already exists.

Evaluate the trained adapter (renders 3 images, no style-strength slider):

```bash
python code/eval_lora.py \
    --weights lora_out/pytorch_lora_weights.safetensors \
    --prompt "a busy market, in <sks> style" \
    --outdir samples
```

## Expected GPU / runtime

- GPU: single NVIDIA GPU with >= 16 GB VRAM (e.g. A100/V100/RTX 3090), fp16 mixed precision.
- Training: ~800 steps, batch size 1 with gradient accumulation 4, roughly 20-30 minutes on an A100.
- Evaluation: ~1-2 minutes for 3 images at 30 inference steps.
- CPU-only is possible for smoke testing but training 800 steps is impractically slow.
