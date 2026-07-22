#!/bin/bash -l
#SBATCH --job-name=sks_lora_train
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log

set -euo pipefail

ASSIGN_DIR=/home/hpc/v123be/v123be43/thesis/gibli/assignment4
PYTHON=/home/hpc/v123be/v123be43/.conda/envs/sks_lora/bin/python

export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/vault/v123be/v123be43/hf_cache
export HF_HUB_OFFLINE=1

cd "$ASSIGN_DIR"

"$PYTHON" code/train_lora.py \
    --data_dir style_imgs/512 \
    --instance_token "<sks>" \
    --output_dir lora_out \
    --rank 8 \
    --max_steps 800 \
    --overwrite

"$PYTHON" code/eval_lora.py \
    --weights lora_out/pytorch_lora_weights.safetensors \
    --prompt "a busy market, in <sks> style" \
    --outdir samples
