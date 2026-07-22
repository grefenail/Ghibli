"""Train a dual-adapter (UNet + text encoder) LoRA style adapter on Stable Diffusion 1.5."""
import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a LoRA style adapter for SD 1.5.")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory of training images.")
    parser.add_argument("--instance_token", type=str, default="<sks>")
    parser.add_argument("--prompt", type=str, default="a busy market, in <sks> style")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the LoRA weights into.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "bf16", "no"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing lora weights file in output_dir. Without this flag, the script "
        "refuses to run if the output file already exists.",
    )
    return parser.parse_args()


class StyleImageDataset(Dataset):
    """Pairs every training image with the same fixed style prompt (textual-inversion style)."""

    def __init__(self, data_dir, tokenizer, prompt, resolution=512):
        self.paths = sorted(p for p in Path(data_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self.paths:
            raise ValueError(f"No images found in {data_dir}")
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        pixel_values = self.transform(image)
        tokenized = self.tokenizer(
            self.prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
        }


def collate_fn(examples):
    return {
        "pixel_values": torch.stack([ex["pixel_values"] for ex in examples]).float(),
        "input_ids": torch.stack([ex["input_ids"] for ex in examples]),
        "attention_mask": torch.stack([ex["attention_mask"] for ex in examples]),
    }


def main():
    args = parse_args()

    weights_path = Path(args.output_dir) / "pytorch_lora_weights.safetensors"
    if weights_path.exists() and not args.overwrite:
        raise FileExistsError(f"{weights_path} already exists. Pass --overwrite to replace it.")

    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load the frozen SD 1.5 backbone components individually (rather than the full pipeline)
    # so LoRA adapters can be attached to just the UNet and text encoder below.
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_name, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_name, subfolder="scheduler")

    # Register the new style token (e.g. "<sks>") as its own embedding so the model can
    # learn a dedicated representation for it instead of reusing an existing token's meaning.
    num_added = tokenizer.add_tokens(args.instance_token)
    if num_added == 0:
        print(f"{args.instance_token} was already in the tokenizer.")
    else:
        print(f"Added {args.instance_token} to the tokenizer.")
        text_encoder.resize_token_embeddings(len(tokenizer))

    # Freeze the base weights; only the LoRA adapter weights added below will be trained.
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Attach low-rank adapters to the attention projections of both the UNet (image
    # denoising) and the text encoder (prompt conditioning) — a "dual-adapter" LoRA setup.
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    text_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    unet.add_adapter(unet_lora_config)
    text_encoder.add_adapter(text_lora_config)

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" and device == "cuda" else torch.float32
    vae.to(device, dtype=weight_dtype)
    unet.to(device)
    text_encoder.to(device)

    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    trainable_params += list(filter(lambda p: p.requires_grad, text_encoder.parameters()))
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    dataset = StyleImageDataset(args.data_dir, tokenizer, args.prompt, args.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_epochs = math.ceil(args.max_steps / updates_per_epoch)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16" and device == "cuda"))
    global_step = 0
    progress = tqdm(total=args.max_steps, desc="Training")

    unet.train()
    text_encoder.train()

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Encode to the frozen VAE's latent space; no grad needed since the VAE isn't trained.
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Standard DDPM training objective: add noise at a random timestep, then have
            # the UNet predict that noise from the noisy latents.
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(args.mixed_precision == "fp16" and device == "cuda"),
            ):
                encoder_hidden_states = text_encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                # Rescale so accumulated gradients across steps average to the true batch loss.
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            # Only step the optimizer once enough micro-batches have accumulated gradients,
            # emulating a larger effective batch size than train_batch_size alone allows.
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.item() * args.gradient_accumulation_steps:.4f}")

                if global_step >= args.max_steps:
                    break

        if global_step >= args.max_steps:
            break

    progress.close()
    print(f"Finished {global_step} optimizer steps.")

    # Extract only the (small) LoRA adapter weights, not the full frozen base model.
    unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    text_encoder_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder))

    StableDiffusionPipeline.save_lora_weights(
        save_directory=args.output_dir,
        unet_lora_layers=unet_lora_layers,
        text_encoder_lora_layers=text_encoder_lora_layers,
        safe_serialization=True,
    )

    print(weights_path, weights_path.exists(), f"{weights_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
