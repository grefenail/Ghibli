"""Load base SD 1.5 plus the trained LoRA adapter and render sample images.

No style-strength slider is exposed, per the assignment brief.
"""
import argparse
from pathlib import Path

import torch

from diffusers import StableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA style adapter for SD 1.5.")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--weights", type=str, required=True, help="Path to the pytorch_lora_weights.safetensors file.")
    parser.add_argument("--instance_token", type=str, default="<sks>")
    parser.add_argument("--prompt", type=str, default="a busy market, in <sks> style")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_images < 3:
        raise ValueError("The assignment requires at least three rendered images.")

    weights_path = Path(args.weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    # Re-create the same tokenizer/embedding change made in training so the instance
    # token's learned embedding lines up with the LoRA weights being loaded below.
    pipe.tokenizer.add_tokens(args.instance_token)
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    pipe.load_lora_weights(str(weights_path.parent), weight_name=weights_path.name)
    pipe.set_progress_bar_config(disable=True)

    sample_dir = Path(args.outdir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    images = pipe(
        [args.prompt] * args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images

    for i, image in enumerate(images, start=1):
        out = sample_dir / f"adapter_sample_{i}.png"
        image.save(out)
        print(out)


if __name__ == "__main__":
    main()
