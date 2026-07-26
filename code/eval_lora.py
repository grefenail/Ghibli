"""Load base SD 1.5 plus the trained LoRA adapter and render sample images.

No style-strength slider is exposed, per the assignment brief.
"""
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from diffusers import StableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA style adapter for SD 1.5.")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--weights", type=str, required=True, help="Path to the pytorch_lora_weights.safetensors file.")
    parser.add_argument("--instance_token", type=str, default="<sks>")
    parser.add_argument("--prompt", type=str, default="a busy market, in <sks> style")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="no deformed face,no extra limbs, no extra head, not too many people, no face with multiple eyes",
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--guidance_scales",
        type=str,
        default="7.5",
        help="Comma-separated CFG values to sweep, e.g. '4.0,5.5,7.5,10.0'. "
        "Overrides --guidance_scale and renders one image per seed for each value.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="36,46,350,460,780,1005,1120,1250,1610,1850",
        help="Comma-separated seeds to use, e.g. '1,45,50,95,500'. "
        "Overrides --num_images/--seed with an evenly spaced default of "
        "--num_images seeds if not given.",
    )
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

    # diffusers' LoRA loader strictly requires every key to contain "lora" - pull our
    # non-LoRA embedding key out first and hand it a clean dict instead of the raw file,
    # otherwise load_lora_weights raises "Invalid LoRA checkpoint".
    state_dict = load_file(weights_path)
    token_embedding = state_dict.pop("sks_token_embedding", None)
    pipe.load_lora_weights(state_dict)

    if token_embedding is not None:
        token_id = pipe.tokenizer.convert_tokens_to_ids(args.instance_token)
        pipe.text_encoder.get_input_embeddings().weight.data[token_id] = token_embedding.to(
            pipe.text_encoder.get_input_embeddings().weight.dtype
        )
        print(f"Loaded learned {args.instance_token} embedding from {weights_path}")

    pipe.set_progress_bar_config(disable=True)

    sample_dir = Path(args.outdir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    if args.seeds is None and args.guidance_scales is None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        images = pipe(
            [args.prompt] * args.num_images,
            negative_prompt=[args.negative_prompt] * args.num_images if args.negative_prompt else None,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images

        for i, image in enumerate(images, start=1):
            out = sample_dir / f"adapter_sample_{i}.png"
            image.save(out)
            print(out)
        return

    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        # Evenly spaced seeds instead of clustering near args.seed.
        step = 10000 // args.num_images
        seeds = [args.seed + i * step for i in range(args.num_images)]

    guidance_scales = (
        [float(g) for g in args.guidance_scales.split(",")]
        if args.guidance_scales is not None
        else [args.guidance_scale]
    )

    for cfg in guidance_scales:
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            image = pipe(
                args.prompt,
                negative_prompt=args.negative_prompt or None,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=cfg,
                generator=generator,
            ).images[0]
            out = sample_dir / f"sample_cfg{cfg}_seed{seed}.png"
            image.save(out)
            print(out)


if __name__ == "__main__":
    main()
