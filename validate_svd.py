#!/usr/bin/env python
# validate_svd.py
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers.utils import load_image
from train_svd import get_dataloader

# --- helpers (adapted from train_svd.py) ---
def export_to_gif(frames, output_path, fps=8):
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    pil_frames[0].save(output_path, format="GIF", append_images=pil_frames[1:], save_all=True, duration=int(1000/fps), loop=0)

# --- CLI ---
def parse_args():
    parser = argparse.ArgumentParser(description="Validate Stable Video Diffusion from a checkpoint or saved pipeline.")
    parser.add_argument(
        "--base_folder",
        required=True,
        type=str,
    )
    parser.add_argument("--pretrained_model_name_or_path", required=True, type=str,
                        help="Base pretrained model (same as used during training).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint directory (e.g., checkpoint-5000) or an output dir that contains 'unet' or 'unet_ema' subfolders. If omitted, loads models from `--pretrained_model_name_or_path`.")
    parser.add_argument("--output_dir", type=str, default="./validation_outputs")
    parser.add_argument("--reference_dir", type=str, default=None, help="Directory containing reference images to use for validation; if provided, will use all image files in it.")
    parser.add_argument("--reference_images", nargs="+", default=None, help="List of reference images to use for validation; if provided, will loop over them. Ignored if --reference_dir is set.")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--num_validation_images", type=int, default=1)
    parser.add_argument("--motion_bucket_id", type=int, default=127)
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--noise_aug_strength", type=float, default=0.02)
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g., 'cuda' or 'cpu'). Default: cuda if available.")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # load encoders / vae
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder")
    #feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", variant="fp16" if weight_dtype==torch.float16 else None)

    # decide unet path or load default
    unet = None
    if not args.checkpoint:
        raise ValueError("You must provide --checkpoint to validate Stable Video Diffusion.")
    
    if os.path.isdir(os.path.join(args.checkpoint, "unet")):
        unet = UNetSpatioTemporalConditionModel.from_pretrained(args.checkpoint, subfolder="unet", torch_dtype=weight_dtype)
    else:
        raise ValueError(f"Cannot find 'unet' in checkpoint '{args.checkpoint}'.")

    # Move models to device / dtype
    image_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    # Build pipeline
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=weight_dtype,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)

    os.makedirs(args.output_dir, exist_ok=True)
    val_dir = os.path.join(args.output_dir)
    os.makedirs(val_dir, exist_ok=True)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # Run validation
    # Determine reference images list. Priority: --reference_dir (all images in dir) > --reference_images > --image
    # if getattr(args, "reference_dir", None):
    #     exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
    #     if not os.path.isdir(args.reference_dir):
    #         raise ValueError(f"reference_dir does not exist or is not a directory: {args.reference_dir}")
    #     reference_images = sorted([
    #         os.path.join(args.reference_dir, f)
    #         for f in os.listdir(args.reference_dir)
    #         if f.lower().endswith(exts)
    #     ])
    #     if len(reference_images) == 0:
    #         raise ValueError(f"No image files found in reference_dir: {args.reference_dir}")
    # else:
    #     reference_images = args.reference_images if getattr(args, "reference_images", None) is not None else [args.image]
    val_dataloader, _, val_indices = get_dataloader(
        base_folder=args.base_folder,
        train=False,
        width=args.width,
        height=args.height,
        num_frames=1,
        per_gpu_batch_size=1,
    )
    print(f"Created DataLoader (dataset size={len(val_indices):,}")

    for smpl in iter(val_dataloader):
        ref = smpl['frame_paths'][0][0]  # use first frame path as reference since num_frames=1
        img = load_image(ref).convert("RGB").resize((args.width, args.height))
        for idx in range(args.num_validation_images):
            out = pipeline(img,
                           height=args.height,
                           width=args.width,
                           num_frames=args.num_frames,
                           decode_chunk_size=8,
                           motion_bucket_id=args.motion_bucket_id,
                           fps=args.fps,
                           noise_aug_strength=args.noise_aug_strength,
                           generator=generator)
            frames = out.frames[0]  # list[PIL.Image]
            frames_np = [np.array(f) for f in frames]

            basename = Path(args.checkpoint).name if args.checkpoint else Path(args.pretrained_model_name_or_path).name
            gif_path = os.path.join(val_dir, f"{basename}_{Path(ref).stem}_val_{idx}.gif")
            mp4_path = os.path.join(val_dir, f"{basename}_{Path(ref).stem}_val_{idx}.mp4")

            # Save GIF
            export_to_gif(frames_np, gif_path, fps=args.fps)

            # Save frames to folder:
            frames_path = os.path.join(val_dir + "_frames", f"{basename}_{Path(ref).stem}")
            Path(frames_path).mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(frames):
                f.save(os.path.join(frames_path, f"{Path(ref).stem}_{i}.png"))


            # Optionally save MP4 (simple OpenCV writer if available)
            try:
                import cv2
                h, w, _ = frames_np[0].shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_vid = cv2.VideoWriter(mp4_path, fourcc, args.fps, (w, h))
                for f in frames_np:
                    out_vid.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                out_vid.release()
            except Exception:
                # If cv2 not available, skip mp4
                pass

            print(f"Saved: {gif_path}")

if __name__ == "__main__":
    main()