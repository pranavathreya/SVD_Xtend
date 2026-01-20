#!/bin/bash
#SBATCH --job-name="svd_xtend"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bdxf-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH -t 12:00:00
#SBATCH -e sout/slurm-%j.err
#SBATCH -o sout/slurm-%j.out

python validate_svd.py \
    --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid \
    --checkpoint=/work/hdd/bdxf/pathreya/svd_xtend_outputs/checkpoint-50000/ \
    --base_folder=/work/hdd/bdxf/WEB360/svd_xtend_frames \
    --output_dir=/work/hdd/bdxf/pathreya/svd_xtend_validation \
    --num_frames=50 \
    --fps=10

# python tools/gpu_memory_test.py \
#   --pretrained_model_name_or_path /work/hdd/bdxf/WEB360/hf/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e/ \
#   --width 1024 --height 512 \
#   --num_frames 1 --device cuda --mixed_precision fp16 --decode_chunk_size 1 --num_frames 

# /work/hdd/bdxf/WEB360/hf/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e