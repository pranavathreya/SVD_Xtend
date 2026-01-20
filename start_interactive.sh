#!/bin/bash

srun --mem=50g \
     --nodes=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=16 \
     --partition=gpuA100x8 \
     --account=bdxf-delta-gpu \
     --gpus-per-node=2 \
     --time=01:00:00 \
     --pty \
     /bin/bash