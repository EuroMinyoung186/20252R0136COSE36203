#!/bin/bash

# GPU 자동 탐지
NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "Detected $NUM_GPUS GPUs. Starting training..."

torchrun --nproc_per_node=$NUM_GPUS fashion_recommendation/train.py \
    --img_base_path data/cloth/img_square \
    --json_file data/cloth/train.json \
    --save_path fashion_recommendation/ckpt/ckpt.pt