#!/bin/bash
# FlashVLA training launcher.
#
# Usage:
#   bash train/train.sh <config.yaml> [num_gpus]      # num_gpus default: 8
#
# The selected config controls whether a multi-GPU launch uses DDP or FSDP2.
#
# Examples:
#   bash train/train.sh train/configs/pi05/libero/pi05_flashvla.yaml      # 8 GPUs (default)
#   bash train/train.sh train/configs/pi05/libero/pi05_flashvla.yaml 1    # single GPU
#

CONFIG=${1:?usage: bash train/train.sh <config.yaml> [num_gpus]}
NUM_GPUS=${2:-8}
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ "$NUM_GPUS" -gt 1 ]; then
  accelerate launch --multi_gpu --num_processes="$NUM_GPUS" \
    --module flashvla.train --config_path="$CONFIG"
else
  python -m flashvla.train --config_path="$CONFIG"
fi
