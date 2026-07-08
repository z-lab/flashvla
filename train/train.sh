#!/bin/bash
# FlashVLA training launcher.
#
# Usage:
#   bash train/train.sh <config.yaml> [num_gpus]
#
# Examples:
#   bash train/train.sh train/configs/pi05/libero/pi05_flashvla.yaml      # single GPU
#   bash train/train.sh train/configs/pi05/libero/pi05_flashvla.yaml 4    # 4 GPUs
#

CONFIG=${1:?usage: bash train/train.sh <config.yaml> [num_gpus]}
NUM_GPUS=${2:-1}
HERE="$(cd "$(dirname "$0")" && pwd)"

SCRIPT="$HERE/train.py"

if [ "$NUM_GPUS" -gt 1 ]; then
  accelerate launch --multi_gpu --num_processes="$NUM_GPUS" "$SCRIPT" --config_path="$CONFIG"
else
  python "$SCRIPT" --config_path="$CONFIG"
fi
