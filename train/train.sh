#!/bin/bash
# FlashVLA training launcher.
#
# Usage:
#   bash train/train.sh <config.yaml> [num_gpus]      # num_gpus default: 8
#
# The training configs enable FSDP2 (bf16 mixed precision) by default, so the
# default 8-GPU launch runs sharded mixed-precision training out of the box.
#
# Examples:
#   bash train/train.sh train/configs/pi05/libero/pi05_flashvla.yaml      # 8 GPUs (default)
#   bash train/train.sh train/configs/pi05/libero/pi05_flashvla.yaml 1    # single GPU
#

CONFIG=${1:?usage: bash train/train.sh <config.yaml> [num_gpus]}
NUM_GPUS=${2:-8}
HERE="$(cd "$(dirname "$0")" && pwd)"

SCRIPT="$HERE/train.py"

if [ "$NUM_GPUS" -gt 1 ]; then
  accelerate launch --multi_gpu --num_processes="$NUM_GPUS" "$SCRIPT" --config_path="$CONFIG"
else
  python "$SCRIPT" --config_path="$CONFIG"
fi
