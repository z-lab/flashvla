#!/bin/bash
# One-line LIBERO evaluation with the released FlashVLA pi0.5 checkpoint.
#
# Usage:
#   bash sim_eval/libero/eval.sh
#
# Override any default via env vars, e.g.:
#   POLICY_PATH=/path/to/ckpt TASK=libero_spatial N_EPISODES=50 \
#     bash sim_eval/libero/eval.sh
#
# Requires the LIBERO extra and headless-render env — see sim_eval/libero/README.md.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

POLICY_PATH="${POLICY_PATH:-z-lab/flashvla-pi05-libero}"
TASK="${TASK:-libero_spatial,libero_object,libero_goal,libero_10}"
N_EPISODES="${N_EPISODES:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-5}"
INFERENCE_OVERLAP_STEPS="${INFERENCE_OVERLAP_STEPS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval/libero}"
SEED="${SEED:-1000}"

# Default to EGL headless rendering unless the caller set MUJOCO_GL.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

python "$HERE/eval.py" \
    --policy.path="$POLICY_PATH" \
    --policy.compile_model=true --policy.fuse_qkv=true --policy.fuse_gate_up=true \
    --env.type=libero --env.task="$TASK" \
    --env.max_parallel_tasks=1 --eval.batch_size=1 --eval.n_episodes="$N_EPISODES" \
    --policy.n_action_steps="$N_ACTION_STEPS" --inference_overlap_steps="$INFERENCE_OVERLAP_STEPS" \
    --output_dir="$OUTPUT_DIR" --seed="$SEED"
