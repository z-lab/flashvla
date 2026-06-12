#!/bin/bash
# Start the pi05_flashvla model server.
#
# Runs in the FlashVLA conda env (torch 2.7 + flashvla + transformers).
# Listens on a TCP port; eval_client.sh connects to it from the RoboTwin env.
#
# Usage:
#   bash serve.sh <policy_path> [port] [gpu_id] [cold_start_mode] \
#                 [inference_overlap_steps] [n_action_steps] [compile_model]
#
# Examples:
#   # Sync (default)
#   bash serve.sh /path/to/flashvla_robotwin_ckpt 9999 0
#
#   # Async overlap=1 with compile + n_action_steps=5
#   bash serve.sh /path/to/flashvla_robotwin_ckpt 9999 0 current_state 1 5 true
#
# Prerequisites (one-time setup in the flashvla env):
#   conda activate flashvla   # the env where `pip install -e flashvla` was run
#   pip install -e /path/to/flashvla

set -e

policy_path=${1:?"usage: $0 <policy_path> [port] [gpu_id] [cold_start_mode] [inference_overlap_steps] [n_action_steps] [compile_model]"}
port=${2:-9999}
gpu_id=${3:-0}
cold_start_mode=${4:-current_state}
inference_overlap_steps=${5:-0}
n_action_steps=${6:-}
compile_model=${7:-}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id: ${gpu_id}, port: ${port}\033[0m"
echo -e "\033[33mpolicy_path: ${policy_path}\033[0m"
echo -e "\033[33mcold_start_mode: ${cold_start_mode}\033[0m"
echo -e "\033[33minference_overlap_steps: ${inference_overlap_steps}\033[0m"
echo -e "\033[33mn_action_steps: ${n_action_steps:-(use ckpt default)}\033[0m"
echo -e "\033[33mcompile_model: ${compile_model:-(use ckpt default)}\033[0m"

# Build override list — only include n_action_steps / compile_model when set,
# because policy_model_server.py's override parser doesn't understand "null"
# (it would store the literal string and break downstream int/bool coercion).
overrides=(
    --policy_name pi05_flashvla
    --policy_path "${policy_path}"
    --cold_start_mode "${cold_start_mode}"
    --device cuda
    --inference_overlap_steps "${inference_overlap_steps}"
)
if [[ -n "${n_action_steps}" ]]; then
    overrides+=(--n_action_steps "${n_action_steps}")
fi
if [[ -n "${compile_model}" ]]; then
    overrides+=(--compile_model "${compile_model}")
fi

cd ../..  # → RoboTwin root (policy_model_server.py expects CWD here)

PYTHONWARNINGS=ignore::UserWarning \
python script/policy_model_server.py \
    --config policy/pi05_flashvla/deploy_policy.yml \
    --port ${port} \
    --overrides "${overrides[@]}"
