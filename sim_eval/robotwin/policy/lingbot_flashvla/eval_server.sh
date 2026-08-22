#!/bin/bash
# Start the lingbot_flashvla model server.
#
# Runs in the FlashVLA conda env (torch + flashvla + transformers). Listens on
# a TCP port; eval_client.sh connects to it from the RoboTwin env.
#
# Usage:
#   bash eval_server.sh [policy_path] [port] [gpu_id] [cold_start_mode] \
#                 [inference_overlap_steps] [n_action_steps] [compile_model] \
#                 [skip_stale_actions]
#
# Examples:
#   # Sync
#   bash eval_server.sh /path/to/lingbot_flashvla_checkpoint 9999 0
#
#   # Async overlap=1 with compile
#   bash eval_server.sh /path/to/lingbot_ckpt 9999 0 current_state 1 10 true
#
# Env knobs:
#   SKIP_STALE_ACTIONS=1  force RTC realignment server-wide (same effect as the
#     8th positional arg; requires inference_overlap_steps + n_action_steps
#     <= chunk_size). e.g. SKIP_STALE_ACTIONS=1 bash eval_server.sh ... 1 10 true
#
# Prerequisites (one-time setup in the flashvla env):
#   conda activate flashvla   # the env where `pip install -e flashvla` was run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

policy_path=${1:-}
if [[ -z "${policy_path}" ]]; then
    echo "Usage: bash eval_server.sh POLICY_PATH [port] [gpu_id] [cold_start_mode] [inference_overlap_steps] [n_action_steps] [compile_model] [skip_stale_actions]" >&2
    echo "POLICY_PATH must be a local exported checkpoint or an accessible Hugging Face model repo." >&2
    exit 2
fi
port=${2:-9999}
gpu_id=${3:-0}
cold_start_mode=${4:-current_state}
inference_overlap_steps=${5:-0}
n_action_steps=${6:-}
compile_model=${7:-}
skip_stale_actions=${8:-}
tokenizer_path=${TOKENIZER_PATH:-}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id: ${gpu_id}, port: ${port}\033[0m"
echo -e "\033[33mpolicy_path: ${policy_path}\033[0m"
echo -e "\033[33mcold_start_mode: ${cold_start_mode}\033[0m"
echo -e "\033[33minference_overlap_steps: ${inference_overlap_steps}\033[0m"
echo -e "\033[33mn_action_steps: ${n_action_steps:-(use ckpt default)}\033[0m"
echo -e "\033[33mcompile_model: ${compile_model:-(use ckpt default)}\033[0m"
echo -e "\033[33mskip_stale_actions: ${skip_stale_actions:-false}\033[0m"
echo -e "\033[33mtokenizer_path: ${tokenizer_path:-(use ckpt default)}\033[0m"

# Build override list — only include n_action_steps / compile_model when set,
# because policy_model_server.py's override parser doesn't understand "null"
# (it would store the literal string and break downstream int/bool coercion).
overrides=(
    --policy_name lingbot_flashvla
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
if [[ -n "${skip_stale_actions}" ]]; then
    overrides+=(--skip_stale_actions "${skip_stale_actions}")
fi
if [[ -n "${tokenizer_path}" ]]; then
    overrides+=(--tokenizer_path "${tokenizer_path}")
fi

cd "${ROBOTWIN_ROOT}"  # policy_model_server.py expects the RoboTwin root

PYTHONWARNINGS=ignore::UserWarning \
python script/policy_model_server.py \
    --config "${SCRIPT_DIR}/deploy_policy.yml" \
    --port ${port} \
    --overrides "${overrides[@]}"
