#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

task_name=${1:-beat_block_hammer}
task_config=${2:-demo_clean}
port=${3:-9999}
seed=${4:-0}
gpu_id=${5:-0}

if [[ -z "${ROBOTWIN_VENV:-}" ]]; then
    echo "ROBOTWIN_VENV must point to the installed RoboTwin virtual environment" >&2
    exit 1
fi
VENV=${ROBOTWIN_VENV}
PYTHON="${VENV}/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "robotwin venv not found at ${VENV}; run the install script first" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
# RoboTwin's sapien rendering uses EGL through the system NVIDIA driver. The
# vendor lib path keeps SAPIEN from falling back to a software renderer if the
# loader's default search misses it (same fix used by the LIBERO eval).
export MUJOCO_GL=${MUJOCO_GL:-egl}
export __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}
# FlashVLA holds the current qpos for the first (num_buffer_slots - 1)
# observations while the denoising buffer fills. RoboTwin evaluation uses
# num_buffer_slots=4 and n_action_steps=16, so 3 x 16 = 48
# stationary steps; extend each task's step budget by that much so the cold
# start does not eat into the task horizon.
buffer_slots=${BUFFER_SLOTS:-4}
executed_steps=${N_ACTION_STEPS:-16}
export EVAL_STEP_LIM_OFFSET=${EVAL_STEP_LIM_OFFSET:-$(((buffer_slots - 1) * executed_steps))}

echo -e "\033[33mgpu id: ${gpu_id}, port: ${port}, seed: ${seed}\033[0m"
echo -e "\033[33mtask: ${task_name} / ${task_config}; step offset: ${EVAL_STEP_LIM_OFFSET}\033[0m"
echo -e "\033[33mpython: ${PYTHON}\033[0m"

cd "${ROBOTWIN_ROOT}"  # eval_policy_client.py expects the RoboTwin root

PYTHONWARNINGS=ignore::UserWarning \
${PYTHON} script/eval_policy_client.py \
    --config "${SCRIPT_DIR}/deploy_policy.yml" \
    --port ${port} \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting lingbot_flashvla \
    --seed ${seed} \
    --policy_name lingbot_flashvla
