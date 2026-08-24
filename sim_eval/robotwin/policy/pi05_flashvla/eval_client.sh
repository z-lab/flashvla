#!/bin/bash

set -e

task_name=${1:-beat_block_hammer}
task_config=${2:-demo_clean}
port=${3:-9999}
seed=${4:-0}
gpu_id=${5:-0}

VENV=${ROBOTWIN_VENV:-/path/to/robotwin_venv}
PYTHON="${VENV}/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "robotwin venv not found at ${VENV}; run the install script first" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
# RoboTwin's sapien rendering uses EGL through the system NVIDIA driver. The
# vendor lib path keeps SAPIEN from falling back to a software renderer if
# the loader's default search misses it (same fix used by the LIBERO eval).
export MUJOCO_GL=${MUJOCO_GL:-egl}
export __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}

echo -e "\033[33mgpu id: ${gpu_id}, port: ${port}, seed: ${seed}\033[0m"
echo -e "\033[33mtask: ${task_name} / ${task_config}\033[0m"
echo -e "\033[33mpython: ${PYTHON}\033[0m"

cd ../..  # → RoboTwin root (eval_policy_client.py expects CWD here)

PYTHONWARNINGS=ignore::UserWarning \
${PYTHON} script/eval_policy_client.py \
    --config policy/pi05_flashvla/deploy_policy.yml \
    --port ${port} \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting flashvla_flashvla \
    --seed ${seed} \
    --policy_name pi05_flashvla
