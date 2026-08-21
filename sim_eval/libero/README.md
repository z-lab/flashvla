# LIBERO Evaluation

Batched LIBERO eval with async chunk-overlap inference. `eval.sh` runs the full
multi-seed / multi-suite sweep on the released checkpoint; `eval.py` is the
underlying single-run script.

## Setup

LIBERO isn't in the base env — install the build tools, add the extra, and set
EGL for headless rendering. `--no-build-isolation` matters: the cmake that
lerobot's deps pull in via pip is a Python entry script, invisible inside pip's
isolated build env.

```bash
conda install -y cxx-compiler make
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install --no-build-isolation -e ".[libero]"
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

## Run

```bash
bash sim_eval/libero/eval.sh                 # 4 suites × 5 seeds, auto-resumes
GPUS="0 1 2 3" bash sim_eval/libero/eval.sh  # multi-GPU
```

Override via env vars (`POLICY_PATH`, `SUITES`, `N_EPISODES`, `GPUS`) and a first
positional `overlap_steps` (default `1` = async; `0` = sync). Each job writes
`eval_info.json` (per-task success + timing) to the output dir.

Single suite / debugging:

```bash
python sim_eval/libero/eval.py --policy.path=z-lab/flashvla-pi05-libero \
    --env.type=libero --env.task=libero_spatial --eval.n_episodes=50 \
    --policy.n_action_steps=5 --inference_overlap_steps=1 --policy.compile_model=true
```
