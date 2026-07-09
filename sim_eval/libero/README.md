# LIBERO Evaluation

Batched LIBERO evaluation with async chunk-overlap inference. `eval.py` in
this directory is the implementation (a standalone script on top of the
`flashvla` library); `eval.sh` is the multi-seed /
multi-suite / multi-GPU orchestrator.

## Requirements

The base `environment.yml` installs the FlashVLA env **without** LIBERO. After
creating and activating it (`conda env create -f environment.yml && conda activate
flashvla`), add the LIBERO extra:

```bash
# egl_probe builds a tiny native extension via cmake. Its CMakeLists predates
# CMake 4 (declares cmake_minimum_required < 3.5), which CMake 4.x refuses by
# default — so pass CMAKE_POLICY_VERSION_MINIMUM=3.5 to let the modern cmake
# (shipped by environment.yml) configure it.
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install -e ".[libero]"
```

This needs a `cmake` and a C/C++ compiler on PATH; `environment.yml` provides
them via conda (otherwise `conda install -c conda-forge cmake make c-compiler
cxx-compiler`, or `sudo apt install cmake build-essential`).

> Do **not** `pip uninstall cmake`: the conda `cmake` and lerobot's PyPI `cmake`
> dependency share the same `$CONDA_PREFIX/bin/cmake`, so uninstalling the pip
> one deletes the binary entirely and the build then fails with
> `RuntimeError: CMake must be installed`.

Headless rendering needs EGL:

```bash
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

## Single run

```bash
python sim_eval/libero/eval.py \
    --policy.path=/path/to/flashvla_action_libero_ckpt \
    --policy.compile_model=true --policy.fuse_qkv=true --policy.fuse_gate_up=true \
    --env.type=libero --env.task=libero_spatial \
    --env.max_parallel_tasks=1 --eval.batch_size=1 --eval.n_episodes=50 \
    --policy.n_action_steps=5 --inference_overlap_steps=1 \
    --output_dir=outputs/eval/libero_spatial --seed=1000
```

Key knobs:
- `--inference_overlap_steps` — `0` = sync (chunk boundary blocks on
  inference); `N >= 1` = launch the next chunk inference N steps early and
  hide it behind env stepping. Requires `--policy.compile_model=true`
  (CUDA-graph dispatch must be non-blocking).
- `--policy.n_action_steps` — actions executed per chunk before replanning.
  Smaller values replan more often (fresher observations); `5` was the best
  LIBERO setting for chunk_size=10 checkpoints.
- The first rollout triggers a one-shot warmup that captures the CUDA graph
  (so episode-1 latency isn't dominated by torch.compile autotune).

The script writes `eval_info.json` (per-task success rates, timing stats,
async transition latency percentiles) into `--output_dir`.

## Full sweep (4 suites x N seeds, multi-GPU)

```bash
# overlap=1, seeds 1000/2000/3000, 4 GPUs
GPUS="0 1 2 3" N_EPISODES=50 POLICY_PATH=/path/to/ckpt \
  bash sim_eval/libero/eval.sh 1 1000 2000 3000
```

- One subprocess per (seed, suite), pulled from a shared queue so a slow
  suite doesn't idle other GPUs; resumes by skipping any (seed, suite) whose
  `eval_info.json` already exists.
- `COMPILE_MODEL=true` can be preset in the env to force-compile sync runs
  too (compile does not change success rates, only speed).
