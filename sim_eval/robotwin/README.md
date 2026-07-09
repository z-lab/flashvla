# RoboTwin Evaluation

Evaluates FlashVLA policies on [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)
via a TCP client/server split: the **server** runs the FlashVLA policy in the
`flashvla` python env; the **client** runs the SAPIEN simulator in RoboTwin's
own env (different torch/python constraints). This directory contains only
the FlashVLA-specific pieces — RoboTwin itself is not vendored. It also ships a
one-click pipeline to build the RoboTwin **training** dataset (see below).

```
robotwin/
├── robotwin_pipeline/        # download + convert RoboTwin 2.0 -> LeRobot v3 training data
├── policy/
│   └── pi05_flashvla/        # flashvla server adapter (eval_server.sh, eval_client.sh, model wrapper)
└── overlay/                  # small patches to RoboTwin core (see below)
    ├── script/policy_model_server.py
    ├── script/eval_policy_client.py
    └── envs/_base_task.py
```

## Building the training dataset

`robotwin_pipeline/` downloads the official RoboTwin 2.0 dataset and converts it
into the LeRobot v3 layout FlashVLA trains on (`train/configs/pi05/robotwin/`).
One-click, runs in the `flashvla` env — no GPU, no RoboTwin sim env, no SLURM:

```bash
conda activate flashvla
bash robotwin_pipeline/run_pipeline.sh /path/to/data_dir
```

It runs all stages idempotently (safe to resume): download
`TianxingChen/RoboTwin2.0` (aloha-agilex `clean_50` + `randomized_500`, ~273 GB)
→ unzip → convert each task×setting to LeRobot v3 → augment per-task quantile
stats (q01/q99) → pool exact global stats. Options:
`bash robotwin_pipeline/run_pipeline.sh [DATA_DIR] [PARALLELISM]`, plus env
overrides `SETTINGS="aloha-agilex_clean_50"` (one setting) and `SKIP_DOWNLOAD=1`.

Then point a training config at the result:

```yaml
robotwin_multitask:
  enable: true
  root: <data_dir>/RoboTwin-LeRobot-v3.0
  config_subdirs: [aloha-agilex_clean_50, aloha-agilex_randomized_500]
  stats_path: <data_dir>/RoboTwin-LeRobot-v3.0/_pooled_stats/pooled_stats.json
```

The `robotwin_pipeline/scripts/` (convert / augment / compute) are also runnable
standalone. (~273 GB raw — delete the zips after conversion to reclaim space.)

## Setup

1. **Clone upstream RoboTwin** and follow its installation guide (SAPIEN,
   curobo, assets download) in a dedicated python env:

   ```bash
   git clone https://github.com/RoboTwin-Platform/RoboTwin.git
   # follow RoboTwin's INSTALL instructions in its own env
   ```

2. **Copy the FlashVLA pieces into the RoboTwin tree**:

   ```bash
   ROBOTWIN=/path/to/RoboTwin
   cp -r policy/pi05_flashvla $ROBOTWIN/policy/
   cp overlay/script/policy_model_server.py overlay/script/eval_policy_client.py $ROBOTWIN/script/
   cp overlay/envs/_base_task.py $ROBOTWIN/envs/
   ```

   The overlay files carry three small modifications on top of upstream
   (re-apply by hand if your RoboTwin version has diverged):
   - `script/eval_policy_client.py` (+95 lines): async-aware `ModelClient`
     with socket timeouts and per-episode efficiency metrics
     (`steps/ep`, `wall/ep`, `time/step`, per-call latency) written to
     `_metrics.json`.
   - `script/policy_model_server.py` (+7 lines): server dispatch fix for
     policy adapters.
   - `envs/_base_task.py` (+10 lines): `EVAL_STEP_LIM_OFFSET` env var adds a
     constant to each task's step limit — set it to
     `(num_buffer_slots - 1) * chunk_size` (e.g. 40) for flashvla
     policies to compensate for the stationary cold-start steps. Baseline
     policies don't need it.

3. **Install flashvla in the server env** (separate from the RoboTwin env):

   ```bash
   conda activate flashvla
   pip install -e /path/to/flashvla
   ```

## Running an eval

Terminal 1 — server (flashvla env, GPU 0):

```bash
cd $ROBOTWIN/policy/pi05_flashvla
bash eval_server.sh /path/to/flashvla_robotwin_ckpt 9999 0 current_state 1 10 true
#             ckpt                            port gpu cold_start  o n_act compile
```

Terminal 2 — client (RoboTwin env, GPU 1):

```bash
cd $ROBOTWIN/policy/pi05_flashvla
EVAL_STEP_LIM_OFFSET=40 ROBOTWIN_VENV=/path/to/robotwin_venv \
  bash eval_client.sh beat_block_hammer demo_clean 9999 0 1
#                     task              task_config port seed gpu
```

Results land at
`$ROBOTWIN/eval_result/<task>/pi05_flashvla/<task_config>/.../_result.txt`
(success rate) and `_metrics.json` (efficiency metrics).

Notes:
- `cold_start_mode=current_state` is required on RoboTwin (absolute-qpos
  control — a zero action would crash the arm).
- `compile=true` + `inference_overlap_steps>=1` enables async overlap: the
  server's chunk inference is hidden behind SAPIEN env stepping.
- First server boot pays torch.compile autotune (~minutes); subsequent boots
  reuse the inductor cache. `eval_server.sh` warms up the CUDA graph before
  accepting connections so the client's first request doesn't time out.
- See `policy/pi05_flashvla/README.md` for the full protocol / version
  conflict matrix.
