# RoboTwin Evaluation

Evaluates FlashVLA on [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)
via a client/server split: the FlashVLA policy runs in the `flashvla` env, the
SAPIEN simulator in RoboTwin's own env (incompatible torch/python). Only the
FlashVLA-specific pieces live here — RoboTwin itself is not vendored.

## Building the training dataset

Downloads RoboTwin 2.0 and converts it into the LeRobot v3 layout FlashVLA trains
on. Runs in the `flashvla` env (no GPU, no sim), idempotent and resumable:

```bash
bash robotwin_pipeline/run_pipeline.sh /path/to/data_dir
```

Then point the training config at the result:

```yaml
robotwin_multitask:
  root: <data_dir>/RoboTwin-LeRobot-v3.0
  config_subdirs: [aloha-agilex_clean_50, aloha-agilex_randomized_500]
  stats_path: <data_dir>/RoboTwin-LeRobot-v3.0/_pooled_stats/pooled_stats.json
```

## Setup

1. Clone [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) and install it
   in its own env (SAPIEN, curobo, assets — follow its INSTALL guide).
2. Copy the FlashVLA pieces into the RoboTwin tree:

   ```bash
   ROBOTWIN=/path/to/RoboTwin
   cp -r policy/pi05_flashvla $ROBOTWIN/policy/
   cp overlay/script/*.py $ROBOTWIN/script/
   cp overlay/envs/_base_task.py $ROBOTWIN/envs/
   ```

   `overlay/` holds small patches over upstream RoboTwin (async client metrics, a
   server dispatch fix, and an `EVAL_STEP_LIM_OFFSET` env var that extends each
   task's step limit by the flashvla cold-start length, e.g. 40).
3. `pip install -e /path/to/flashvla` in the flashvla env.

## Run

Two terminals in two envs, from `$ROBOTWIN/policy/pi05_flashvla/`:

```bash
# terminal 1 — flashvla env (policy server)
bash eval_server.sh

# terminal 2 — RoboTwin env (SAPIEN sim)
EVAL_STEP_LIM_OFFSET=40 ROBOTWIN_VENV=/path/to/robotwin_venv \
  bash eval_client.sh beat_block_hammer demo_clean
```

`eval_server.sh` defaults to the released `z-lab/flashvla-pi05-robotwin` and sync
inference; for the async overlap used in the paper, pass overlap + compile
(`bash eval_server.sh <ckpt> 9999 0 current_state 1 5 true`). `cold_start_mode=current_state`
is required (RoboTwin is absolute-qpos — a zero action would crash the arm).
Results land in `$ROBOTWIN/eval_result/<task>/pi05_flashvla/...`.
