# RoboTwin Evaluation

Evaluates FlashVLA on [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)
via a client/server split: the FlashVLA policy runs in the `flashvla` env, the
SAPIEN simulator in RoboTwin's own env (incompatible torch/python). Only the
FlashVLA-specific pieces live here — RoboTwin itself is not vendored.

## Building the training dataset

For full 50-task training, the provided configs use
[`lerobot/robotwin_unified`](https://huggingface.co/datasets/lerobot/robotwin_unified)
directly, so no local dataset build is required. To train on a specific task or
configuration subset, use the pipeline below to download RoboTwin 2.0 and
convert the selected data into the LeRobot v3 layout. It runs in the `flashvla`
env (no GPU or simulator required) and is idempotent and resumable:

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
   cp -r policy/pi05_flashvla $ROBOTWIN/policy/      # π0.5 adapter
   cp -r policy/lingbot_flashvla $ROBOTWIN/policy/   # LingBot-VLA adapter
   cp overlay/script/*.py $ROBOTWIN/script/
   cp overlay/envs/_base_task.py $ROBOTWIN/envs/
   ```

   `overlay/` holds small patches over upstream RoboTwin (async client metrics, a
   server dispatch fix, and an `EVAL_STEP_LIM_OFFSET` env var that extends each
   task's step limit by the FlashVLA cold-start length; the reported 4-slot,
   16-action execution setting uses an offset of 48).
3. `pip install -e /path/to/flashvla` in the flashvla env.

## Run

Two terminals in two envs, from `$ROBOTWIN/policy/pi05_flashvla/`:

```bash
# terminal 1 — flashvla env (policy server)
bash eval_server.sh /path/to/flashvla_robotwin_ckpt 9999 0 current_state 1 16 true

# terminal 2 — RoboTwin env (SAPIEN sim)
EVAL_STEP_LIM_OFFSET=48 ROBOTWIN_VENV=/path/to/robotwin_venv \
  bash eval_client.sh beat_block_hammer demo_clean
```

`eval_server.sh` defaults to the released `z-lab/flashvla-pi05-robotwin` and sync
inference; for the async overlap used in the paper, pass overlap + compile
(`bash eval_server.sh <ckpt> 9999 0 current_state 1 16 true`). `cold_start_mode=current_state`
is required (RoboTwin is absolute-qpos — a zero action would crash the arm).
Results land in `$ROBOTWIN/eval_result/<task>/pi05_flashvla/...`.

### LingBot-VLA

Use the same two-terminal flow from `$ROBOTWIN/policy/lingbot_flashvla/`:

```bash
# terminal 1 — flashvla env
bash eval_server.sh /path/to/lingbot_flashvla_checkpoint

# terminal 2 — RoboTwin env
EVAL_STEP_LIM_OFFSET=48 ROBOTWIN_VENV=/path/to/robotwin_venv \
  bash eval_client.sh beat_block_hammer demo_clean
```

`eval_server.sh` deliberately requires an explicit checkpoint: pass either a
local exported checkpoint or an accessible Hugging Face model repo. Set
`TOKENIZER_PATH` when the checkpoint's saved Qwen repository path needs to be
overridden.

The default LingBot streaming recipe uses four buffer slots and executes 16
actions per call, so its padded cold start holds the current qpos for
`(4 - 1) × 16 = 48` simulator steps. Its `tokenizer_path` must point to a full
Qwen2.5-VL-3B-Instruct repository because the policy reads both tokenizer and
backbone configuration there. The adapter accepts raw 14-dimensional ALOHA
qpos and performs LingBot's sparse 75-dimensional typed-joint mapping inside
the policy; deploy a checkpoint together with its saved pre/postprocessors.
