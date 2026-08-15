# LingBot-VLA integration

FlashVLA supports the LingBot-VLA 4B architecture as both a conventional
flow-matching baseline (`policy.type: lingbot`) and a streaming policy
(`policy.type: lingbot-flashvla`). The implementation uses a Qwen2.5-VL vision
and language prefix plus a Qwen2 action expert joined through per-layer
attention.

## Checkpoints

Upstream LingBot checkpoints do not contain every runtime option in
`config.json`. Training or benchmarking from such a checkpoint therefore uses
an explicit FlashVLA YAML config, including a full `tokenizer_path` such as
`Qwen/Qwen2.5-VL-3B-Instruct`.

The loader accepts a single `model.safetensors`, a standard sharded index, or
LingBot's unindexed `model*.safetensors` shards. Loading is strict. The one
allowed architecture migration is the released joint action/time projection
to FlashVLA's separate time projection:

- four `action_time_mlp_*` tensors are intentionally dropped;
- four `time_mlp_*` tensors are initialized from scratch;
- any other missing, unexpected, or shape-mismatched tensor is an error.

For a fresh streaming run, `flashvla_init.mode: ae-norm-zero` resets the new
time MLP, restores the action expert's RMS scales to one, and zeros its
adaptive gamma/beta/gate projections. Resume skips this initialization.

## RoboTwin action layout

RoboTwin exposes 14-dimensional ALOHA state and action vectors:

```text
[left_arm(6), left_gripper, right_arm(6), right_gripper]
```

With `robotwin_feature_layout: true`, LingBot internally maps that vector to:

```text
arm(12) | arm_padding(2) | effectors(2) | padding_to_75
```

Only the 12 arm and 2 effector dimensions contribute to the training loss.
The saved processor pipelines own the raw-14 quantile normalization, so a
deployed checkpoint must include its processor JSON and safetensor files.

## Training

Baseline and streaming recipes are available at:

```text
train/configs/lingbot/robotwin/lingbot_vla_baseline.yaml
train/configs/lingbot/robotwin/lingbot_flashvla.yaml
```

The included streaming YAML is the generic 4B-base L2/FM recipe
(`N=4`, `C=20`, `n_action_steps=10`). It is intentionally not presented as
the distinct upstream RoboTwin-posttrained L1 recipe, which executes all 20
actions per slot.

Launch either through the normal entrypoint:

```bash
bash train/train.sh train/configs/lingbot/robotwin/lingbot_flashvla.yaml
```

The integration has been validated with the repository environment
(Python 3.12, PyTorch 2.9.1 + CUDA 12.8, Transformers 5.3, LeRobot 0.5.1,
and Accelerate 1.14). Run `pip check` after installing or updating the
environment; the vendored compatibility layer targets the stock Transformers
5 API rather than the older private-development pin.

The configs use the current policy-owned FSDP2 implementation. LingBot's wrap
plan covers Qwen vision blocks, VLM decoder layers, action-expert decoder
layers, the patch merger, and token embeddings. `use_lm_head=true` is rejected
for FSDP2 because the tied embedding/head parameter would cross communication
groups. Activation checkpointing remains disabled until a gradient-correct
implementation is available.

## Evaluation and latency

RoboTwin server/client instructions are in
[`sim_eval/robotwin/README.md`](../sim_eval/robotwin/README.md). The adapter
supports async overlap and lazy rendering: cached actions remain on CPU so
replaying them does not synchronize with in-flight GPU inference.

Latency configs are provided for both paths:

```text
benchmarks/configs/latency_lingbot_baseline.yaml
benchmarks/configs/latency_lingbot_flashvla.yaml
```

Depth-alignment and the private DINOv3 expert-vision tower from internal
LingBot variants are not part of this open-source integration. Configurations
that request either feature fail explicitly.
