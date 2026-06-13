# FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference

**Paper** | **Project Page** | **Demo Video** | **Checkpoints**

**FlashVLA** is a general streaming action decoding method for flow-matching
VLA models. This repo provides training, simulation evaluation, latency
benchmark, and real-robot deployment code. The commands below use pi0.5 as an
example.

<p align="center">
  <img alt="FlashVLA" src="assets/logo.png" width="42%">
</p>

> Demo video and pi0.5 checkpoints will be added soon.

## Installation

```bash
git clone https://github.com/z-lab/flashvla.git
cd flashvla
conda env create -f environment.yml
conda activate flashvla
```

`environment.yml` installs the LIBERO extra by default. For training,
benchmarking, RoboTwin, or real-robot deployment without LIBERO, replace
`-e .[libero]` with `-e .` in `environment.yml` before creating the env.

## Training pi0.5

Train FlashVLA on LIBERO:

```bash
python train/train.py \
  --config_path=train/configs/pi05/libero/flashvla_action.yaml
```

Multi-GPU:

```bash
accelerate launch --multi_gpu --num_processes=4 \
  train/train.py \
  --config_path=train/configs/pi05/libero/flashvla_action.yaml
```

Train a baseline policy:

```bash
python train/train_baseline.py \
  --config_path=train/configs/pi05/sync.yaml
```

For real-world Franka training, use
[`train/configs/pi05/franka/flashvla_action_franka_dynamic_pap.yaml`](train/configs/pi05/franka/flashvla_action_franka_dynamic_pap.yaml).
For RoboTwin training, use
[`train/configs/pi05/robotwin/flashvla_action_per_task.yaml`](train/configs/pi05/robotwin/flashvla_action_per_task.yaml).

## Testing pi0.5 on LIBERO

See [`sim_eval/libero/`](sim_eval/libero/) for single-run and full-sweep
evaluation.

## Testing pi0.5 on RoboTwin

See [`sim_eval/robotwin/`](sim_eval/robotwin/) for RoboTwin setup and
server/client evaluation.

## Latency Benchmark

```bash
python benchmarks/benchmark_latency.py \
  --config_path=benchmarks/configs/latency_flashvla.yaml

python benchmarks/benchmark_latency.py \
  --config_path=benchmarks/configs/latency_baseline.yaml
```

Use `--num_views=1`, `--num_views=2`, or `--num_views=3` to sweep camera views.

## Real-Robot Deployment

```python
from flashvla.async_manager import AsyncStreamingActionManager

manager = AsyncStreamingActionManager(policy, overlap_steps=1)
manager.warmup(preprocessor(first_obs))
manager.reset()

while running:
    action = manager.act(preprocessor(obs))
    robot.send_action(postprocessor(action)[0].cpu().numpy())
```

See [`realworld/README.md`](realworld/README.md) for the full deployment
contract.

## Repository Layout

```text
flashvla/      library code
train/         training scripts and configs
sim_eval/      LIBERO and RoboTwin evaluation
benchmarks/    latency benchmark
realworld/     real-robot deployment notes
assets/        images for README
```

## Acknowledgement

This project builds on [LeRobot](https://github.com/huggingface/lerobot) and
[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin).

## Citation

Citation information will be added with the paper release.

## License

Apache-2.0
