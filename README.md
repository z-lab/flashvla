# FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference

**Paper** | **Blog**

**FlashVLA** is a general streaming action decoding method for flow-matching VLA models, achieving fast and asynchronous execution.

<p align="center">
  <a href="https://youtu.be/-qU_243aaVw">
    <img src="https://img.youtube.com/vi/-qU_243aaVw/maxresdefault.jpg" width="640" alt="FlashVLA demo video">
  </a>
</p>

## Installation

```bash
git clone https://github.com/z-lab/flashvla.git
cd flashvla
conda env create -f environment.yml
conda activate flashvla
```

This sets up the core FlashVLA environment for training, benchmarking, and
real-robot deployment. LIBERO and RoboTwin evaluation each need additional,
simulator-specific setup — see their READMEs under [`sim_eval/`](sim_eval/).

## Training

All training runs go through a single launcher, which selects the streaming or
baseline trainer from the config's policy type and handles single- or multi-GPU:

```bash
# single GPU
bash train/train.sh train/configs/pi05/libero/pi05_flashvla.yaml

# multi-GPU (e.g. 4 GPUs)
bash train/train.sh train/configs/pi05/libero/pi05_flashvla.yaml 4

# baseline policy (same launcher, picks the baseline trainer)
bash train/train.sh train/configs/pi05/libero/pi05_baseline.yaml
```

Configs live under [`train/configs/`](train/configs/) — pi0.5 and pi0, on LIBERO
and RoboTwin, in FlashVLA and baseline variants. For RoboTwin, point the launcher
at e.g.
[`train/configs/pi05/robotwin/pi05_flashvla_clean_per_task.yaml`](train/configs/pi05/robotwin/pi05_flashvla_clean_per_task.yaml).

## Evaluate on LIBERO

See [`sim_eval/libero/`](sim_eval/libero/) for the LIBERO environment setup and
how to launch single-run and full-sweep evaluation.

## Evaluate on RoboTwin

See [`sim_eval/robotwin/`](sim_eval/robotwin/) for the RoboTwin setup and the
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

## Acknowledgement

This project builds on [LeRobot](https://github.com/huggingface/lerobot) and
[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin).

## Citation

Citation information will be added with the paper release.

## License

Apache-2.0
