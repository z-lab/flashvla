# FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference

[**Paper**](https://arxiv.org/abs/2608.27384) | [**Blog**](https://z-lab.ai/projects/flashvla/) | [**Models**](https://huggingface.co/collections/z-lab/flashvla)

**FlashVLA** is a general streaming action decoding method for flow-matching VLA models, achieving fast and asynchronous execution.

## Install

```bash
git clone https://github.com/z-lab/flashvla.git
cd flashvla
conda env create -f environment.yml
conda activate flashvla
```

This installs the core FlashVLA env for training, benchmarking, and real-robot
deployment. LIBERO and RoboTwin evaluation each need extra, simulator-specific
setup — see their READMEs under [`sim_eval/`](sim_eval/).

## Evaluation

Evaluate the released FlashVLA pi0.5 checkpoints with async chunk-overlap inference.

**LIBERO** ([`z-lab/flashvla-pi05-libero`](https://huggingface.co/z-lab/flashvla-pi05-libero)):

```bash
bash sim_eval/libero/eval.sh
```

**RoboTwin 2.0** ([`z-lab/flashvla-pi05-robotwin`](https://huggingface.co/z-lab/flashvla-pi05-robotwin))
uses a server (flashvla env) / client (RoboTwin env) split. After the one-time setup in
[`sim_eval/robotwin/`](sim_eval/robotwin/), from `$ROBOTWIN/policy/pi05_flashvla/`:

```bash
bash eval_server.sh                 # terminal 1 — flashvla env (starts the policy server)
ROBOTWIN_VENV=... bash eval_client.sh   # terminal 2 — RoboTwin env (runs the SAPIEN sim)
```

See [`sim_eval/libero/`](sim_eval/libero/) and [`sim_eval/robotwin/`](sim_eval/robotwin/)
for setup and full options (single run, multi-seed sweeps, and the RoboTwin server/client).

## Performance

π0.5 vs. **+ FlashVLA** (**bold** = the better result). `d` denotes how many
steps before the current chunk ends the next inference is launched: `d=0` is
synchronous, while `d≥1` overlaps inference with robot execution.

### LIBERO

Success rate (%) + per-step latency (averaged over 2,000 episodes):

| Method | Spatial | Object | Goal | Long | Avg | Time/Step (ms) |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| π0.5 | **98.8** | 98.2 | **98.0** | 92.4 | 96.9 | 53.8 |
| **+ FlashVLA** (`d=0`) | 98.6 | 99.0 | 97.8 | **96.2** | **97.9** (↑1.0) | – |
| **+ FlashVLA** (`d=1`) | **98.8** | **99.6** | 97.6 | **95.4** | **97.8** (↑0.9) | **22.1** (2.43×) |
| **+ FlashVLA** (`d=2`) | **99.0** | **99.8** | 97.2 | **97.4** | **98.3** (↑1.4) | **20.6** (2.62×) |

### RoboTwin 2.0

50-task multitask success rate (%):

| Method | Clean | Random | Avg |
|:--|:--:|:--:|:--:|
| π0.5 | 86.1 | 85.8 | 86.0 |
| **+ FlashVLA** (`d=0`) | **90.8** | **90.2** | **90.5** (↑4.5) |
| **+ FlashVLA** (`d=1`) | **91.2** | **89.9** | **90.6** (↑4.6) |
| **+ FlashVLA** (`d=2`) | **91.0** | **90.2** | **90.6** (↑4.6) |

### Cross-Architecture Generalization

| Backbone | Method | Avg SR (%) | Time/Step (ms) | Inference Latency (ms) |
|:--|:--|:--:|:--:|:--:|
| SmolVLA | Baseline | 80.1 | 41.2 | 19.7 |
| SmolVLA | **+ FlashVLA** (`d=0`) | **80.1** (↑0.0) | **29.7** (1.39×) | **10.1** (1.95×) |
| SmolVLA | **+ FlashVLA** (`d=1`) | 79.5 (↓0.6) | **28.7** (1.44×) | – |
| LingBot-VLA | Baseline | 85.8 | 46.7 | 70.6 |
| LingBot-VLA | **+ FlashVLA** (`d=0`) | **88.6** (↑2.8) | **43.5** (1.07×) | **25.1** (2.81×) |
| LingBot-VLA | **+ FlashVLA** (`d=1`) | **89.3** (↑3.5) | **40.5** (1.15×) | – |

Inference latency is independent of `d` and is reported once at `d=0`.

## Latency Benchmark

Per-action inference latency is measured with `benchmarks/benchmark_latency.py`:

```bash
python benchmarks/benchmark_latency.py --config_path=benchmarks/configs/latency_flashvla.yaml
```

Use `--num_views=1`, `--num_views=2`, or `--num_views=3` to sweep camera views.

Inference latency (ms) on RTX 4090 / 5090 with two and three camera views,
averaged over 100 samples after 10 warm-up iterations. π0.5 and FlashVLA use
the same CUDA Graph and kernel-fusion optimizations.

| Method | RTX 4090 (2 views) | RTX 4090 (3 views) | RTX 5090 (2 views) | RTX 5090 (3 views) |
|:--|:--:|:--:|:--:|:--:|
| π0.5 | 45.8 | 55.4 | 37.0 | 44.8 |
| + Realtime-VLA | 29.2 | 38.9 | 26.6 | 34.2 |
| **+ FlashVLA** | **26.7** | **36.8** | **20.3** | **27.1** |

## Training

**LIBERO:**

```bash
bash train/train.sh train/configs/pi05/libero/pi05_flashvla.yaml
```

**RoboTwin:**

```bash
bash train/train.sh train/configs/pi05/robotwin/pi05_flashvla.yaml
```

## Acknowledgement

This project builds on [LeRobot](https://github.com/huggingface/lerobot) and
[VLASH](https://github.com/mit-han-lab/vlash).

## Citation

```bibtex
@article{flashvla2026,
  title   = {{FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference}},
  author  = {Li, Zekai and Tang, Jiaming and Liu, Zhijian},
  year    = {2026}
}
```

## License

Apache-2.0
