# FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference

**Paper** | **Blog** | [**Models**](https://huggingface.co/collections/z-lab/flashvla)

**FlashVLA** is a general streaming action decoding method for flow-matching VLA models, achieving fast and asynchronous execution. The repository includes π0.5, π0, SmolVLA, and LingBot-VLA (Qwen2.5-VL) backbones.

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

π0.5 vs. **+ FlashVLA** (**bold** = the better result). `d` is the async step delay:
`d=0` is synchronous, `d≥1` overlaps the next chunk's inference with robot execution.

### LIBERO

Success rate (%) + per-step latency (averaged over 2,000 episodes):

| Method | Spatial | Object | Goal | Long | Avg | Time/Step (ms) |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| π0.5 | **98.8** | 98.2 | **98.0** | 92.4 | 96.9 | 53.8 |
| **+ FlashVLA** (`d=0`) | 98.6 | 99.0 | 97.8 | **96.2** | **97.9** (↑1.0) | – |
| **+ FlashVLA** (`d=1`) | 96.0 | **99.6** | 96.4 | 96.0 | 97.0 (↑0.1) | **29.4** (1.83×) |

### RoboTwin 2.0

50-task multitask success rate (%) — `d=0` synchronous, `d=1`/`d=2` asynchronous:

| Method | Clean | Random | Avg |
|:--|:--:|:--:|:--:|
| π0.5 | 86.58 | 86.06 | 86.32 |
| **+ FlashVLA** (`d=0`) | 90.64 (↑4.06) | 90.06 (↑4.00) | 90.35 (↑4.03) |
| **+ FlashVLA** (`d=1`) | **91.14** (↑4.56) | **90.60** (↑4.54) | **90.87** (↑4.55) |
| **+ FlashVLA** (`d=2`) | 90.20 (↑3.62) | 89.66 (↑3.60) | 89.93 (↑3.61) |

## Latency Benchmark

Per-action inference latency is measured with `benchmarks/benchmark_latency.py`:

```bash
python benchmarks/benchmark_latency.py --config_path=benchmarks/configs/latency_flashvla.yaml
```

The same harness also benchmarks LingBot-VLA on RoboTwin:

| Backbone | Baseline config | FlashVLA config |
|:--|:--|:--|
| π0.5 | `benchmarks/configs/latency_baseline.yaml` | `benchmarks/configs/latency_flashvla.yaml` |
| LingBot-VLA | `benchmarks/configs/latency_lingbot_baseline.yaml` | `benchmarks/configs/latency_lingbot_flashvla.yaml` |

Use `--num_views=1`, `--num_views=2`, or `--num_views=3` to sweep camera views.

Per-action latency (ms) on RTX 4090 / 5090 with 2 and 3 camera views (π0.5 uses PyTorch max-autotune):

| Method | RTX 4090 (2 views) | RTX 4090 (3 views) | RTX 5090 (2 views) | RTX 5090 (3 views) |
|:--|:--:|:--:|:--:|:--:|
| π0.5 | 46.1 | 56.1 | 37.5 | 45.4 |
| **+ FlashVLA** | **26.7** | **36.8** | **20.3** | **27.1** |

## Training

`train/train.sh <config> [num_gpus]` launches a training recipe (8 GPUs by
default):

```bash
bash train/train.sh train/configs/pi05/libero/pi05_flashvla.yaml
```

| Backbone | Benchmark | Config |
|:--|:--|:--|
| π0.5 | LIBERO | `train/configs/pi05/libero/pi05_flashvla.yaml` |
| π0.5 | RoboTwin 2.0 | `train/configs/pi05/robotwin/pi05_flashvla.yaml` |
| π0 | LIBERO | `train/configs/pi0/libero/pi0_flashvla.yaml` |
| SmolVLA | LIBERO | `train/configs/smolvla/libero/smolvla_flashvla.yaml` |
| LingBot-VLA | RoboTwin 2.0 | `train/configs/lingbot/robotwin/lingbot_vla_baseline.yaml` |
| LingBot-VLA + FlashVLA | RoboTwin 2.0 | `train/configs/lingbot/robotwin/lingbot_flashvla.yaml` |

The RoboTwin recipes require the LeRobot-format dataset described in
[Building the training dataset](sim_eval/robotwin/README.md#building-the-training-dataset).
See [the LingBot integration notes](docs/lingbot_vla_flashvla.md) for its
checkpoint migration and typed-action layout.

## Acknowledgement

This project builds on [LeRobot](https://github.com/huggingface/lerobot),
[VLASH](https://github.com/mit-han-lab/vlash), and
[LingBot-VLA](https://github.com/Robbyant/lingbot-vla).

LingBot-VLA support is built with Qwen. Qwen model materials are subject to
the separate Qwen Research License; see
[`NOTICE.md`](NOTICE.md).

## Citation

```bibtex
@article{flashvla2026,
  title   = {{FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference}},
  author  = {Li, Zekai and Tang, Jiaming and Liu, Zhijian},
  year    = {2026}
}
```

## License

FlashVLA source code is Apache-2.0. Third-party model weights, tokenizers,
datasets, and simulator assets retain their own licenses; see
[`NOTICE.md`](NOTICE.md).
