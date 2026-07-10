---
license: apache-2.0
base_model: lerobot/pi05_base
pipeline_tag: robotics
tags:
  - robotics
  - vla
  - vision-language-action
  - flow-matching
  - flashvla
  - pi0.5
  - lerobot
datasets:
  - HuggingfaceVLA/libero
language:
  - en
---

# FlashVLA · π0.5 · LIBERO

A **π0.5** flow-matching vision-language-action policy finetuned on **LIBERO** and served
with [**FlashVLA**](https://github.com/z-lab/flashvla) streaming action decoding for fast,
asynchronous inference.

- **Base model:** [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base)
- **Method:** [FlashVLA](https://github.com/z-lab/flashvla) — streaming action decoding for flow-matching VLAs (async chunk-overlap execution)
- **Benchmark:** LIBERO (Spatial / Object / Goal / Long)
- **License:** Apache-2.0

## Results

LIBERO success rate (%) and per-step latency. `d` is the async step delay:
`d=0` is synchronous, `d≥1` overlaps the next chunk's inference with robot execution.

| Model | Spatial | Object | Goal | Long | Avg | Time/Step (ms) |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| π0.5 (base) | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 | 53.8 |
| **This model** (`d=0`) | 98.6 | 99.0 | 97.8 | 96.2 | **97.9** | — |
| **This model** (`d=1`) | 96.0 | 99.6 | 96.4 | 96.0 | 97.0 | **29.4** (1.83× faster) |

FlashVLA raises the average success rate over the π0.5 base while cutting per-step latency
by up to **1.83×** through asynchronous chunk overlap.

## Usage

Install the [FlashVLA](https://github.com/z-lab/flashvla) library.

See [`sim_eval/libero/`](https://github.com/z-lab/flashvla/tree/main/sim_eval/libero) for evaluation setup.

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
