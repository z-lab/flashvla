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
  - robotwin
language:
  - en
---

# FlashVLA · π0.5 · RoboTwin 2.0

A **π0.5** flow-matching vision-language-action policy finetuned on **RoboTwin 2.0**
(50-task multitask) and served with [**FlashVLA**](https://github.com/z-lab/flashvla)
streaming action decoding for fast, asynchronous inference.

- **Base model:** [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base)
- **Method:** [FlashVLA](https://github.com/z-lab/flashvla) — streaming action decoding for flow-matching VLAs (async chunk-overlap execution)
- **Benchmark:** RoboTwin 2.0, 50-task multitask (clean / randomized)
- **License:** Apache-2.0

## Results

RoboTwin 2.0 50-task multitask success rate (%). `d` is the async step delay:
`d=0` is synchronous, `d=1`/`d=2` overlap the next chunk's inference with execution.

| Model | Clean | Random | Avg |
|:--|:--:|:--:|:--:|
| π0.5 (base) | 82.74 | 76.76 | 79.75 |
| **This model** (`d=0`) | 90.64 | 90.06 | 90.35 |
| **This model** (`d=1`) | **91.14** | **90.60** | **90.87** |
| **This model** (`d=2`) | 90.20 | 89.66 | 89.93 |

FlashVLA improves the clean/randomized average over the π0.5 base by **~10 points**, and
holds up under asynchronous execution (`d=1`/`d=2`).

## Usage

Install the [FlashVLA](https://github.com/z-lab/flashvla) library.

See [`sim_eval/robotwin/`](https://github.com/z-lab/flashvla/tree/main/sim_eval/robotwin) for evaluation setup.

## Training

- **Data:** RoboTwin 2.0 ([`TianxingChen/RoboTwin2.0`](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)), aloha-agilex `clean_50` + `randomized_500`, converted to LeRobot v3 and pooled into a 50-task multitask set
- **Init:** finetuned from `lerobot/pi05_base`
- **Schedule:** 50k steps · AdamW (lr 3e-5, cosine decay, 2k warmup) · bf16
- **Action streaming:** `chunk_size=20`, `num_buffer_slots=4`, `cold_start_mode=current_state` (absolute-qpos control)
- **Normalization:** QUANTILES (state / action)
- **Parallelism:** FSDP2 bf16 mixed precision

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
