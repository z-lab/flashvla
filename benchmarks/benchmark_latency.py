#!/usr/bin/env python
# Copyright 2025 FlashVLA team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import json
import logging
from pathlib import Path
from pprint import pformat
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.configs import parser
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.random_utils import set_seed
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging
from lerobot.utils.constants import OBS_PREFIX

from benchmark_config import BenchmarkConfig

import flashvla.configs  # noqa: F401 — registers PI05 configs with draccus

logger = logging.getLogger(__name__)

VALID_VARIANTS = ["baseline", "flashvla"]




def _predict_accepts_profile(policy) -> bool:
    """True iff ``policy.predict_action_chunk`` accepts a ``profile`` kwarg.

    Keeps the harness robust across policies that have not been instrumented
    with the in-model encode/prefill/action breakdown: those fall back to
    total-latency-only timing instead of raising a TypeError. A ``**kwargs``
    sink could swallow ``profile`` without honoring it, so only an explicit
    parameter counts as support.
    """
    try:
        sig = inspect.signature(policy.predict_action_chunk)
    except (TypeError, ValueError):
        return False
    return "profile" in sig.parameters


def compute_latency_stats(latencies: list[float]) -> dict:
    """Compute summary statistics from a list of latency measurements (ms)."""
    arr = np.array(latencies)
    return {
        "num_samples": len(arr),
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "fps": float(1000.0 / np.mean(arr)) if np.mean(arr) > 0 else 0.0,
    }


def compute_breakdown_stats(breakdowns: list[dict]) -> dict:
    """Compute per-stage mean/median/std from a list of breakdown dicts."""
    if not breakdowns:
        return {}
    all_stages: list[str] = []
    for bd in breakdowns:
        for k in bd:
            if k not in all_stages:
                all_stages.append(k)

    stats = {}
    for stage in all_stages:
        values = [bd[stage] for bd in breakdowns if stage in bd]
        if values:
            arr = np.array(values)
            stats[stage] = {
                "mean_ms": float(np.mean(arr)),
                "median_ms": float(np.median(arr)),
                "std_ms": float(np.std(arr)),
            }
    return stats



def load_dataset(cfg: BenchmarkConfig) -> tuple[LeRobotDataset, LeRobotDatasetMetadata]:
    """Load dataset for benchmarking.

    Sets up delta_timestamps so each image observation includes
    ``cfg.policy.num_frames_per_view`` history frames per view (default 1).
    View count is filtered later via ``cfg.num_views`` in ``_setup_features``.
    """
    logger.info(f"Loading dataset: {cfg.dataset.repo_id}")
    repo_id = cfg.dataset.repo_id
    revision = cfg.dataset.revision

    try:
        ds_meta = LeRobotDatasetMetadata(
            repo_id, root=cfg.dataset.root, revision=revision
        )
    except RevisionNotFoundError:
        fallback_revision = "main"
        logger.warning(
            "Revision '%s' not found for dataset '%s'; retrying with '%s'",
            revision or "v3.0", repo_id, fallback_revision,
        )
        ds_meta = LeRobotDatasetMetadata(
            repo_id, root=cfg.dataset.root, revision=fallback_revision
        )
        cfg.dataset.revision = fallback_revision

    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    n_frames = getattr(cfg.policy, "num_frames_per_view", 1)
    if n_frames > 1:
        for key in ds_meta.features:
            if key.startswith(OBS_PREFIX + "images."):
                delta_timestamps[key] = [i / ds_meta.fps for i in range(n_frames)]

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=cfg.dataset.root,
        delta_timestamps=delta_timestamps,
        revision=cfg.dataset.revision,
    )

    logger.info(f"Dataset loaded: {len(dataset)} samples, {dataset.num_episodes} episodes")
    return dataset, ds_meta



def _setup_features(cfg, ds_meta):
    """Common feature setup for policy loading.

    If ``cfg.num_views`` is set, restricts the policy's image inputs to the
    first ``num_views`` ``observation.images.*`` keys in dataset feature
    order; non-image keys (state, etc.) are kept. Lets the user sweep view
    count without rewriting yaml.
    """
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.feature_utils import dataset_to_policy_features

    features = dataset_to_policy_features(ds_meta.features)
    if not cfg.policy.output_features:
        cfg.policy.output_features = {
            key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
        }
    if not cfg.policy.input_features:
        cfg.policy.input_features = {
            key: ft for key, ft in features.items() if key not in cfg.policy.output_features
        }

    num_views = getattr(cfg, "num_views", None)
    if num_views is not None and num_views > 0:
        image_keys = [
            k for k in cfg.policy.input_features
            if k.startswith(OBS_PREFIX + "images.")
        ]
        non_image_keys = [
            k for k in cfg.policy.input_features
            if not k.startswith(OBS_PREFIX + "images.")
        ]
        if num_views > len(image_keys):
            logger.warning(
                f"num_views={num_views} but dataset only has {len(image_keys)} "
                f"image keys: {image_keys}. Capping to {len(image_keys)}."
            )
        kept = image_keys[:num_views]
        dropped = image_keys[num_views:]
        if dropped:
            logger.info(
                f"num_views={num_views}: keeping {kept}, "
                f"dropping {dropped}"
            )
        cfg.policy.input_features = {
            **{k: cfg.policy.input_features[k] for k in kept},
            **{k: cfg.policy.input_features[k] for k in non_image_keys},
        }


def load_policy(variant: str, cfg: BenchmarkConfig, ds_meta: LeRobotDatasetMetadata) -> PreTrainedPolicy:
    """Load the appropriate policy for the given variant."""
    _setup_features(cfg, ds_meta)

    if hasattr(cfg.policy, "tokenizer_max_length"):
        if cfg.policy.tokenizer_max_length != 20:
            logger.info(
                f"Overriding policy.tokenizer_max_length: "
                f"{cfg.policy.tokenizer_max_length} -> 20 (latency-bench short-lang)"
            )
            cfg.policy.tokenizer_max_length = 20

    n_views = sum(
        1 for k in cfg.policy.input_features
        if k.startswith(OBS_PREFIX + "images.")
    )
    n_frames = getattr(cfg.policy, "num_frames_per_view", 1)
    logger.info(
        f"Variant={variant} | views={n_views} | frames_per_view={n_frames} | "
        f"image inputs: {[k for k in cfg.policy.input_features if 'images.' in k]}"
    )

    # Dispatch on the policy's own ``type`` through the shared factory so every
    # registered backbone loads by one path. ``variant`` stays an output label
    # and a cross-check: a flashvla benchmark must be paired with a
    # ``-flashvla`` policy type (the rolling-buffer 1-step path), and a baseline
    # benchmark must not be.
    from flashvla.policies.factory import get_policy_class

    policy_type = cfg.policy.type
    is_streaming_type = policy_type.endswith("-flashvla")
    if variant == "flashvla" and not is_streaming_type:
        raise ValueError(
            f"variant='flashvla' requires a '-flashvla' policy type "
            f"(rolling-buffer 1-step decode); got policy.type='{policy_type}'."
        )
    if variant == "baseline" and is_streaming_type:
        logger.warning(
            "variant='baseline' paired with streaming policy.type='%s'; the "
            "rolling-buffer 1-step path will be exercised, not a full-denoise "
            "baseline.", policy_type,
        )
    if variant not in ("baseline", "flashvla"):
        raise ValueError(f"Unknown variant: {variant}")

    policy_cls = get_policy_class(policy_type)
    logger.info(f"Loading policy class: {policy_cls.__name__} (type={policy_type})")

    policy = policy_cls.from_pretrained(
        pretrained_name_or_path=cfg.policy.pretrained_path,
        config=cfg.policy,
        dataset_stats=ds_meta.stats,
    )
    policy.to(cfg.policy.device)
    policy.eval()

    logger.info(f"Policy loaded on device: {cfg.policy.device}")
    return policy



def prepare_batch_dataloader(batch: dict, device: torch.device, preprocessor=None) -> dict:
    """Move a DataLoader batch to device + run policy preprocessor.

    The preprocessor handles language tokenization (string → token ids /
    attention mask) so the model gets ``OBS_LANGUAGE_TOKENS`` etc.
    """
    prepared = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            prepared[k] = v.to(device)
        else:
            prepared[k] = v
    if "language_instruction" in batch:
        prepared["task"] = batch["language_instruction"]
    if preprocessor is not None:
        prepared = preprocessor(prepared)
    return prepared



def warmup_non_streaming(
    policy: PreTrainedPolicy,
    dataloader: DataLoader,
    cfg: BenchmarkConfig,
    preprocessor=None,
):
    """Warm up model before benchmarking."""
    if cfg.warmup_steps <= 0:
        return

    logger.info(f"Warming up model for {cfg.warmup_steps} steps...")
    device = get_safe_torch_device(cfg.policy.device)

    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            if i >= cfg.warmup_steps:
                break
            batch = prepare_batch_dataloader(batch, device, preprocessor=preprocessor)
            _ = policy.predict_action_chunk(batch)

    if device.type == "cuda":
        torch.cuda.synchronize()
    logger.info("Warmup complete")


def benchmark_non_streaming(
    policy: PreTrainedPolicy,
    dataloader: DataLoader,
    cfg: BenchmarkConfig,
    preprocessor=None,
) -> tuple[dict, dict | None]:
    """Run non-streaming benchmark. Returns (overall_stats, breakdown_stats).

    When ``cfg.policy.compile_model`` is True we skip the in-model ``profile=True``
    path: it inserts cuda.Event ops that force dynamo graph breaks and a second
    specialization of the compiled graph, which on weaker GPUs (e.g. 3090) shows
    up as catastrophic per-call latency. In compile mode we only report the
    overall total from external perf_counter timing.
    """
    device = get_safe_torch_device(cfg.policy.device)
    latencies: list[float] = []
    breakdowns: list[dict] = []

    # The per-stage breakdown needs both (a) an un-compiled model — cuda.Event
    # ops force dynamo graph breaks / a second specialization that wrecks
    # latency on weaker GPUs — and (b) a policy whose predict_action_chunk
    # actually accepts a ``profile`` kwarg. Either missing → total latency only.
    profile_supported = _predict_accepts_profile(policy)
    use_internal_breakdown = (not cfg.policy.compile_model) and profile_supported
    if cfg.policy.compile_model:
        logger.info(
            "compile_model=True → measuring total latency only "
            "(per-stage breakdown disabled to keep compiled graph specialized for profile=False)"
        )
    elif not profile_supported:
        logger.info(
            "policy.predict_action_chunk does not accept a 'profile' kwarg → "
            "measuring total latency only (no encode/prefill/action breakdown)."
        )

    logger.info(f"Starting non-streaming benchmark with {cfg.num_samples} samples...")

    cold_start_skipped = 0
    with torch.inference_mode():
        for batch in dataloader:
            if len(latencies) >= cfg.num_samples:
                break

            batch = prepare_batch_dataloader(batch, device, preprocessor=preprocessor)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = perf_counter()

            if use_internal_breakdown:
                actions, profile_results = policy.predict_action_chunk(batch, profile=True)
            else:
                actions = policy.predict_action_chunk(batch)
                profile_results = None

            if device.type == "cuda":
                torch.cuda.synchronize()
            latency = (perf_counter() - start_time) * 1000

            if actions is None:
                cold_start_skipped += 1
                continue

            latencies.append(latency)

            if profile_results:
                breakdowns.append({k: v for k, v in profile_results.items() if k != "total"})

            if len(latencies) % 10 == 0:
                logger.info(f"Processed {len(latencies)}/{cfg.num_samples} samples...")

    if cold_start_skipped:
        logger.info(
            f"Skipped {cold_start_skipped} cold-start step(s) "
            f"(predict_action_chunk returned None — buffer not yet full)"
        )

    stats = compute_latency_stats(latencies)
    bd_stats = compute_breakdown_stats(breakdowns) if breakdowns else None
    return stats, bd_stats



def print_results(variant: str, results: dict):
    """Print a uniform results table."""
    overall = results["overall"]

    print("\n" + "=" * 80)
    print(f"LATENCY BENCHMARK: {variant}")
    print("=" * 80)
    print(f"  Samples:  {overall['num_samples']}")
    print(f"  Mean:     {overall['mean_ms']:.1f} ms  ({overall['fps']:.1f} FPS)")
    print(f"  Median:   {overall['median_ms']:.1f} ms")
    print(f"  P95:      {overall['p95_ms']:.1f} ms")

    bd = results.get("breakdown")
    if bd:
        total = sum(s["mean_ms"] for s in bd.values())
        print(f"\n  Breakdown (mean ms):")
        for stage, s in bd.items():
            pct = s["mean_ms"] / total * 100 if total > 0 else 0
            print(f"    {stage:15s}  {s['mean_ms']:7.1f} ms  ({pct:5.1f}%)")
        print(f"    {'TOTAL':15s}  {total:7.1f} ms")

    print("=" * 80 + "\n")



def save_results(results: dict, cfg: BenchmarkConfig, variant: str):
    """Save benchmark results to JSON file."""
    if cfg.output_file is None:
        return

    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pretrained_path = getattr(cfg.policy, "pretrained_path", None)
    n_views = sum(
        1 for k in cfg.policy.input_features
        if k.startswith(OBS_PREFIX + "images.")
    )

    output_data = {
        "config": {
            "variant": variant,
            "policy_path": str(pretrained_path) if pretrained_path else None,
            "dataset_repo_id": cfg.dataset.repo_id,
            "device": cfg.policy.device,
            "compile_model": cfg.policy.compile_model,
            "num_frames_per_view": getattr(cfg.policy, "num_frames_per_view", 1),
            "num_views": n_views,
            "num_samples": cfg.num_samples,
            "warmup_steps": cfg.warmup_steps,
            "seed": cfg.seed,
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to: {output_path}")



@parser.wrap()
def benchmark_latency(cfg: BenchmarkConfig):
    """Unified latency benchmark entry point."""
    init_logging()

    variant = cfg.variant
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Invalid variant '{variant}'. Must be one of: {VALID_VARIANTS}")

    logger.info(f"Starting latency benchmark: variant={variant}")
    cfg.validate()
    logger.info(pformat(cfg.to_dict()))

    set_seed(cfg.seed)

    # pi0-family policies were trained with a 4th (empty) camera slot and expect
    # one at num_views==3. LingBot uses exactly 3 real cameras (high /
    # left_wrist / right_wrist) and must keep empty_cameras=0 — a padding camera
    # would corrupt its prefix layout.
    if cfg.num_views == 3 and cfg.policy.type.startswith("pi0"):
        cfg.policy.empty_cameras = 1

    dataset, ds_meta = load_dataset(cfg)
    policy = load_policy(variant, cfg, ds_meta)

    from flashvla.policies.factory import make_pre_post_processors
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=ds_meta.stats,
        preprocessor_overrides={"device_processor": {"device": cfg.policy.device}},
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.policy.device == "cuda",
    )

    warmup_non_streaming(policy, dataloader, cfg, preprocessor=preprocessor)

    overall, bd_stats = benchmark_non_streaming(
        policy, dataloader, cfg, preprocessor=preprocessor,
    )

    results = {
        "overall": overall,
        "breakdown": bd_stats,
    }

    print_results(variant, results)
    save_results(results, cfg, variant)

    logger.info(f"Latency benchmark ({variant}) complete!")


def main():
    benchmark_latency()


if __name__ == "__main__":
    main()
