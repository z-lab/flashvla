#!/usr/bin/env python
"""Augment RoboTwin-LeRobot-v3.0 sub-datasets with quantile stats (q01..q99).

The HF release of hxma/RoboTwin-LeRobot-v3.0 ships meta/stats.json with only
min/max/mean/std/count. Our pi05 configs use QUANTILES normalization for
STATE and ACTION, and lerobot's NormalizerProcessorStep hard-fails when
q01/q99 are missing.

lerobot's own scripts/augment_dataset_quantile_stats.py is unsuitable here:
it decodes every video frame (hours over 77 GB) and ends with
dataset.push_to_hub(). VISUAL normalization is IDENTITY in our configs, so
image quantiles are never read — we only need quantiles for the
parquet-backed float features (action, observation.state), which we can
compute directly from the data parquets in seconds per subset.

Quantiles are computed globally over all frames of a subset (exact), rather
than lerobot's per-episode-then-count-weighted-average (approximate). Both
are compatible with aggregate_stats() pooling at multitask training time.

Usage:
    python scripts/augment_robotwin_quantile_stats.py \
        --root /path/to/RoboTwin-LeRobot-v3.0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]
FEATURES = ["action", "observation.state"]


def augment_subset(subset_root: Path, overwrite: bool) -> str:
    stats_path = subset_root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())

    q_keys = [f"q{int(q * 100):02d}" for q in QUANTILES]
    if not overwrite and all(
        all(k in stats.get(ft, {}) for k in q_keys) for ft in FEATURES
    ):
        return "skip (already augmented)"

    parquets = sorted(subset_root.glob("data/chunk-*/file-*.parquet"))
    if not parquets:
        return "ERROR: no data parquets found"

    df = pd.concat([pd.read_parquet(p, columns=FEATURES) for p in parquets])
    for ft in FEATURES:
        values = np.stack(df[ft].to_numpy())
        qs = np.quantile(values, QUANTILES, axis=0)
        for q, q_key in zip(qs, q_keys):
            stats[ft][q_key] = q.tolist()

    stats_path.write_text(json.dumps(stats, indent=4))
    return f"augmented ({len(df)} frames)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="RoboTwin-LeRobot-v3.0 root: <root>/<task>/<config_subdir>/")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if quantile keys already exist")
    args = parser.parse_args()

    subsets = sorted(p.parent.parent for p in args.root.glob("*/*/meta/stats.json"))
    if not subsets:
        raise SystemExit(f"No subsets found under {args.root}")

    for subset in subsets:
        result = augment_subset(subset, args.overwrite)
        print(f"{subset.relative_to(args.root)}: {result}")

    print(f"\nDone: {len(subsets)} subsets")


if __name__ == "__main__":
    main()
