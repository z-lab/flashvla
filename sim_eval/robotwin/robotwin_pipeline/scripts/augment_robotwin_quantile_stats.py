#!/usr/bin/env python

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
    parser = argparse.ArgumentParser()
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
