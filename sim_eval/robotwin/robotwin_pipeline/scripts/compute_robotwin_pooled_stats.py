#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]
FEATURES = ["action", "observation.state"]


def read_subset(subset_root: Path) -> dict[str, np.ndarray]:
    parquets = sorted(subset_root.glob("data/chunk-*/file-*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No data parquets under {subset_root}")
    df = pd.concat([pd.read_parquet(p, columns=FEATURES) for p in parquets])
    return {ft: np.stack(df[ft].to_numpy()).astype(np.float32) for ft in FEATURES}


def exact_stats(values: np.ndarray) -> dict:
    q_keys = [f"q{int(q * 100):02d}" for q in QUANTILES]
    qs = np.quantile(values, QUANTILES, axis=0)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
        **{k: q.tolist() for k, q in zip(q_keys, qs)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subdirs", nargs="+",
                        default=["aloha-agilex_clean_50", "aloha-agilex_randomized_500"])
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Task subset; default = all tasks having every requested subdir")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.tasks:
        tasks = args.tasks
    else:
        tasks = sorted(
            p.name for p in args.root.iterdir()
            if p.is_dir() and all((p / sd / "meta" / "info.json").is_file() for sd in args.subdirs)
        )
    subsets = [args.root / t / sd for t in tasks for sd in args.subdirs]
    print(f"{len(tasks)} tasks x {len(args.subdirs)} subdirs = {len(subsets)} subsets")

    chunks: dict[str, list[np.ndarray]] = {ft: [] for ft in FEATURES}
    sub_stats: list[dict] = []
    for subset in subsets:
        data = read_subset(subset)
        for ft in FEATURES:
            chunks[ft].append(data[ft])
        sub_stats.append(json.loads((subset / "meta" / "stats.json").read_text()))

    pooled = {}
    q_keys = [f"q{int(q * 100):02d}" for q in QUANTILES]
    for ft in FEATURES:
        values = np.concatenate(chunks[ft])
        pooled[ft] = exact_stats(values)
        print(f"\n=== {ft}: {values.shape[0]} frames ===")

        counts = np.array([s[ft]["count"][0] for s in sub_stats], dtype=np.float64)
        exact_range = np.array(pooled[ft]["q99"]) - np.array(pooled[ft]["q01"])
        for qk in ("q01", "q99"):
            weighted = (
                np.stack([np.array(s[ft][qk]) for s in sub_stats]) * counts[:, None]
            ).sum(axis=0) / counts.sum()
            diff = np.abs(weighted - np.array(pooled[ft][qk]))
            rel = diff / np.maximum(exact_range, 1e-8)
            print(f"  {qk}: weighted-avg vs exact | max abs diff {diff.max():.4f} "
                  f"| max rel-to-range {100 * rel.max():.2f}% (dim {rel.argmax()})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pooled, indent=4))
    print(f"\nWrote exact pooled stats to {args.output}")


if __name__ == "__main__":
    main()
