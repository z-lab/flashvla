#!/usr/bin/env python
# Copyright 2025 FlashVLA team. All rights reserved.
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

import json
import logging
from pathlib import Path

import numpy as np
import packaging.version


def patched_check_version(repo_id, version_to_check, current_version, enforce_breaking_major=True):
    """Allow v2.x datasets with warning instead of error.
    
    The original function raises an error for major version mismatches.
    This patch logs a warning instead, enabling v2.1 dataset loading.
    """
    v_check = packaging.version.parse(version_to_check) if isinstance(version_to_check, str) else version_to_check
    v_current = packaging.version.parse(current_version) if isinstance(current_version, str) else current_version
    if v_check.major < v_current.major:
        logging.warning(f"Dataset {repo_id} is v{v_check}, loading with v2.1 compatibility.")


def patched_get_safe_version(repo_id: str, revision: str | None) -> str:
    """Get dataset version, allowing v2.x datasets without error.
    
    The original function raises BackwardCompatibilityError for v2.x datasets.
    This patch returns the actual version instead, enabling loading.
    """
    from huggingface_hub import HfApi
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
    
    api = HfApi()
    dataset_info = api.list_repo_refs(repo_id, repo_type="dataset")
    
    versions = [tag.name for tag in dataset_info.tags if tag.name.startswith("v")]
    
    if not versions:
        return revision or "main"
    
    parsed_versions = []
    for v in versions:
        try:
            parsed_versions.append((packaging.version.parse(v), v))
        except Exception:
            continue
    
    if not parsed_versions:
        return revision or "main"
    
    parsed_versions.sort(key=lambda x: x[0], reverse=True)
    
    if revision:
        for _pv, v in parsed_versions:
            if v == revision:
                logging.warning(f"Dataset {repo_id} is {v}, loading with v2.1 compatibility.")
                return v
    
    latest_parsed, latest_tag = parsed_versions[0]
    codebase_parsed = packaging.version.parse(CODEBASE_VERSION)
    
    if latest_parsed.major < codebase_parsed.major:
        logging.warning(f"Dataset {repo_id} is {latest_tag}, loading with v2.1 compatibility.")
    
    return latest_tag


def patched_load_episodes(local_dir: Path):
    """Load episodes from jsonl (v2.1) or parquet (v3.0).
    
    For v2.1 format:
    - Reads meta/episodes.jsonl
    - Adds v3.0-style fields: dataset indices, chunk indices, timestamps
    
    Returns:
        HuggingFace Dataset with episode metadata.
    """
    import datasets
    from lerobot.datasets.utils import LEGACY_EPISODES_PATH, EPISODES_DIR, load_nested_dataset, load_info
    
    jsonl_path = local_dir / LEGACY_EPISODES_PATH
    
    if not jsonl_path.exists():
        episodes = load_nested_dataset(local_dir / EPISODES_DIR)
        return episodes.select_columns([k for k in episodes.features if not k.startswith("stats/")])
    
    with open(jsonl_path) as f:
        episodes_list = sorted([json.loads(line) for line in f], key=lambda x: x["episode_index"])
    
    info = load_info(local_dir)
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    chunks_size = info.get("chunks_size", 1000)
    fps = info.get("fps", 30)
    
    cumulative = 0
    for ep in episodes_list:
        ep_idx = ep["episode_index"]
        ep_chunk = ep_idx // chunks_size
        
        ep["dataset_from_index"] = cumulative
        cumulative += ep["length"]
        ep["dataset_to_index"] = cumulative
        
        ep["data/chunk_index"] = ep_chunk
        ep["data/file_index"] = ep_idx
        
        for vid_key in video_keys:
            ep[f"videos/{vid_key}/chunk_index"] = ep_chunk
            ep[f"videos/{vid_key}/file_index"] = ep_idx
            ep[f"videos/{vid_key}/from_timestamp"] = 0.0
            ep[f"videos/{vid_key}/to_timestamp"] = ep["length"] / fps
    
    return datasets.Dataset.from_list(episodes_list)


def aggregate_stats(stats_list: list[dict]) -> dict:
    """Aggregate per-episode stats into global stats.
    
    Uses the parallel variance algorithm:
        σ²_total = Σ(σ²_i + δ²_i) * n_i / N
    
    where δ_i = μ_i - μ_total is the difference between episode mean
    and global mean.
    
    Args:
        stats_list: List of per-episode stats dictionaries.
        
    Returns:
        Aggregated global stats with min, max, mean, std, count.
    """
    def aggregate_feature(stats_ft_list):
        means = np.stack([s["mean"] for s in stats_ft_list])
        variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
        counts = np.stack([s["count"] for s in stats_ft_list])
        total_count = counts.sum(axis=0)
        
        while counts.ndim < means.ndim:
            counts = np.expand_dims(counts, axis=-1)
        
        total_mean = (means * counts).sum(axis=0) / total_count
        
        delta_means = means - total_mean
        total_variance = ((variances + delta_means ** 2) * counts).sum(axis=0) / total_count
        
        return {
            "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
            "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
            "mean": total_mean,
            "std": np.sqrt(total_variance),
            "count": total_count,
        }
    
    data_keys = {key for stats in stats_list for key in stats}
    return {key: aggregate_feature([s[key] for s in stats_list if key in s]) for key in data_keys}


def patched_load_stats(local_dir: Path):
    """Load stats from stats.json (v3.0) or episodes_stats.jsonl (v2.1).
    
    For v2.1 format:
    - Reads per-episode stats from meta/episodes_stats.jsonl
    - Aggregates into global stats using parallel variance algorithm
    
    Returns:
        Dictionary of feature stats (min, max, mean, std, count).
    """
    from lerobot.datasets.io_utils import STATS_PATH, load_json, cast_stats_to_numpy
    
    if (local_dir / STATS_PATH).exists():
        return cast_stats_to_numpy(load_json(local_dir / STATS_PATH))
    
    episodes_stats_path = local_dir / "meta/episodes_stats.jsonl"
    if not episodes_stats_path.exists():
        return None
    
    with open(episodes_stats_path) as f:
        episodes_stats = [json.loads(line) for line in f]
    
    if not episodes_stats:
        return None
    
    stats_list = [
        {k: {sk: np.array(sv) for sk, sv in v.items()} for k, v in ep["stats"].items()}
        for ep in episodes_stats
    ]
    return aggregate_stats(stats_list)


def patched_load_tasks(local_dir: Path):
    """Load tasks from jsonl (v2.1) or parquet (v3.0).
    
    Returns:
        DataFrame with task_index column and task string as index.
    """
    import pandas as pd
    from lerobot.datasets.utils import LEGACY_TASKS_PATH, DEFAULT_TASKS_PATH
    
    jsonl_path = local_dir / LEGACY_TASKS_PATH
    
    if not jsonl_path.exists():
        return pd.read_parquet(local_dir / DEFAULT_TASKS_PATH)
    
    with open(jsonl_path) as f:
        tasks_list = sorted([json.loads(line) for line in f], key=lambda x: x["task_index"])
    return pd.DataFrame({"task_index": [t["task_index"] for t in tasks_list]}, index=[t["task"] for t in tasks_list])


def is_v21_format(info: dict) -> bool:
    """Check if dataset uses v2.1 path format.
    
    v2.1 uses {episode_chunk} and {episode_index} in path templates.
    """
    video_path = info.get("video_path", "")
    return "episode_chunk" in video_path or "episode_index" in video_path


def make_patched_path_method(original_method, for_video=False):
    """Create patched get_data_file_path or get_video_file_path for v2.1.
    
    Handles v2.1 path format strings with {episode_chunk} and {episode_index}.
    """
    def patched(self, ep_index, vid_key=None):
        if not is_v21_format(self.info):
            return original_method(self, ep_index, vid_key) if for_video else original_method(self, ep_index)
        
        ep_chunk = ep_index // self.info.get("chunks_size", 1000)
        fmt_args = {"episode_chunk": ep_chunk, "episode_index": ep_index}
        if for_video:
            fmt_args["video_key"] = vid_key
        
        path_template = self.video_path if for_video else self.data_path
        return Path(path_template.format(**fmt_args))
    return patched


def apply_patches():
    """Apply all compatibility patches to lerobot.
    
    Patches:
    - get_safe_version: Allow v2.x without BackwardCompatibilityError
    - check_version_compatibility: Allow v2.x with warning
    - load_episodes: Support jsonl format
    - load_tasks: Support jsonl format
    - load_stats: Support per-episode stats aggregation
    - get_data_file_path: Support v2.1 path format
    - get_video_file_path: Support v2.1 path format
    """
    import lerobot.datasets.utils as utils_module
    import lerobot.datasets.lerobot_dataset as dataset_module
    
    utils_module.get_safe_version = patched_get_safe_version
    dataset_module.get_safe_version = patched_get_safe_version
    
    
    for module in [utils_module, dataset_module]:
        module.check_version_compatibility = patched_check_version
        module.load_episodes = patched_load_episodes
        module.load_tasks = patched_load_tasks
        module.load_stats = patched_load_stats
    
    meta_cls = dataset_module.LeRobotDatasetMetadata
    meta_cls.get_data_file_path = make_patched_path_method(meta_cls.get_data_file_path, for_video=False)
    meta_cls.get_video_file_path = make_patched_path_method(meta_cls.get_video_file_path, for_video=True)
    
    logging.info("FlashVLA: Applied v2.1 compatibility patches to lerobot")


apply_patches()
