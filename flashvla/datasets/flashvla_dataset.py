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
"""FlashVLA Dataset.

Returns all N buffer configurations per observation for shared observation
training. Each config k (k=1..N) has k real slots of ground truth actions
and (N-k) padded slots.

Buffer configs (N=5, C=10):
  Config 1: [real_slot_0, pad, pad, pad, pad]   -> 10 real + 40 padded actions
  Config 2: [real_slot_0, real_slot_1, pad, pad, pad] -> 20 real + 30 padded
  ...
  Config 5: [real_slot_0, ..., real_slot_4]       -> 50 real actions

State is the same for all configs (no async delay in v2).
"""

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data._utils.collate import default_collate

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class FlashVLADataset(LeRobotDataset):
    """Dataset that returns all N buffer configurations per observation.

    For each observation, returns N configs where config k (k=1..N) has
    k real slots of ground truth actions and (N-k) padded slots.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        num_buffer_slots: int = 5,
        chunk_size: int = 10,
        use_action_prefix: bool = False,
    ):
        """Initialize FlashVLADataset.

        Args:
            num_buffer_slots: N, number of slots in the denoising buffer.
            chunk_size: C, number of actions per slot.
            use_action_prefix: If True, prepend a clean chunk (C past actions)
                as history conditioning. Buffer becomes (N+1)*C per config.
            Other args: same as LeRobotDataset.
        """
        self.num_buffer_slots = num_buffer_slots
        self.chunk_size = chunk_size
        self.use_action_prefix = use_action_prefix
        self.total_action_horizon = num_buffer_slots * chunk_size

        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
            batch_encoding_size=batch_encoding_size,
        )

    def __getitem__(self, idx) -> dict:
        """Get sample with all N buffer configurations.

        When use_action_prefix=False (default):
            Each config k has k real noisy slots + (N-k) padded slots.
            Per-config length: H = N*C.

        When use_action_prefix=True:
            Each config k has 1 clean prefix (C past actions) + k real noisy
            slots + (N-k) padded slots.  Per-config length: H_cfg = (N+1)*C.
            Buffer layout per config: [clean(C), noisy(k*C), pad((N-k)*C)]

        Returns:
            Dictionary containing:
            - observation.images.*: Shared images [C_img, H, W]
            - task: Task string (shared)
            - observation.state: State [state_dim]
            - action: Actions [N * H_cfg, action_dim]
            - action_is_pad: [N * H_cfg] boolean (True for padded positions)
        """
        N = self.num_buffer_slots
        C = self.chunk_size

        # Get base item
        base_item = super().__getitem__(idx)

        result = {}

        # Copy shared observation keys
        for key in base_item:
            if key.startswith("observation.images.") or key == "task" or key == "observation.state" or key == "episode_index":
                result[key] = base_item[key]

        full_actions = base_item["action"]
        full_action_is_pad = base_item.get("action_is_pad", torch.zeros(full_actions.shape[0], dtype=torch.bool))

        if self.use_action_prefix:
            # full_actions has (N+1)*C entries: first C = past (prefix), rest = future (noisy)
            prefix_actions = full_actions[:C]                # [C, action_dim]
            prefix_is_pad = full_action_is_pad[:C]           # [C]
            noisy_actions = full_actions[C:]                  # [N*C, action_dim]
            noisy_is_pad = full_action_is_pad[C:]             # [N*C]

            H_cfg = (N + 1) * C  # per-config length

            actions_list = []
            action_is_pad_list = []

            for k_idx in range(N):
                k = k_idx + 1  # number of real noisy slots

                # Build config: prefix + noisy slots + padding
                action_config = torch.zeros(H_cfg, full_actions.shape[-1])
                is_pad_config = torch.ones(H_cfg, dtype=torch.bool)

                # Prefix (always present)
                action_config[:C] = prefix_actions
                is_pad_config[:C] = prefix_is_pad

                # Noisy slots: first k*C of the noisy portion
                real_len = min(k * C, noisy_actions.shape[0])
                action_config[C:C + real_len] = noisy_actions[:real_len]
                is_pad_config[C:C + real_len] = noisy_is_pad[:real_len]

                actions_list.append(action_config)
                action_is_pad_list.append(is_pad_config)
        else:
            H_cfg = N * C  # per-config length

            actions_list = []
            action_is_pad_list = []

            for k_idx in range(N):
                k = k_idx + 1

                action_config = torch.zeros_like(full_actions)
                action_is_pad_config = torch.ones(H_cfg, dtype=torch.bool)

                real_len = min(k * C, full_actions.shape[0])
                action_config[:real_len] = full_actions[:real_len]
                action_is_pad_config[:real_len] = full_action_is_pad[:real_len]

                actions_list.append(action_config)
                action_is_pad_list.append(action_is_pad_config)

        result["action"] = torch.cat(actions_list, dim=0)          # [N * H_cfg, action_dim]
        result["action_is_pad"] = torch.cat(action_is_pad_list, dim=0)  # [N * H_cfg]
        return result


def flashvla_collate_fn(batch: list[dict]) -> dict:
    """Collate function for FlashVLADataset.

    Since all samples have exactly N configs, standard default_collate works.
    Final batch shapes:
      - action: [B, N * H, action_dim]
      - action_is_pad: [B, N * H]
      - observation.state: [B, N, state_dim]
    """
    return default_collate(batch)


class _MultiDatasetMetaShim:
    """Duck-typed stand-in for LeRobotDatasetMetadata, exposing the fields used
    during training (make_policy, make_pre_post_processors, logging).
    """

    def __init__(self, sub_metas: list, aggregated_stats: dict):
        assert len(sub_metas) > 0, "MultiFlashVLADataset needs at least one sub-dataset"
        self._sub_metas = sub_metas
        self.stats = aggregated_stats

        first = sub_metas[0]
        # Sanity checks: all sub-datasets must agree on schema and fps.
        for m in sub_metas[1:]:
            if m.fps != first.fps:
                raise ValueError(f"Mixed fps across sub-datasets: {first.fps} vs {m.fps}")
            if set(m.features.keys()) != set(first.features.keys()):
                raise ValueError(
                    "Sub-datasets must share the same feature keys. "
                    f"Got {set(m.features.keys())} vs {set(first.features.keys())}"
                )

        self.info = dict(first.info)
        self.info["total_episodes"] = sum(m.total_episodes for m in sub_metas)
        self.info["total_frames"] = sum(m.total_frames for m in sub_metas)
        self.info["total_tasks"] = sum(m.total_tasks for m in sub_metas)

    # ── fields consumed by LeRobot factory / flashvla training loop ──
    @property
    def fps(self) -> int:
        return self._sub_metas[0].fps

    @property
    def features(self) -> dict:
        return self._sub_metas[0].features

    @property
    def camera_keys(self) -> list:
        return self._sub_metas[0].camera_keys

    @property
    def video_keys(self) -> list:
        return self._sub_metas[0].video_keys

    @property
    def image_keys(self) -> list:
        return self._sub_metas[0].image_keys

    @property
    def names(self) -> dict:
        return self._sub_metas[0].names

    @property
    def shapes(self) -> dict:
        return self._sub_metas[0].shapes

    @property
    def robot_type(self):
        return self._sub_metas[0].robot_type

    @property
    def total_episodes(self) -> int:
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        return self.info["total_tasks"]


class MultiFlashVLADataset(ConcatDataset):
    """Concatenation of multiple FlashVLADataset subsets for joint training.

    All subsets must share identical feature schema and fps. Stats are pooled
    across subsets using lerobot.datasets.compute_stats.aggregate_stats so that
    normalization is consistent across the union.

    Exposes a .meta attribute that duck-types the fields of
    LeRobotDatasetMetadata used by flashvla/train.py:
      - fps, features, camera_keys, video_keys, image_keys, names, shapes
      - stats, robot_type
      - total_episodes, total_frames, total_tasks, info
    Also exposes .num_frames, .num_episodes, and .episodes for training loop logging.
    """

    def __init__(self, subsets: list[FlashVLADataset]):
        if len(subsets) == 0:
            raise ValueError("MultiFlashVLADataset needs at least one subset")
        super().__init__(subsets)
        self.subsets = subsets

        sub_metas = [s.meta for s in subsets]
        pooled = aggregate_stats([m.stats for m in sub_metas if m.stats is not None])
        # aggregate_stats returns numpy; LeRobot normalization expects tensors, same as load_stats.
        self.meta = _MultiDatasetMetaShim(sub_metas, pooled)

        self.num_frames = sum(s.num_frames for s in subsets)
        self.num_episodes = sum(s.num_episodes for s in subsets)
        # `episodes` attribute is read in training-time logging; concatenate.
        self.episodes = None  # EpisodeAwareSampler path is not used by FlashVLA.


def make_robotwin_multitask_dataset(
    root: str | Path,
    config_subdir: str,
    tasks: list[str] | None,
    *,
    num_buffer_slots: int,
    chunk_size: int,
    use_action_prefix: bool = False,
    delta_timestamps: dict[str, list[float]] | None = None,
    image_transforms: Callable | None = None,
    video_backend: str | None = None,
) -> MultiFlashVLADataset:
    """Discover RoboTwin-LeRobot-v3.0 subsets under `root` and build a multi-task dataset.

    Layout: <root>/<task_name>/<config_subdir>/{meta,data,videos}/...
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"RoboTwin root does not exist: {root}")

    if tasks:
        task_names = list(tasks)
    else:
        # Auto-discover: every immediate subdir with <name>/<config_subdir>/meta/info.json
        task_names = sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and (p / config_subdir / "meta" / "info.json").is_file()
        )

    if not task_names:
        raise FileNotFoundError(
            f"No RoboTwin sub-datasets found under {root} with config_subdir={config_subdir}"
        )

    subsets: list[FlashVLADataset] = []
    for task in task_names:
        sub_root = root / task / config_subdir
        if not (sub_root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Missing meta/info.json for RoboTwin subset: {sub_root}")
        subset = FlashVLADataset(
            repo_id=f"robotwin/{task}_{config_subdir}",
            root=sub_root,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=None,
            video_backend=video_backend,
            num_buffer_slots=num_buffer_slots,
            chunk_size=chunk_size,
            use_action_prefix=use_action_prefix,
        )
        subsets.append(subset)

    return MultiFlashVLADataset(subsets)


def make_multi_root_flashvla_dataset(
    repo_id: str,
    roots: list[str | Path],
    *,
    num_buffer_slots: int,
    chunk_size: int,
    use_action_prefix: bool = False,
    delta_timestamps: dict[str, list[float]] | None = None,
    image_transforms: Callable | None = None,
    video_backend: str | None = None,
) -> MultiFlashVLADataset:
    """Build MultiFlashVLADataset from multiple roots sharing the same schema.

    Intended use: mix different settings of the SAME task (e.g., clean_50 +
    randomized_500 for `click_bell`). Sub-datasets are concatenated via
    ConcatDataset; stats are pooled across roots so quantile normalization is
    consistent over the union.

    All roots must have matching features/fps (enforced by _MultiDatasetMetaShim).
    """
    if not roots:
        raise ValueError("make_multi_root_flashvla_dataset needs at least one root")

    subsets: list[FlashVLADataset] = []
    for i, r in enumerate(roots):
        r = Path(r).expanduser()
        if not (r / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Missing meta/info.json for root #{i}: {r}")
        subset = FlashVLADataset(
            repo_id=f"{repo_id}_root{i}",
            root=r,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=None,
            video_backend=video_backend,
            num_buffer_slots=num_buffer_slots,
            chunk_size=chunk_size,
            use_action_prefix=use_action_prefix,
        )
        subsets.append(subset)

    return MultiFlashVLADataset(subsets)
