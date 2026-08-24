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

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import ConcatDataset
from torch.utils.data._utils.collate import default_collate

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


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
    ):
        """Initialize FlashVLADataset.

        Args:
            num_buffer_slots: N, number of slots in the denoising buffer.
            chunk_size: C, number of actions per slot.
            Other args: same as LeRobotDataset.
        """
        self.num_buffer_slots = num_buffer_slots
        self.chunk_size = chunk_size
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

        Each config k has k real noisy slots + (N-k) padded slots.
        Per-config length: H = N*C.

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

        base_item = super().__getitem__(idx)

        result = {}

        for key in base_item:
            if key.startswith("observation.images.") or key == "task" or key == "observation.state" or key == "episode_index":
                result[key] = base_item[key]

        full_actions = base_item["action"]
        full_action_is_pad = base_item.get("action_is_pad", torch.zeros(full_actions.shape[0], dtype=torch.bool))

        H_cfg = N * C

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

        result["action"] = torch.cat(actions_list, dim=0)
        result["action_is_pad"] = torch.cat(action_is_pad_list, dim=0)
        return result


def flashvla_collate_fn(batch: list[dict]) -> dict:
    """Collate function for FlashVLADataset.

    Since all samples have exactly N configs, standard default_collate works.
    Final batch shapes:
      - action: [B, N * H, action_dim]
      - action_is_pad: [B, N * H]
      - observation.state: [B, state_dim]
    """
    return default_collate(batch)


class _MultiDatasetMeta:
    """Duck-typed stand-in for LeRobotDatasetMetadata, exposing the fields used
    during training (make_policy, make_pre_post_processors, logging).
    """

    def __init__(self, sub_metas: list, aggregated_stats: dict):
        assert len(sub_metas) > 0, "MultiFlashVLADataset needs at least one sub-dataset"
        self._sub_metas = sub_metas
        self.stats = aggregated_stats

        first = sub_metas[0]
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
    """Concatenation of multiple LeRobotDataset-compatible subsets for joint training.

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

    def __init__(self, subsets: list[LeRobotDataset]):
        if len(subsets) == 0:
            raise ValueError("MultiFlashVLADataset needs at least one subset")
        super().__init__(subsets)
        self.subsets = subsets

        sub_metas = [s.meta for s in subsets]
        pooled = aggregate_stats([m.stats for m in sub_metas if m.stats is not None])
        self.meta = _MultiDatasetMeta(sub_metas, pooled)

        self.num_frames = sum(s.num_frames for s in subsets)
        self.num_episodes = sum(s.num_episodes for s in subsets)
        self.episodes = None


def make_robotwin_multitask_flashvla_dataset(cfg, image_transforms=None) -> MultiFlashVLADataset:
    """Build a RoboTwin multi-task dataset by concatenating LeRobot leaves.

    Two modes, both driven by `cfg.robotwin_multitask`:
      * `roots` — explicit LeRobot leaf roots (e.g. clean + randomized settings of
        one task), concatenated directly.
      * `root` + `config_subdir(s)` (+ optional `tasks`) — discover task leaves under
        a RoboTwin-LeRobot-v3.0 tree (`<root>/<task>/<config_subdir>/...`). A list of
        `config_subdirs` pools every (task, subdir) leaf; tasks are auto-discovered
        only if they contain ALL requested subdirs.

    `stats_path`, if set, overrides the pooled stats with exact global ones. A
    FlashVLA policy gets FlashVLADataset subsets, while a baseline policy gets
    ordinary LeRobotDataset subsets with its configured action timestamps.
    """
    mt = cfg.robotwin_multitask
    if mt.roots:
        root_specs = [
            (f"{cfg.dataset.repo_id}_root{i}", Path(r).expanduser())
            for i, r in enumerate(mt.roots)
        ]
    else:
        if not mt.root:
            raise ValueError(
                "cfg.robotwin_multitask.enable=True requires robotwin_multitask.roots "
                "or robotwin_multitask.root to be set"
            )
        config_subdirs = mt.config_subdirs or [mt.config_subdir]
        root = Path(mt.root).expanduser()
        if mt.tasks:
            task_names = list(mt.tasks)
        else:
            task_names = sorted(
                p.name for p in root.iterdir()
                if p.is_dir()
                and all((p / s / "meta" / "info.json").is_file() for s in config_subdirs)
            )
        if not task_names:
            raise FileNotFoundError(
                f"No RoboTwin sub-datasets found under {root} with config_subdirs={config_subdirs}"
            )
        root_specs = [
            (f"robotwin/{task}_{subdir}", root / task / subdir)
            for task in task_names
            for subdir in config_subdirs
        ]

    probe_repo_id, probe_root = root_specs[0]
    ds_meta = LeRobotDatasetMetadata(probe_repo_id, root=probe_root, revision=None)
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

    is_flashvla_policy = hasattr(cfg.policy, "num_buffer_slots")
    subsets: list[LeRobotDataset] = []
    for repo_id, sub_root in root_specs:
        if not (sub_root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Missing meta/info.json for RoboTwin subset: {sub_root}")
        dataset_kwargs = {
            "repo_id": repo_id,
            "root": sub_root,
            "delta_timestamps": delta_timestamps,
            "image_transforms": image_transforms,
            "revision": None,
            "video_backend": cfg.dataset.video_backend,
        }
        if is_flashvla_policy:
            subset = FlashVLADataset(
                **dataset_kwargs,
                num_buffer_slots=cfg.policy.num_buffer_slots,
                chunk_size=cfg.policy.chunk_size,
            )
        else:
            subset = LeRobotDataset(**dataset_kwargs)
        subsets.append(subset)

    dataset = MultiFlashVLADataset(subsets)

    if mt.stats_path:
        import json as _json
        from lerobot.datasets.io_utils import cast_stats_to_numpy

        with open(Path(mt.stats_path).expanduser()) as f:
            dataset.meta.stats.update(cast_stats_to_numpy(_json.load(f)))

    return dataset
