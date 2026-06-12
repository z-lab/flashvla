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
"""FlashVLA Datasets Module.

This module provides dataset utilities for FlashVLA training:
- DelayAugmentedDataset: Extends LeRobotDataset with temporal delay augmentation
- Compatibility patches for lerobot v2.1/v3.0 formats

The compat module is imported first to patch lerobot before other imports.
"""

# Apply compatibility patches for lerobot v2.1/v3.0 before importing anything else
import flashvla.datasets.compat  # noqa: F401

from flashvla.datasets.delay_dataset import (
    DelayAugmentedDataset,
    SharedObservationDataset,
    shared_observation_collate_fn,
)
from flashvla.datasets.flashvla_dataset import FlashVLADataset, flashvla_collate_fn
from flashvla.datasets.task_filter import (
    LIBERO_SUITE_TASKS,
    get_episode_indices_for_suite,
    get_episode_indices_for_tasks,
    get_frame_indices_for_suite,
    resolve_suite_episodes,
)

__all__ = [
    "DelayAugmentedDataset",
    "SharedObservationDataset",
    "shared_observation_collate_fn",
    "FlashVLADataset",
    "flashvla_collate_fn",
    "LIBERO_SUITE_TASKS",
    "get_episode_indices_for_suite",
    "get_episode_indices_for_tasks",
    "get_frame_indices_for_suite",
    "resolve_suite_episodes",
]
