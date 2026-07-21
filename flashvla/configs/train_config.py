#!/usr/bin/env python

# Copyright 2025 FlashVLA team. All rights reserved.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""FlashVLA Training Configuration.

This module defines configuration classes for FlashVLA training:
- FlashVLATrainConfig: FlashVLA action-streaming training config extending LeRobot's TrainPipelineConfig
"""

from dataclasses import dataclass, field
from typing import List

from lerobot.configs.train import TrainPipelineConfig


@dataclass
class FSDPConfig:
    """FSDP2 per-parameter sharding configuration."""
    enable: bool = False
    mixed_precision: str = "no"
    reduce_dtype: str = "float32"
    reshard_after_forward: bool = False
    cpu_offload: bool = False
    state_dict_type: str = "SHARDED_STATE_DICT"
    save_pretrained_max_shard_size: str = "50GB"


@dataclass
class RoboTwinMultiTaskConfig:
    """Multi-task training config for the RoboTwin-LeRobot-v3.0 dataset layout.

    The dataset on disk is organized as:
        <root>/<task_name>/<config_subdir>/{meta,data,videos}/...
    Each leaf directory is itself a complete LeRobotDataset v3.0. When enabled,
    the trainer builds one dataset per leaf and concatenates them.
    """

    enable: bool = False
    roots: List[str] = field(default_factory=list)
    root: str = ""
    config_subdir: str = "aloha-agilex_randomized_500"
    config_subdirs: List[str] = field(default_factory=list)
    tasks: List[str] = field(default_factory=list)
    stats_path: str = ""


@dataclass
class FlashVLATrainConfig(TrainPipelineConfig):
    """FlashVLA flashvla training configuration.

    Uses shared observation training with padded cold start buffer.
    """

    grad_accum_steps: int = 1

    fsdp: FSDPConfig = field(default_factory=FSDPConfig)

    robotwin_multitask: RoboTwinMultiTaskConfig = field(default_factory=RoboTwinMultiTaskConfig)

    # Keep frequent checkpoints for requeue safety without retaining every
    # 60+ GB FSDP2 optimizer snapshot forever. Zero preserves all checkpoints.
    checkpoint_keep_last: int = 0
    checkpoint_keep_every_n_steps: int = 0
