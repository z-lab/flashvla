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
class FlashVLAInitConfig:
    """Optional fresh-run initialization for a streaming action expert.

    ``ae-norm-zero`` is used by the LingBot FlashVLA recipe when converting
    the released joint action/time projection to a separate time projection.
    It is deliberately applied only before a fresh optimizer is created;
    resumed runs keep the parameters stored in their checkpoint.
    """

    mode: str = "default"
    include_final_norm: bool = True
    reset_time_mlp: bool = True
    time_mlp_init: str = "random"
    action_expert_norm_init: str = "zero"


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
    """Training configuration for baseline and FlashVLA policy variants."""

    grad_accum_steps: int = 1

    fsdp: FSDPConfig = field(default_factory=FSDPConfig)

    robotwin_multitask: RoboTwinMultiTaskConfig = field(default_factory=RoboTwinMultiTaskConfig)

    flashvla_init: FlashVLAInitConfig = field(default_factory=FlashVLAInitConfig)

    # Keep frequent checkpoints for requeue safety without retaining every
    # 60+ GB FSDP2 optimizer snapshot forever. Zero preserves all checkpoints.
    checkpoint_keep_last: int = 0
    checkpoint_keep_every_n_steps: int = 0

    def validate(self) -> None:
        if getattr(self.policy, "gradient_checkpointing", False):
            raise ValueError(
                "gradient_checkpointing is not supported by the current FlashVLA "
                "trainer because the previous implementation produced incorrect "
                "gradients. Set it to false for training."
            )
        if self.flashvla_init.mode not in {"default", "ae-norm-zero"}:
            raise ValueError(
                "flashvla_init.mode must be 'default' or 'ae-norm-zero', got "
                f"{self.flashvla_init.mode!r}"
            )
        if self.flashvla_init.time_mlp_init not in {"random", "zero"}:
            raise ValueError(
                "flashvla_init.time_mlp_init must be 'random' or 'zero', got "
                f"{self.flashvla_init.time_mlp_init!r}"
            )
        if self.flashvla_init.action_expert_norm_init not in {"random", "zero"}:
            raise ValueError(
                "flashvla_init.action_expert_norm_init must be 'random' or 'zero', got "
                f"{self.flashvla_init.action_expert_norm_init!r}"
            )
        if (
            self.policy.type == "lingbot-flashvla"
            and not self.resume
            and self.flashvla_init.mode == "ae-norm-zero"
        ):
            if not self.flashvla_init.reset_time_mlp:
                raise ValueError(
                    "Fresh LingBot FlashVLA training requires "
                    "flashvla_init.reset_time_mlp=true"
                )
            if self.flashvla_init.time_mlp_init != "random":
                raise ValueError(
                    "Fresh LingBot FlashVLA training requires a randomly initialized "
                    "time MLP"
                )
        super().validate()
