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
- BaselineTrainConfig: Baseline finetuning config extending LeRobot's TrainPipelineConfig
"""

from dataclasses import dataclass, field
from typing import List

from lerobot.configs.train import TrainPipelineConfig


@dataclass
class DeepSpeedConfig:
    """Configuration for DeepSpeed ZeRO training.

    DeepSpeed ZeRO-2 shards optimizer states and gradients across GPUs
    without flattening parameters, so mixed-precision parameters (some bf16,
    some fp32) work without issues — unlike FSDP which requires uniform dtype.

    We deliberately set bf16.enabled=false in the DeepSpeed config so that
    DeepSpeed does NOT call model.bfloat16() or use its BF16_Optimizer (which
    unconditionally casts all params back to bf16 after each step). Instead,
    bf16 compute is handled by torch.autocast, preserving fp32 for sensitive
    parameters (layernorms, vision tower embeddings).
    """

    enable: bool = False

    # ZeRO stage: 2 (shard optimizer + gradients) or 3 (+ parameters)
    stage: int = 2

    # Offload optimizer states to CPU (saves GPU memory, slower)
    offload_optimizer: bool = False

    # Communication optimization
    allgather_bucket_size: int = int(2e8)
    reduce_bucket_size: int = int(2e8)
    overlap_comm: bool = True


@dataclass
class FSDPConfig:
    """FSDP2 (per-parameter sharding) config — mutually exclusive with deepspeed."""
    enable: bool = False
    mixed_precision: str = "no"
    reshard_after_forward: bool = False
    cpu_offload: bool = False
    wrap_layers: List[str] = field(default_factory=lambda: ["PI05ModelLayer", "SiglipEncoderLayer", "PaliGemmaMultiModalProjector", "PI05SuffixEmbedder", "Embedding"])
    ignored_module_classes: List[str] = field(default_factory=lambda: ["SiglipVisionEmbeddings"])
    ignored_module_name_suffixes: List[str] = field(default_factory=lambda: ["vision_model.post_layernorm", "model.language_model.norm", "action_expert.model.norm", "action_out_proj"])
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
    # Explicit LeRobot leaf roots to concatenate. When set, these roots define
    # the multi-dataset training set directly (e.g. clean + randomized settings
    # of a single task), bypassing root/config_subdir discovery.
    roots: List[str] = field(default_factory=list)
    # Local root containing the RoboTwin-LeRobot-v3.0 tree.
    root: str = ""
    # The per-task subdir name (e.g. "aloha-agilex_randomized_500" or "aloha-agilex_clean_50").
    config_subdir: str = "aloha-agilex_randomized_500"
    # Multiple per-task subdirs to pool per task (e.g. clean_50 + randomized_500
    # together). When non-empty, takes precedence over config_subdir.
    config_subdirs: List[str] = field(default_factory=list)
    # List of task names to include. Empty list = use all tasks found under root
    # that contain the requested config_subdir(s).
    tasks: List[str] = field(default_factory=list)
    # Optional JSON with exact pooled normalization stats. aggregate_stats() pools
    # per-subset quantiles by count-weighted average, which is badly wrong across
    # many tasks (up to ~47% of the q01-q99 range on wrist dims); this file
    # overrides the affected features with exact global stats.
    stats_path: str = ""


@dataclass
class BaselineTrainConfig(TrainPipelineConfig):
    """Baseline (non-streaming) pi0.5 / pi0 finetuning configuration."""

    # Gradient accumulation steps
    grad_accum_steps: int = 1

    # RoboTwin multi-task training: when enabled, concatenate RoboTwin leaves
    # into one baseline training set (see make_robotwin_multitask_baseline_dataset).
    robotwin_multitask: RoboTwinMultiTaskConfig = field(default_factory=RoboTwinMultiTaskConfig)


@dataclass
class FlashVLATrainConfig(TrainPipelineConfig):
    """FlashVLA flashvla training configuration.

    Uses shared observation training with padded cold start buffer.
    """

    # Gradient accumulation steps
    grad_accum_steps: int = 1

    # DeepSpeed ZeRO configuration
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)

    # FSDP2 configuration (mutually exclusive with deepspeed)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)

    # RoboTwin multi-task training configuration
    robotwin_multitask: RoboTwinMultiTaskConfig = field(default_factory=RoboTwinMultiTaskConfig)
