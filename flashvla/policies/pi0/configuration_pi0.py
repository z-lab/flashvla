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
"""PI0 Policy Configuration.

This module defines the configuration for the PI0 (π0) Vision-Language-Action
model. PI0 is the base VLA model without adaRMS state conditioning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.paligemma.configuration_paligemma import PaliGemmaConfig


@dataclass
class PI0VLMConfig(PaliGemmaConfig):
    """Configuration for the PaliGemma vision-language backbone."""
    
    def __init__(self):
        super().__init__()
        self._vocab_size = 257152
        self.image_token_index = 257152
        
        self.text_config.hidden_size = 2048
        self.text_config.intermediate_size = 16_384
        self.text_config.num_attention_heads = 8
        self.text_config.head_dim = 256
        self.text_config.num_hidden_layers = 18
        self.text_config.num_key_value_heads = 1
        self.text_config.hidden_activation = "gelu_pytorch_tanh"
        self.text_config.torch_dtype = "float32"
        self.text_config.vocab_size = 257152
        self.text_config.use_adarms = False
        self.text_config.adarms_cond_dim = None
        
        self.vision_config.intermediate_size = 4304
        self.vision_config.projection_dim = 2048
        self.vision_config.projector_hidden_act = "gelu_fast"
        self.vision_config.torch_dtype = "float32"


@dataclass
class PI0ActionExpertConfig(GemmaConfig):
    """Configuration for the action expert network.
    """
    
    def __init__(self):
        super().__init__(
            head_dim=256,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=8,
            num_hidden_layers=18,
            num_key_value_heads=1,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
        )
        self.use_adarms = False
        self.adarms_cond_dim = None


@dataclass
class PI0Config(PreTrainedConfig):
    """Main configuration for PI0 policy."""

    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "bfloat16"

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32

    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    image_resolution: tuple[int, int] = (224, 224)
    empty_cameras: int = 0

    tokenizer_max_length: int = 200

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,  
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None

    fuse_qkv: bool = False
    fuse_gate_up: bool = False

    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 200

    vlm_config: PI0VLMConfig = field(default_factory=PI0VLMConfig)
    action_expert_config: PI0ActionExpertConfig = field(default_factory=PI0ActionExpertConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features["action"] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        """Get optimizer configuration from policy settings."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        """Get learning rate scheduler configuration."""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        """Observation frame indices relative to current timestep."""
        return None

    @property
    def action_delta_indices(self) -> list:
        """Action frame indices to predict."""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """Reward indices (not used)."""
        return None


@dataclass
class PI0FlashVLAConfig(PI0Config):
    """Configuration for PI0 FlashVLA (padded cold start with shared-obs training).

    Mirrors PI05FlashVLAConfig but applied to pi0's non-adaRMS architecture.
    Per-slot time is injected through pi0's existing concat-time path
    (``action_time_mlp_in/out``) — no FiLM, no adaRMS, no GemmaRMSNorm patching.
    """

    chunk_size: int = 10
    num_buffer_slots: int = 5
    n_action_steps: int = 10

    cold_start_mode: str = "zero_delta"

    use_adarms_time_cond: bool = False

    @property
    def total_action_horizon(self) -> int:
        """Number of action steps in the noisy prediction window."""
        return self.num_buffer_slots * self.chunk_size

    @property
    def total_buffer_slots(self) -> int:
        """Total slots in the buffer."""
        return self.num_buffer_slots

    @property
    def total_buffer_length(self) -> int:
        """Total buffer length in action tokens."""
        return self.total_buffer_slots * self.chunk_size

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.total_action_horizon))


__all__ = ["PI0Config", "PI0FlashVLAConfig"]
