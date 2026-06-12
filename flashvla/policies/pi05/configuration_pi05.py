#!/usr/bin/env python

# Copyright 2025 FlashVLA team. All rights reserved.
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""PI0.5 Policy Configuration.

This module defines the configuration for the PI0.5 (π0.5) Vision-Language-Action
model.
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
class PI05VLMConfig(PaliGemmaConfig):
    """Configuration for the PaliGemma vision-language backbone.

    This configures the multimodal encoder that processes images and
    text prompts to produce embeddings for action generation.
    """

    def __init__(self):
        super().__init__()
        # Vocabulary configuration
        self._vocab_size = 257152
        self.image_token_index = 257152

        # Text encoder (Gemma) configuration
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

        # Vision encoder (SigLIP) configuration
        self.vision_config.intermediate_size = 4304
        self.vision_config.projection_dim = 2048
        self.vision_config.projector_hidden_act = "gelu_fast"
        self.vision_config.torch_dtype = "float32"


@dataclass
class PI05ActionExpertConfig(GemmaConfig):
    """Configuration for the action expert network.

    The action expert is a smaller Gemma model that takes VLM embeddings
    and generates action predictions via flow matching.
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
        # Adaptive RMS conditioning; cond_dim must equal the expert hidden size.
        self.use_adarms = True
        self.adarms_cond_dim = 1024


@dataclass
class PI05Config(PreTrainedConfig):
    """Main configuration for PI0.5 policy."""

    # === Model Architecture ===
    paligemma_variant: str = "gemma_2b"  # VLM backbone: "gemma_300m" or "gemma_2b"
    action_expert_variant: str = "gemma_300m"  # Action expert: "gemma_300m" or "gemma_2b"
    dtype: str = "bfloat16"  # Compute dtype: "bfloat16" or "float32"

    # === Action Prediction ===
    n_obs_steps: int = 1  # Number of observation frames to use
    chunk_size: int = 50  # Number of actions to predict per inference
    n_action_steps: int = 50  # Number of actions to execute before re-inference

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # State conditioning: use robot state in adarmsnorm
    state_cond: bool = False

    # === Flow Matching Parameters ===
    num_inference_steps: int = 10  # Denoising steps during inference
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # === Image Processing ===
    image_resolution: tuple[int, int] = (224, 224)
    empty_cameras: int = 0
    num_frames_per_view: int = 1

    # === Tokenization ===
    tokenizer_max_length: int = 200

    # === Normalization ===
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # === Training Settings ===
    # NOTE: attention_backend / gradient_checkpointing are not read anywhere at
    # runtime. They are kept only because existing checkpoints serialized them
    # into config.json and draccus rejects unknown fields when loading.
    attention_backend: str = "sdpa"
    gradient_checkpointing: bool = False
    freeze_vision_encoder: bool = False  # Freeze SigLIP vision tower only (keep language backbone trainable)
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None

    # Attention/MLP fusion (for inference optimization)
    fuse_qkv: bool = False
    fuse_gate_up: bool = False

    # === Optimizer Settings ===
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # === Scheduler Settings ===
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # === Sub-model Configurations ===
    vlm_config: PI05VLMConfig = field(default_factory=PI05VLMConfig)
    action_expert_config: PI05ActionExpertConfig = field(default_factory=PI05ActionExpertConfig)

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
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None


@dataclass
class PI05FlashVLAConfig(PI05Config):
    """Configuration for FlashVLA (padded cold start with shared observation training).

    Uses a fixed chunk_size (actions per slot), a buffer of num_buffer_slots slots,
    and padding-based warmup instead of full ODE cold start.
    """

    chunk_size: int = 10           # Actions per slot (fixed)
    num_buffer_slots: int = 5      # N = number of slots in buffer
    n_action_steps: int = 10       # Execute one full slot per step
    freeze_vlm: bool = False        # Freeze VLM backbone, only train action expert + suffix embedder
    use_action_prefix: bool = False  # Prepend a clean (fully denoised) chunk as history action conditioning

    # 2026-05-06: RMSNorm-on-time_mlp_out experiment was rejected. It killed
    # the σ_max cascade but also killed eval (0% on bbh). Replaced by
    # ``time_mlp_weight_decay`` below. Field kept (commented) so re-running
    # the ablation only needs an uncomment + flag flip.
    # use_time_mlp_norm: bool = False

    # Time-conditioning sampling for shared-observation training. Only
    # affects how t is drawn for the N buffer configs that share an
    # observation; flow-matching math (within a chunk) is unchanged.
    #
    #   "per-sample" (default, PerSeg) — sample one global t per slot level
    #                  shared across all N configs. time_mlp runs on N (or
    #                  N+1) unique inputs via segment_lookup. Eliminates the
    #                  per-batch incoherence that drove the cascade.
    #   "per-chunk"   — each buffer config independently samples its own t
    #                   for every chunk, even at the same slot level.
    #                   Reproduces the pre-fix behavior (5 different t per
    #                   slot level per batch). Useful for ablations only.
    timestep_sample_mode: str = "per-sample"

    # Optional weight-decay applied ONLY to the suffix embedder's time_mlp_in /
    # time_mlp_out parameters (weight + bias). Direct attack on the cascade
    # root cause (σ_max growth at noise floor) without changing forward pass
    # like RMSNorm did. Default 0.0 = no decay (matches old training setup).
    # Recommended starting value for 50k cosine: 1e-4.
    time_mlp_weight_decay: float = 0.0

    # Cold start action source (for the N-1 steps before the buffer is full).
    #   "zero_delta":   return -mean/std so postprocessor(action) = 0 (safe for delta-action envs).
    #   "current_state": return the current robot state as the action (required for absolute-qpos
    #                    envs like RoboTwin, where zero joints would crash the robot).
    cold_start_mode: str = "zero_delta"

    @property
    def total_action_horizon(self) -> int:
        """Number of action steps in the noisy prediction window (excludes action prefix)."""
        return self.num_buffer_slots * self.chunk_size

    @property
    def total_buffer_slots(self) -> int:
        """Total slots including action prefix if enabled."""
        return self.num_buffer_slots + (1 if self.use_action_prefix else 0)

    @property
    def total_buffer_length(self) -> int:
        """Total buffer length in action tokens."""
        return self.total_buffer_slots * self.chunk_size

    @property
    def action_delta_indices(self) -> list:
        if self.use_action_prefix:
            # Include C past actions for action prefix conditioning
            return list(range(-self.chunk_size, self.num_buffer_slots * self.chunk_size))
        return list(range(self.total_action_horizon))


__all__ = [
    "PI05Config",
    "PI05FlashVLAConfig",
]
