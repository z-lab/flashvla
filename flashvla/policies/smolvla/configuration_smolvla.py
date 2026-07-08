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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.utils.constants import OBS_IMAGES


@dataclass
class SmolVLAConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    max_state_dim: int = 32
    max_action_dim: int = 32

    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    empty_cameras: int = 0

    adapt_to_pi_aloha: bool = False

    use_delta_joint_actions_aloha: bool = False

    tokenizer_max_length: int = 48

    num_steps: int = 10

    use_cache: bool = True

    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = False

    add_image_special_tokens: bool = False

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"

    num_expert_layers: int = -1
    num_vlm_layers: int = 16
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75

    min_period: float = 4e-3
    max_period: float = 4.0

    compile_model: bool = False
    compile_mode: str = "max-autotune"

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

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
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None


@dataclass
class SmolVLAFlashVLAConfig(SmolVLAConfig):
    """SmolVLA + FlashVLA (padded cold start + shared-observation training).

    Architecture deltas vs the baseline SmolVLA:
    - Buffer of ``num_buffer_slots`` × ``chunk_size`` action tokens; one-step
      inference in steady state instead of the baseline's 10-step ODE.
    - Per-slot time replaces the baseline's single scalar t; the per-token
      concat path (``action_time_mlp_in/out``) handles this naturally without FiLM.
    - State stays in the prefix (smolvla's native layout — unlike pi0/lingbot
      which would put state at suffix index 0). Shared-observation training
      reuses one prefix encoding across all N buffer configs.
    - The suffix uses pi05/pi0-style block-causal ``att_masks=[1]+[0]*(C-1)`` per
      slot (replacing the baseline's per-token AR ``[1]*L_act``); slot-block-causal
      structure emerges.
    """

    chunk_size: int = 10
    num_buffer_slots: int = 5
    n_action_steps: int = 10

    cold_start_mode: str = "zero_delta"

    use_adarms_time_cond: bool = False

    @property
    def total_action_horizon(self) -> int:
        return self.num_buffer_slots * self.chunk_size

    @property
    def total_buffer_slots(self) -> int:
        return self.num_buffer_slots

    @property
    def total_buffer_length(self) -> int:
        return self.total_buffer_slots * self.chunk_size

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.total_action_horizon))
