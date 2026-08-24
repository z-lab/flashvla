# Portions derived from LingBot-VLA, Copyright Robbyant Team.
# Source: https://github.com/Robbyant/lingbot-vla (commit 4eb34b7).
# Modified by the FlashVLA team. Licensed under Apache-2.0.

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@dataclass
class LingBotConfig(PreTrainedConfig):
    """Configuration for LingBot-VLA baseline policy."""

    # === Model Paths ===
    pretrained_path: str | None = None       # QwenvlWithExpert checkpoint
    tokenizer_path: str = ""                 # Qwen2.5-VL-3B-Instruct path

    # === Model Architecture ===
    dtype: str = "bfloat16"

    # === Action Prediction ===
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 75
    max_action_dim: int = 75
    action_dim: int = 14          # Actual action dimensions (before padding)

    # Upstream RoboTwin FeatureTransform does more than right-padding: it
    # maps raw ALOHA order
    #   [left_arm(6), left_grip, right_arm(6), right_grip]
    # to LingBot's typed-joint layout
    #   [arm(12), arm_pad(2), effector(2), pad_to_75].
    # The two arm-pad dimensions are intentionally excluded from the loss.
    # Keep this explicit so released LingBot RoboTwin weights retain their
    # coordinate semantics when trained/deployed through LeRobot processors.
    robotwin_feature_layout: bool = True
    robotwin_arm_dim: int = 12
    robotwin_arm_padded_dim: int = 14
    robotwin_effector_dim: int = 2

    # === Flow Matching ===
    # LingBot's released checkpoints use 10 Euler denoising steps.  Keep the
    # upstream field name (``num_steps``) because FlowMatching.sample_actions
    # reads it directly. ``num_inference_steps`` remains as an optional legacy
    # alias for older FlashVLA configs.
    num_steps: int = 10
    num_inference_steps: int | None = None
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0
    use_cache: bool = True

    # === Image Processing ===
    image_resolution: tuple[int, int] = (224, 224)
    image_resize_mode: str = "stretch"  # upstream RoboTwin uses Resize((224, 224))
    empty_cameras: int = 0
    num_frames_per_view: int = 1

    # === Tokenization ===
    # Matches the released LingBot-VLA RoboTwin training/deployment recipe.
    tokenizer_max_length: int = 72

    # === Qwen-specific ===
    attention_implementation: str = "eager"   # "eager", "flex"
    # Vision attention is independent of the custom VLM/action-expert joint
    # attention above. SDPA preserves per-image/window block semantics without
    # materializing an attention matrix over every image in the whole batch.
    vision_attention_implementation: str = "sdpa"  # "sdpa", "eager", "flash_attention_2"
    adanorm_time: bool = True
    old_adanorm: bool = True
    split_gate_liner: bool = False
    nosplit_gate_liner: bool = False
    separate_time_proj: bool = False
    final_norm_adanorm: bool = False
    norm_qkv: bool = False
    use_lm_head: bool = False

    # Upstream's DINOv3 expert-vision tower. Accepted so released checkpoints
    # keep parsing, but the tower itself is not part of this release and
    # __post_init__ rejects any value other than the default.
    enable_expert_vision: bool = False
    expert_vision_type: str | None = None

    # === Training ===
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    train_state_proj: bool = True
    loss_type: str = "L1_fm"              # "fm" (MSE) or "L1_fm" (L1)
    # Compatibility field for configs written by the earlier LingBot branch.
    # The current trainer deliberately has no activation-checkpointing path.
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None

    # === Normalization ===
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # === Optimizer ===
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # === Scheduler ===
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 1e-5

    # Upstream's depth-alignment auxiliary head. Accepted so released
    # checkpoints keep parsing; the head is not part of this release.
    align_params: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.enable_expert_vision or self.expert_vision_type:
            raise ValueError(
                "The DINOv3 expert-vision tower is not part of the open-source "
                "FlashVLA release; set enable_expert_vision=False and "
                "expert_vision_type=null"
            )
        if self.align_params:
            raise ValueError(
                "The depth-alignment auxiliary head is not part of the open-source "
                "FlashVLA release; leave align_params empty"
            )
        if self.separate_time_proj and not self.adanorm_time:
            raise ValueError(
                "LingBot separate time projection requires adanorm_time=True; "
                "otherwise the time MLP output is unused"
            )
        if self.num_inference_steps is not None:
            self.num_steps = int(self.num_inference_steps)
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        if self.image_resize_mode not in {"stretch", "pad"}:
            raise ValueError(
                f"image_resize_mode must be 'stretch' or 'pad', got {self.image_resize_mode!r}"
            )
        if self.vision_attention_implementation not in {
            "sdpa",
            "eager",
            "flash_attention_2",
        }:
            raise ValueError(
                "vision_attention_implementation must be 'sdpa', 'eager', or "
                f"'flash_attention_2', got {self.vision_attention_implementation!r}"
            )
        if self.robotwin_feature_layout:
            required_model_dim = self.robotwin_arm_padded_dim + self.robotwin_effector_dim
            if self.robotwin_arm_padded_dim < self.robotwin_arm_dim:
                raise ValueError(
                    "robotwin_arm_padded_dim must be >= robotwin_arm_dim, got "
                    f"{self.robotwin_arm_padded_dim} < {self.robotwin_arm_dim}"
                )
            if self.action_dim != self.robotwin_arm_dim + self.robotwin_effector_dim:
                raise ValueError(
                    "robotwin_feature_layout requires action_dim == "
                    f"robotwin_arm_dim + robotwin_effector_dim, got {self.action_dim}"
                )
            if min(self.max_state_dim, self.max_action_dim) < required_model_dim:
                raise ValueError(
                    "max_state_dim/max_action_dim are too small for the RoboTwin "
                    f"LingBot layout ({required_model_dim})"
                )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) > chunk_size ({self.chunk_size})"
            )
        if self.chunk_size <= 0 or self.n_action_steps <= 0:
            raise ValueError("chunk_size and n_action_steps must both be positive")
        if self.loss_type not in {"fm", "L1_fm"}:
            raise ValueError(f"loss_type must be 'fm' or 'L1_fm', got {self.loss_type!r}")

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            self.input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
        if "observation.state" not in self.input_features:
            self.input_features["observation.state"] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(
                    self.action_dim
                    if self.robotwin_feature_layout
                    else self.max_state_dim,
                ),
            )
        if "action" not in self.output_features:
            self.output_features["action"] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(
                    self.action_dim
                    if self.robotwin_feature_layout
                    else self.max_action_dim,
                ),
            )

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
class LingBotFlashVLAConfig(LingBotConfig):
    """Configuration for LingBot-VLA with FlashVLA action streaming."""

    # The default FlashVLA path follows PI0.5: action tokens are projected
    # independently while a fresh time-only MLP drives AdaRMSNorm. Setting
    # this false selects the released LingBot joint action_time_mlp module
    # layout. Whether those loaded weights are retained or reset is controlled
    # independently by flashvla_init.reset_time_mlp in the training recipe.
    separate_time_proj: bool = True
    chunk_size: int = 20
    num_buffer_slots: int = 4
    # RoboTwin executes the full model slot before replanning.
    n_action_steps: int = 20
    timestep_sample_mode: str = "per-sample"

    # Cold-start action strategy while the buffer fills (first N-1 steps emit no
    # chunk). "current_state": re-encode current qpos as the held action (for
    # absolute-qpos envs like RoboTwin); "zero_delta": zero action (delta envs).
    cold_start_mode: str = "current_state"

    # Camera keys whose images are additionally fed to the DINOv3 expert-vision
    # encoder (suffix tokens). Only used when enable_expert_vision=True; empty =
    # expert vision off (default), which matches the baseline (expert_imgs=None).
    expert_image_keys: list[str] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if self.num_buffer_slots <= 0:
            raise ValueError(
                f"num_buffer_slots must be positive, got {self.num_buffer_slots}"
            )
        if not self.use_cache:
            raise ValueError(
                "LingBot FlashVLA inference is suffix-only and requires "
                "use_cache=True (shared-observation training still disables cache "
                "inside its forward pass)"
            )
        if self.timestep_sample_mode not in {"per-sample", "per-chunk"}:
            raise ValueError(
                "timestep_sample_mode must be 'per-sample' or 'per-chunk', "
                f"got {self.timestep_sample_mode!r}"
            )
        if self.cold_start_mode not in {"current_state", "zero_delta"}:
            raise ValueError(
                "cold_start_mode must be 'current_state' or 'zero_delta', "
                f"got {self.cold_start_mode!r}"
            )

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


__all__ = [
    "LingBotConfig",
    "LingBotFlashVLAConfig",
]
