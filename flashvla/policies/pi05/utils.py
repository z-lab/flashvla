#!/usr/bin/env python

# Copyright 2025 FlashVLA team. All rights reserved.
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""PI0.5 Utility Functions.

This module provides utility functions for the PI0.5 model:
- Dtype handling for device compatibility
- Sinusoidal positional embeddings for flow matching
- Vector padding for variable-length inputs
- Attention mask construction
- Image resizing with aspect-ratio-preserving padding
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor




def get_safe_dtype(dtype: torch.dtype, device: str | torch.device) -> torch.dtype:
    """Get a device-compatible dtype, falling back if necessary.
    
    Some devices don't support float64:
    - MPS (Apple Silicon): No float64 support
    - Some Intel XPU: May lack FP64 capability
    
    Args:
        dtype: Requested dtype.
        device: Target device.
        
    Returns:
        The original dtype if supported, otherwise float32.
    """
    if isinstance(device, torch.device):
        device = device.type
        
    # MPS doesn't support float64
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    
    # Some Intel XPU devices lack FP64
    if device == "xpu" and dtype == torch.float64:
        if hasattr(torch.xpu, "get_device_capability"):
            device_capability = torch.xpu.get_device_capability()
            if not device_capability.get("has_fp64", False):
                logging.warning(f"Device {device} does not support float64, using float32 instead.")
                return torch.float32
        else:
            logging.warning(
                f"Device {device} capability check failed. Assuming no support for float64, using float32 instead."
            )
            return torch.float32
        return dtype
    else:
        return dtype


# ============================================================
# Cold-start helpers for FlashVLA inference
# ============================================================
#
# Action streaming fills a rolling denoise buffer over the first
# (N-1) inference calls. During this window no real action can be
# produced yet, so the policy must return a "hold the robot still"
# action in whatever space the env expects. The helpers below keep
# all normalization math out of the modeling file — the policy only
# needs to load a :class:`ColdStartStats` once and call
# :func:`compute_cold_start_action` per inference.


@dataclass
class ColdStartStats:
    """Per-feature normalization stats used to construct cold-start actions.

    All tensors come straight from the policy's postprocessor safetensors
    (``policy_postprocessor_step_0_unnormalizer_processor.safetensors``).
    Loaded fields depend on the normalization mode declared in the
    postprocessor JSON's ``norm_map`` — MEAN_STD populates
    ``{action,state}_{mean,std}``; QUANTILES populates
    ``{action,state}_{q01,q99}``. Fields for the unused mode stay ``None``.
    """

    # Normalization mode as it appears in policy_postprocessor.json's norm_map.
    # Supported: "MEAN_STD", "QUANTILES". Unknown modes fall back to MEAN_STD
    # with a warning.
    action_norm_mode: str = "MEAN_STD"
    state_norm_mode: str = "MEAN_STD"

    action_mean: Tensor | None = None
    action_std: Tensor | None = None
    action_q01: Tensor | None = None
    action_q99: Tensor | None = None
    state_mean: Tensor | None = None
    state_std: Tensor | None = None
    state_q01: Tensor | None = None
    state_q99: Tensor | None = None

    @property
    def has_action(self) -> bool:
        if self.action_norm_mode == "QUANTILES":
            return self.action_q01 is not None and self.action_q99 is not None
        return self.action_mean is not None and self.action_std is not None

    @property
    def has_state(self) -> bool:
        if self.state_norm_mode == "QUANTILES":
            return self.state_q01 is not None and self.state_q99 is not None
        return self.state_mean is not None and self.state_std is not None


def _read_norm_map(checkpoint_dir: Path) -> dict:
    """Return the postprocessor's norm_map, or empty dict if unavailable."""
    pp_json = checkpoint_dir / "policy_postprocessor.json"
    if not pp_json.exists():
        return {}
    try:
        doc = json.loads(pp_json.read_text())
    except Exception:
        return {}
    for step in doc.get("steps", []):
        if step.get("registry_name") == "unnormalizer_processor":
            return step.get("config", {}).get("norm_map", {}) or {}
    return {}


def load_cold_start_stats(checkpoint_dir: str | Path) -> ColdStartStats:
    """Load action + state normalization stats from a flashvla checkpoint dir.

    Reads ``policy_postprocessor.json`` to pick the right mode per feature
    (MEAN_STD vs QUANTILES), then loads matching tensors from
    ``policy_postprocessor_step_0_unnormalizer_processor.safetensors``.
    Returns an empty :class:`ColdStartStats` if the safetensors file is
    missing — callers must fall back to a safe default (e.g. zero action).
    """
    from safetensors.torch import load_file as _load_sf

    checkpoint_dir = Path(checkpoint_dir)
    stats_file = checkpoint_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    if not stats_file.exists():
        logging.warning(
            f"No postprocessor stats at {stats_file}. "
            "Cold start will fall back to zero action."
        )
        return ColdStartStats()

    raw = _load_sf(str(stats_file))
    norm_map = _read_norm_map(checkpoint_dir)

    def _mode_for(feat_key: str) -> str:
        m = norm_map.get(feat_key, "MEAN_STD")
        if m not in ("MEAN_STD", "QUANTILES"):
            logging.warning(
                f"Unknown norm mode '{m}' for {feat_key} in {checkpoint_dir}; "
                "falling back to MEAN_STD."
            )
            return "MEAN_STD"
        return m

    out = ColdStartStats(
        action_norm_mode=_mode_for("ACTION"),
        state_norm_mode=_mode_for("STATE"),
    )

    # Load whichever pair of stats the declared mode needs; fall back to
    # whatever else is available so downstream can still attempt best-effort.
    def _maybe(key: str) -> Tensor | None:
        return raw[key] if key in raw else None

    out.action_mean = _maybe("action.mean")
    out.action_std = _maybe("action.std")
    out.action_q01 = _maybe("action.q01")
    out.action_q99 = _maybe("action.q99")
    out.state_mean = _maybe("observation.state.mean")
    out.state_std = _maybe("observation.state.std")
    out.state_q01 = _maybe("observation.state.q01")
    out.state_q99 = _maybe("observation.state.q99")

    # Sanity check: warn if the declared mode's stats are missing.
    if out.action_norm_mode == "QUANTILES" and (out.action_q01 is None or out.action_q99 is None):
        logging.warning(f"action.q01/q99 missing in {stats_file} despite QUANTILES mode.")
    if out.action_norm_mode == "MEAN_STD" and (out.action_mean is None or out.action_std is None):
        logging.warning(f"action.mean/std missing in {stats_file}.")
    if out.state_norm_mode == "QUANTILES" and (out.state_q01 is None or out.state_q99 is None):
        logging.warning(f"observation.state.q01/q99 missing in {stats_file} despite QUANTILES mode.")
    return out


def compute_normalized_zero_action(
    stats: ColdStartStats,
    action_dim: int,
    device: torch.device,
) -> Tensor:
    """Normalized action whose postprocessor-unnormalize maps to **zero**.

    MEAN_STD  : ``real = norm*std + mean`` → solve ``real=0`` → ``norm = -mean/std``.
    QUANTILES : ``real = (norm+1)*(q99-q01)/2 + q01`` → ``norm = -2*q01/(q99-q01) - 1``.

    Falls back to zeros when the declared mode's stats aren't available —
    the postprocessor will then emit the *mean* (or midpoint) action instead
    of a true zero.
    """
    if stats.has_action:
        if stats.action_norm_mode == "QUANTILES":
            q01 = stats.action_q01[:action_dim].to(device=device, dtype=torch.float32)
            q99 = stats.action_q99[:action_dim].to(device=device, dtype=torch.float32)
            denom = (q99 - q01).clamp(min=1e-8)
            return (-2.0 * q01 / denom - 1.0).unsqueeze(0)
        mean = stats.action_mean[:action_dim].to(device=device, dtype=torch.float32)
        std = stats.action_std[:action_dim].to(device=device, dtype=torch.float32).clamp(min=1e-8)
        return (-mean / std).unsqueeze(0)
    logging.warning(
        "compute_normalized_zero_action: no action stats loaded; cold start "
        "will send the mean action instead of zero."
    )
    return torch.zeros(1, action_dim, device=device)


def compute_normalized_current_state_action(
    state_normalized: Tensor,
    stats: ColdStartStats,
    action_dim: int,
    device: torch.device,
) -> Tensor:
    """Fake action that postprocessor-unnormalizes to the **current qpos**.

    Needed for absolute-qpos environments (e.g. RoboTwin): zero would
    send the arm to joint zero and crash, so we instead return the
    current robot state encoded in the action-normalization space so
    that ``postprocessor(action) == real_current_qpos``. Pipeline:

      (1) state_normalized ──(unnormalize with state stats)──▶ real qpos
      (2) real qpos        ──(renormalize  with action stats)─▶ fake action

    Args:
        state_normalized: ``[B, state_dim]`` already normalized by the
            policy's preprocessor. Only the first ``action_dim`` slots are
            used; padding dims added by ``prepare_state`` are ignored.
        stats: Stats loaded via :func:`load_cold_start_stats`.
        action_dim: Original (non-padded) action dimensionality.
        device: Where the returned tensor should live.
    """
    state = state_normalized[:, :action_dim].to(device=device, dtype=torch.float32)

    if not stats.has_action:
        logging.warning(
            "compute_normalized_current_state_action: no action stats loaded; "
            "returning raw normalized state (postprocessor will misinterpret units)."
        )
        return state

    # (1) state_normalized → unnormalize with state stats → real qpos
    if stats.has_state:
        if stats.state_norm_mode == "QUANTILES":
            s_q01 = stats.state_q01[:action_dim].to(device=device, dtype=torch.float32)
            s_q99 = stats.state_q99[:action_dim].to(device=device, dtype=torch.float32)
            state_real = (state + 1.0) * (s_q99 - s_q01) / 2.0 + s_q01
        else:
            s_mean = stats.state_mean[:action_dim].to(device=device, dtype=torch.float32)
            s_std = stats.state_std[:action_dim].to(device=device, dtype=torch.float32).clamp(min=1e-8)
            state_real = state * s_std + s_mean
    else:
        # No state stats — assume state is already in real units.
        state_real = state

    # (2) real qpos → renormalize with action stats → fake action
    if stats.action_norm_mode == "QUANTILES":
        a_q01 = stats.action_q01[:action_dim].to(device=device, dtype=torch.float32)
        a_q99 = stats.action_q99[:action_dim].to(device=device, dtype=torch.float32)
        denom = (a_q99 - a_q01).clamp(min=1e-8)
        return 2.0 * (state_real - a_q01) / denom - 1.0

    a_mean = stats.action_mean[:action_dim].to(device=device, dtype=torch.float32)
    a_std = stats.action_std[:action_dim].to(device=device, dtype=torch.float32).clamp(min=1e-8)
    return (state_real - a_mean) / a_std


def compute_cold_start_action(
    mode: str,
    stats: ColdStartStats,
    action_dim: int,
    device: torch.device,
    state_normalized: Tensor | None = None,
    cache: Tensor | None = None,
) -> Tensor:
    """Dispatch to the right cold-start strategy given a config mode.

    Args:
        mode: ``"zero_delta"`` (safe for delta-action envs) or
            ``"current_state"`` (required for absolute-qpos envs).
        stats: Loaded :class:`ColdStartStats`.
        action_dim: Original action dimensionality.
        device: Target device for the returned tensor.
        state_normalized: Current normalized state — only required for
            ``"current_state"`` mode.
        cache: Optional pre-computed zero-action tensor to short-circuit
            the ``"zero_delta"`` path (state-independent, so it's safe to
            cache across calls).

    Returns:
        ``[1, action_dim]`` tensor in the normalized action space.
    """
    if mode == "zero_delta":
        if cache is not None:
            return cache
        return compute_normalized_zero_action(stats, action_dim, device)
    if mode == "current_state":
        if state_normalized is None:
            raise ValueError("current_state cold start requires state_normalized")
        return compute_normalized_current_state_action(
            state_normalized, stats, action_dim, device
        )
    raise ValueError(f"Unknown cold_start_mode: {mode!r}")


# ============================================================
# Flow matching helpers
# ============================================================


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Create sinusoidal positional embeddings for scalar timesteps.
    
    Used in flow matching to encode the diffusion timestep t ∈ [0, 1].
    Creates embeddings with frequencies spanning from min_period to max_period.
    
    The embedding formula:
        emb[i] = sin(t * 2π / period_i)  for i < dim/2
        emb[i] = cos(t * 2π / period_i)  for i >= dim/2
    
    where period_i = min_period * (max_period/min_period)^(i / (dim/2 - 1))
    
    Args:
        time: Scalar timesteps [batch_size].
        dimension: Embedding dimension (must be even).
        min_period: Minimum sinusoidal period.
        max_period: Maximum sinusoidal period.
        device: Target device.
        
    Returns:
        Positional embeddings [batch_size, dimension].
        
    Raises:
        ValueError: If dimension is odd or time is not 1D.
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    # Use float64 for precision, with device compatibility fallback
    dtype = get_safe_dtype(torch.float64, device.type)
    
    # Create log-spaced frequencies from min to max period
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute sinusoidal embedding
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    
    return pos_emb


def create_sinusoidal_pos_embedding_for_blocks(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Create sinusoidal positional embeddings for block-wise timesteps.
    
    Used in flow matching to encode the diffusion timestep t ∈ [0, 1].
    Creates embeddings with frequencies spanning from min_period to max_period.
    
    The embedding formula:
        emb[i] = sin(t * 2π / period_i)  for i < dim/2
        emb[i] = cos(t * 2π / period_i)  for i >= dim/2
    
    where period_i = min_period * (max_period/min_period)^(i / (dim/2 - 1))
    
    Args:
        time: Block-wise timesteps [batch_size, num_blocks * block_size].
        dimension: Embedding dimension (must be even).
        min_period: Minimum sinusoidal period.
        max_period: Maximum sinusoidal period.
        device: Target device.
        
    Returns:
        Positional embeddings [batch_size, num_blocks * block_size, dimension].
        
    Raises:
        ValueError: If dimension is odd or time is not 1D.
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 2:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, num_blocks * block_size)`.")

    # Use float64 for precision, with device compatibility fallback
    dtype = get_safe_dtype(torch.float64, device.type)
    
    # Create log-spaced frequencies from min to max period
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute sinusoidal embedding
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, None, :] * time[:, :, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=2)
    
    return pos_emb


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    """Pad the last dimension of a vector to a target size.
    
    Useful for padding state/action vectors to a fixed maximum dimension
    for batching across different robot configurations.
    
    Args:
        vector: Input tensor, either [batch, features] or [batch, seq, features].
        new_dim: Target size for the last dimension.
        
    Returns:
        Padded tensor with last dimension == new_dim.
        Original values are preserved, padding is zeros.
    """
    if vector.shape[-1] == new_dim:
        return vector
        
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    
    return new_vector


def build_attention_mask_and_position_ids(
    pad_masks: torch.Tensor,
    att_masks: torch.Tensor,
    dtype: torch.dtype,
    extra_kv_length: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build 4D attention mask and position IDs for transformer.

    Constructs attention patterns that support:
    - Padding: masked positions are ignored
    - Block-causal structure: tokens can only attend to earlier blocks

    The att_masks tensor encodes block boundaries. Tokens with the same
    cumulative sum value are in the same block and can attend to each other.

    Args:
        pad_masks: Boolean mask [B, N], True for real tokens, False for padding.
        att_masks: Block structure mask [B, N], non-zero marks block boundaries.
        dtype: Output dtype for attention mask.
        extra_kv_length: Extra columns to append to the kv dimension, masked
            with -inf. Used to match StaticCache's max_cache_len when the cache
            reserves slots for future tokens (e.g. action chunk).

    Returns:
        attention_mask: Additive attention mask [B, 1, N, N + extra_kv_length].
                       0 for allowed attention, -inf for blocked.
        position_ids: Position indices [B, N] for rotary embeddings.

    Raises:
        ValueError: If input tensors are not 2D.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    # Build block-causal mask: token i can attend to j if cumsum[j] <= cumsum[i]
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]

    # Combine with padding mask
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks  # [B, N, N] bool

    # Position IDs: cumulative count of non-padding tokens
    position_ids = torch.cumsum(pad_masks, dim=1) - 1  # [B, N]

    # Convert to additive mask format (0 for attend, -inf for block)
    mask_value = torch.finfo(dtype).min
    attention_mask = torch.where(
        att_2d_masks,
        torch.zeros_like(att_2d_masks, dtype=dtype),
        torch.full_like(att_2d_masks, mask_value, dtype=dtype),
    )
    attention_mask = attention_mask.unsqueeze(1)  # [B, 1, N, N]

    # Extend kv dimension for StaticCache's unfilled future slots
    if extra_kv_length > 0:
        bsz, _, q_len, _ = attention_mask.shape
        extra = torch.full(
            (bsz, 1, q_len, extra_kv_length),
            mask_value,
            device=attention_mask.device,
            dtype=dtype,
        )
        attention_mask = torch.cat([attention_mask, extra], dim=-1)

        last_token_position = position_ids[:, -1:]  # [B, 1] — preserve 2D for broadcasting
        extra_position_ids = last_token_position + torch.arange(
            1, extra_kv_length + 1, device=position_ids.device
        )  # [B, 1] + [E] → [B, E]
        position_ids = torch.cat([position_ids, extra_position_ids], dim=1)  # [B, N+E]

    return attention_mask, position_ids


def resize_with_pad(
    img: Tensor,
    width: int,
    height: int,
    pad_value: float = -1,
) -> Tensor:
    """Resize image to target size while preserving aspect ratio.
    
    The image is scaled to fit within the target dimensions, then
    padded on the left and top to reach the exact target size.
    This matches the preprocessing used by SigLIP in PI0.5.
    
    Args:
        img: Input image [B, C, H, W].
        width: Target width.
        height: Target height.
        pad_value: Value for padding pixels (default -1 for SigLIP range).
        
    Returns:
        Resized and padded image [B, C, height, width].
        
    Raises:
        ValueError: If input is not 4D.
    """
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    # Scale to fit within target dimensions
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    # Pad to reach exact target size (pad left and top)
    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    
    return padded_img


def build_shared_obs_attention_mask_and_position_ids(
    prefix_pad_masks: torch.Tensor,
    prefix_att_masks: torch.Tensor,
    suffix_pad_masks: torch.Tensor,
    suffix_att_masks: torch.Tensor,
    num_offsets: int,
    offset_mask: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build attention mask and position IDs for shared observation training.
    
    This function properly handles padding tokens and builds the correct attention
    pattern by reusing the logic from build_attention_mask_and_position_ids.
    
    Args:
        prefix_pad_masks: Padding mask for prefix [B, prefix_length].
        prefix_att_masks: Attention structure mask for prefix [B, prefix_length].
        suffix_pad_masks: Padding mask for one suffix [B, suffix_length].
        suffix_att_masks: Attention structure mask for one suffix [B, suffix_length].
        num_offsets: Number of offset branches.
        offset_mask: Boolean mask [B, num_offsets] indicating valid offsets.
        dtype: Output dtype for attention mask.
        
    Returns:
        attention_mask: Additive mask [B, 1, total_length, total_length].
        position_ids: Position indices [B, total_length].
    """
    batch_size = prefix_pad_masks.shape[0]
    prefix_length = prefix_pad_masks.shape[1]
    suffix_length = suffix_pad_masks.shape[1]
    total_length = prefix_length + suffix_length * num_offsets
    device = prefix_pad_masks.device
    mask_value = torch.finfo(dtype).min
    
    # Build combined pad_masks and att_masks for the full sequence (vectorized)
    # Suffix is repeated for each offset
    full_pad_masks = torch.zeros(batch_size, total_length, dtype=torch.bool, device=device)
    full_att_masks = torch.zeros(batch_size, total_length, dtype=prefix_att_masks.dtype, device=device)
    
    # Prefix part
    full_pad_masks[:, :prefix_length] = prefix_pad_masks
    full_att_masks[:, :prefix_length] = prefix_att_masks
    
    # Suffix parts: tile suffix masks for all offsets at once
    # [B, suffix_length] -> [B, num_offsets * suffix_length]
    suffix_pad_tiled = suffix_pad_masks.unsqueeze(1).expand(-1, num_offsets, -1).reshape(batch_size, -1)
    suffix_att_tiled = suffix_att_masks.unsqueeze(1).expand(-1, num_offsets, -1).reshape(batch_size, -1)
    full_pad_masks[:, prefix_length:] = suffix_pad_tiled
    full_att_masks[:, prefix_length:] = suffix_att_tiled
    
    # Compute cumsum for block-causal structure
    cumsum = torch.cumsum(full_att_masks, dim=1)
    
    # Build the base 2D attention mask using cumsum logic
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    
    # Apply padding mask
    pad_2d_masks = full_pad_masks[:, None, :] * full_pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    
    # Block cross-offset attention (vectorized)
    # Create a mask that blocks attention between different offset branches
    # For positions in suffix, determine which offset they belong to
    suffix_positions = torch.arange(num_offsets * suffix_length, device=device)
    offset_ids = suffix_positions // suffix_length  # [num_offsets * suffix_length]
    
    # Create cross-offset blocking mask for suffix-to-suffix attention
    # query_offset != key_offset should be blocked
    query_offset_ids = offset_ids.unsqueeze(1)  # [S, 1]
    key_offset_ids = offset_ids.unsqueeze(0)    # [1, S]
    cross_offset_mask = (query_offset_ids == key_offset_ids)  # [S, S], True where same offset
    
    # Apply to the suffix-suffix part of att_2d_masks
    suffix_start = prefix_length
    att_2d_masks[:, suffix_start:, suffix_start:] = att_2d_masks[:, suffix_start:, suffix_start:] & cross_offset_mask
    
    # Apply offset_mask (vectorized): invalid offsets should be fully masked
    # Create per-position validity mask from offset_mask [B, num_offsets] -> [B, num_offsets * suffix_length]
    offset_validity = offset_mask.unsqueeze(2).expand(-1, -1, suffix_length).reshape(batch_size, -1)  # [B, S]
    
    # Mask out invalid offset positions in att_2d_masks
    # Invalid queries can't attend to anything
    att_2d_masks[:, suffix_start:, :] = att_2d_masks[:, suffix_start:, :] & offset_validity.unsqueeze(2)
    # Nothing can attend to invalid keys
    att_2d_masks[:, :, suffix_start:] = att_2d_masks[:, :, suffix_start:] & offset_validity.unsqueeze(1)
    
    # Compute position IDs (vectorized)
    position_ids = torch.zeros(batch_size, total_length, dtype=torch.long, device=device)
    
    # Prefix position IDs
    prefix_pos = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
    position_ids[:, :prefix_length] = prefix_pos
    
    # Get the last valid prefix position for each batch
    last_prefix_pos = prefix_pos[:, -1]  # [B]
    
    # Suffix position IDs: each branch continues from last_prefix_pos + 1
    # Tile suffix positions for all offsets: [B, suffix_length] -> [B, num_offsets * suffix_length]
    suffix_pos_base = torch.cumsum(suffix_pad_masks.long(), dim=1)  # [B, suffix_length]
    suffix_pos_tiled = suffix_pos_base.unsqueeze(1).expand(-1, num_offsets, -1).reshape(batch_size, -1)
    position_ids[:, prefix_length:] = last_prefix_pos[:, None] + suffix_pos_tiled
    
    # Convert to additive mask
    attention_mask = torch.where(
        att_2d_masks,
        torch.zeros_like(att_2d_masks, dtype=dtype),
        torch.full_like(att_2d_masks, mask_value, dtype=dtype),
    )
    attention_mask = attention_mask.unsqueeze(1)

    return attention_mask, position_ids
