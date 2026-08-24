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

import logging
import math

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
        
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    
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

    dtype = get_safe_dtype(torch.float64, device.type)
    
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    
    return pos_emb


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    """Pad the last dimension of a vector to a target size.
    
    Args:
        vector: Input tensor [batch, features] or [batch, seq, features].
        new_dim: Target size for the last dimension.
        
    Returns:
        Padded tensor with zeros.
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build 4D attention mask and position IDs for transformer.
    
    Constructs attention patterns that support:
    - Padding: masked positions are ignored
    - Block-causal structure: tokens can only attend to earlier blocks
    
    Args:
        pad_masks: Boolean mask [B, N], True for real tokens.
        att_masks: Block structure mask [B, N].
        dtype: Output dtype for attention mask.
        
    Returns:
        attention_mask: Additive mask [B, 1, N, N] (0 or -inf).
        position_ids: Position indices [B, N].
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks

    position_ids = torch.cumsum(pad_masks, dim=1) - 1

    mask_value = torch.finfo(dtype).min
    attention_mask = torch.where(
        att_2d_masks,
        torch.zeros_like(att_2d_masks, dtype=dtype),
        torch.full_like(att_2d_masks, mask_value, dtype=dtype),
    )
    attention_mask = attention_mask.unsqueeze(1)

    return attention_mask, position_ids


def resize_with_pad(
    img: Tensor,
    width: int,
    height: int,
    pad_value: float = -1,
) -> Tensor:
    """Resize image preserving aspect ratio with padding.
    
    Args:
        img: Input image [B, C, H, W].
        width: Target width.
        height: Target height.
        pad_value: Padding value (default -1 for SigLIP).
        
    Returns:
        Resized and padded image [B, C, height, width].
    """
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

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
    
    full_pad_masks = torch.zeros(batch_size, total_length, dtype=torch.bool, device=device)
    full_att_masks = torch.zeros(batch_size, total_length, dtype=prefix_att_masks.dtype, device=device)
    
    full_pad_masks[:, :prefix_length] = prefix_pad_masks
    full_att_masks[:, :prefix_length] = prefix_att_masks
    
    suffix_pad_tiled = suffix_pad_masks.unsqueeze(1).expand(-1, num_offsets, -1).reshape(batch_size, -1)
    suffix_att_tiled = suffix_att_masks.unsqueeze(1).expand(-1, num_offsets, -1).reshape(batch_size, -1)
    full_pad_masks[:, prefix_length:] = suffix_pad_tiled
    full_att_masks[:, prefix_length:] = suffix_att_tiled
    
    cumsum = torch.cumsum(full_att_masks, dim=1)
    
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    
    pad_2d_masks = full_pad_masks[:, None, :] * full_pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    
    suffix_positions = torch.arange(num_offsets * suffix_length, device=device)
    offset_ids = suffix_positions // suffix_length
    
    query_offset_ids = offset_ids.unsqueeze(1)
    key_offset_ids = offset_ids.unsqueeze(0)
    cross_offset_mask = (query_offset_ids == key_offset_ids)
    
    suffix_start = prefix_length
    att_2d_masks[:, suffix_start:, suffix_start:] = att_2d_masks[:, suffix_start:, suffix_start:] & cross_offset_mask
    
    offset_validity = offset_mask.unsqueeze(2).expand(-1, -1, suffix_length).reshape(batch_size, -1)
    
    att_2d_masks[:, suffix_start:, :] = att_2d_masks[:, suffix_start:, :] & offset_validity.unsqueeze(2)
    att_2d_masks[:, :, suffix_start:] = att_2d_masks[:, :, suffix_start:] & offset_validity.unsqueeze(1)
    
    position_ids = torch.zeros(batch_size, total_length, dtype=torch.long, device=device)
    
    prefix_pos = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
    position_ids[:, :prefix_length] = prefix_pos
    
    last_prefix_pos = prefix_pos[:, -1]
    
    suffix_pos_base = torch.cumsum(suffix_pad_masks.long(), dim=1)
    suffix_pos_tiled = suffix_pos_base.unsqueeze(1).expand(-1, num_offsets, -1).reshape(batch_size, -1)
    position_ids[:, prefix_length:] = last_prefix_pos[:, None] + suffix_pos_tiled
    
    attention_mask = torch.where(
        att_2d_masks,
        torch.zeros_like(att_2d_masks, dtype=dtype),
        torch.full_like(att_2d_masks, mask_value, dtype=dtype),
    )
    attention_mask = attention_mask.unsqueeze(1)

    return attention_mask, position_ids


def build_flashvla_attention_mask_and_position_ids(
    prefix_pad_masks: torch.Tensor,
    prefix_att_masks: torch.Tensor,
    suffix_pad_masks_per_offset: torch.Tensor,
    suffix_att_masks: torch.Tensor,
    num_offsets: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention mask for shared-observation training with per-offset suffix padding.

    Variant of :func:`build_shared_obs_attention_mask_and_position_ids` that
    accepts independent padding masks for each of the ``num_offsets`` suffix
    branches. Required for FlashVLA, where buffer config k
    (k=0..N-1) has k+1 real slots and N-k-1 padded slots — each config
    therefore needs a different per-token pad mask.
    """
    batch_size = prefix_pad_masks.shape[0]
    prefix_length = prefix_pad_masks.shape[1]
    suffix_length = suffix_pad_masks_per_offset.shape[2]
    total_length = prefix_length + suffix_length * num_offsets
    device = prefix_pad_masks.device
    mask_value = torch.finfo(dtype).min

    full_pad_masks = torch.zeros(batch_size, total_length, dtype=torch.bool, device=device)
    full_att_masks = torch.zeros(batch_size, total_length, dtype=prefix_att_masks.dtype, device=device)

    full_pad_masks[:, :prefix_length] = prefix_pad_masks
    full_att_masks[:, :prefix_length] = prefix_att_masks

    suffix_pad_tiled = suffix_pad_masks_per_offset.reshape(batch_size, -1)
    suffix_att_tiled = suffix_att_masks.unsqueeze(1).expand(-1, num_offsets, -1).reshape(batch_size, -1)

    full_pad_masks[:, prefix_length:] = suffix_pad_tiled
    full_att_masks[:, prefix_length:] = suffix_att_tiled

    cumsum = torch.cumsum(full_att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]

    pad_2d_masks = full_pad_masks[:, None, :] & full_pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks

    suffix_positions = torch.arange(num_offsets * suffix_length, device=device)
    offset_ids = suffix_positions // suffix_length
    cross_offset_mask = offset_ids.unsqueeze(1) == offset_ids.unsqueeze(0)
    suffix_start = prefix_length
    att_2d_masks[:, suffix_start:, suffix_start:] = (
        att_2d_masks[:, suffix_start:, suffix_start:] & cross_offset_mask
    )

    position_ids = torch.zeros(batch_size, total_length, dtype=torch.long, device=device)
    prefix_pos = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
    position_ids[:, :prefix_length] = prefix_pos
    last_prefix_pos = prefix_pos[:, -1]

    suffix_pos_per_offset = torch.cumsum(suffix_pad_masks_per_offset.long(), dim=2)
    suffix_pos_tiled = suffix_pos_per_offset.reshape(batch_size, -1)
    position_ids[:, prefix_length:] = last_prefix_pos[:, None] + suffix_pos_tiled

    attention_mask = torch.where(
        att_2d_masks,
        torch.zeros_like(att_2d_masks, dtype=dtype),
        torch.full_like(att_2d_masks, mask_value, dtype=dtype),
    )
    attention_mask = attention_mask.unsqueeze(1)

    return attention_mask, position_ids


