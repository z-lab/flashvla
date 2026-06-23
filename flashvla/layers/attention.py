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
"""Attention layer with KV cache.

Scaled dot-product attention with optional KV caching:
- First call with use_cache=True: initialize cache with K/V (prefix prefill)
- Subsequent calls: concatenate cached prefix K/V with new suffix K/V
"""

from typing import Optional

import torch
from torch import nn


class Attention(nn.Module):
    """Scaled dot-product attention with optional KV cache.

    Computes: Attention(Q, K, V) = softmax(Q @ K^T / scale) @ V

    The KV cache enables efficient inference by storing prefix K/V
    and reusing them across multiple forward passes.
    """

    def __init__(
        self,
        scale: float,
    ):
        """Initialize attention layer.

        Args:
            scale: Scaling factor for attention scores, typically 1/sqrt(head_dim).
        """
        super().__init__()
        self.scale = scale

        # KV cache buffers: [B, H, L_prefix, D]
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def reset_cache(self) -> None:
        """Clear KV cache. Call when starting a new sequence."""
        self.k_cache = None
        self.v_cache = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_attn_probs: bool = False,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention.

        Args:
            q: Query tensor [B, H_q, L_q, D].
            k: Key tensor [B, H_kv, L_k_new, D].
            v: Value tensor [B, H_kv, L_v_new, D].
            attention_mask: Additive mask [B, 1, L_q, L_k] or [B, H, L_q, L_k].
                           Use 0 for positions to attend, -inf for masked positions.
            use_cache: If True, use internal KV caching for prefix tokens.
            return_attn_probs: If True, return attention weights along with output.

        Returns:
            Output tensor [B, H_q, L_q, D], optionally with attention weights.
        """
        # Handle standard KV cache
        if use_cache:
            if self.k_cache is None:
                # First call: initialize cache with prefix K/V
                # Use copy_() to maintain tensor identity for CUDA graph compatibility
                self.k_cache = k.clone()
                self.v_cache = v.clone()
                k_full = k
                v_full = v
            else:
                # Subsequent calls: concatenate cached prefix with new suffix
                # Note: cache is not updated, always stores prefix only
                k_full = torch.cat([self.k_cache, k], dim=2)
                v_full = torch.cat([self.v_cache, v], dim=2)
        else:
            k_full = k
            v_full = v

        return self._forward_sdpa(q, k_full, v_full, attention_mask, return_attn_probs)

    def _forward_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        return_attn_probs: bool = False,
    ) -> torch.Tensor:
        """Scaled dot-product attention via PyTorch's memory-efficient backend.

        head_dim=256 exceeds Flash Attention's limit (128), so we force the
        memory-efficient backend which supports it and uses O(N) memory
        instead of materializing O(N²).

        The additive float mask (0/-inf) is converted to a boolean mask
        (True=attend, False=masked) because float masks trigger the math
        backend which materializes the full attention matrix in float32.
        """
        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask[:, None, :, :]

        if return_attn_probs:
            # Memory-efficient backend doesn't return attention weights.
            attn_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask.float()
            attn_weights = torch.softmax(attn_scores, dim=-1).to(q.dtype)
            out = torch.matmul(attn_weights, v)
            return out, attn_weights

        attn_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask.float()
        attn_weights = torch.softmax(attn_scores, dim=-1).to(q.dtype)
        out = torch.matmul(attn_weights, v)
        return out
