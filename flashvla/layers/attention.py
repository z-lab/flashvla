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

    Computes: Attention(Q, K, V) = softmax(Q @ K^T * scale) @ V

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
    ) -> torch.Tensor:
        """Compute scaled dot-product attention.

        Args:
            q: Query tensor [B, H_q, L_q, D].
            k: Key tensor [B, H_kv, L_k_new, D].
            v: Value tensor [B, H_kv, L_v_new, D].
            attention_mask: Additive mask [B, 1, L_q, L_k] or [B, H, L_q, L_k].
                           Use 0 for positions to attend, -inf for masked positions.
            use_cache: If True, use internal KV caching for prefix tokens.

        Returns:
            Output tensor [B, H_q, L_q, D].
        """
        if use_cache:
            if self.k_cache is None:
                self.k_cache = k.clone()
                self.v_cache = v.clone()
                k_full = k
                v_full = v
            else:
                k_full = torch.cat([self.k_cache, k], dim=2)
                v_full = torch.cat([self.v_cache, v], dim=2)
        else:
            k_full = k
            v_full = v

        return self._forward_sdpa(q, k_full, v_full, attention_mask)

    def _forward_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Scaled dot-product attention with explicit fp32 scores.

        The additive float mask (0/masked) is converted to a boolean mask
        (True=attend, False=masked). This keeps fully masked padding queries
        finite even if an fp32 mask was cast to bf16 by FSDP.

        FSDP mixed precision casts floating-point forward inputs. An fp32
        finfo.min mask therefore becomes -inf in bf16, and a fully masked
        padding row would produce NaNs in softmax. Applying the mask as a
        boolean to fp32 scores and explicitly zeroing masked probabilities
        avoids the NaN and gives fully masked queries a zero attention output.
        """
        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask[:, None, :, :]

        attn_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        allowed = None
        if attention_mask is not None:
            allowed = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
            attn_scores = attn_scores.masked_fill(
                ~allowed,
                torch.finfo(attn_scores.dtype).min,
            )
        attn_weights = torch.softmax(attn_scores, dim=-1)
        if allowed is not None:
            attn_weights = attn_weights.masked_fill(~allowed, 0.0)
        attn_weights = attn_weights.to(q.dtype)
        out = torch.matmul(attn_weights, v)
        return out
