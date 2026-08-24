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

import torch
from torch import nn

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embedding to input tensor.
    
    Splits x into two halves and applies rotation:
        x1' = x1 * cos - x2 * sin
        x2' = x2 * cos + x1 * sin
    
    Args:
        x: Input tensor [..., D].
        cos: Cosine values [..., D/2].
        sin: Sine values [..., D/2].
        
    Returns:
        Rotated tensor [..., D].
    """
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding layer.
    
    Precomputes cos/sin values for all positions up to max_position_embeddings.
    At forward time, looks up the appropriate values based on position indices.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        """Initialize RoPE layer.
        
        Args:
            head_size: Dimension per attention head.
            rotary_dim: Dimension to apply rotation (must equal head_size).
            max_position_embeddings: Maximum sequence length.
            base: Base for frequency computation (typically 10000).
        """
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size, "rotary_dim must equal head_size"
        
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        
        cos = freqs.cos()
        sin = freqs.sin()
        
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key.
        
        Args:
            positions: Position indices [B, L].
            query: Query tensor [B, H, L, D].
            key: Key tensor [B, H, L, D].
            
        Returns:
            Tuple of (rotated_query, rotated_key).
        """
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        
        return query, key
