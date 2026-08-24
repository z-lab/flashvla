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
import torch.nn.functional as F
from torch import nn


class QKVLinear(nn.Module):
    """Fused Query-Key-Value projection for multi-head attention.
    
    Instead of three separate linear layers (q_proj, k_proj, v_proj),
    this combines them into a single projection:
    
        [Q, K, V] = x @ W^T  where W = [W_q; W_k; W_v]
    """
    
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        """Initialize fused QKV projection.
        
        Args:
            hidden_size: Input embedding dimension.
            head_size: Dimension per attention head.
            total_num_heads: Number of query heads.
            total_num_kv_heads: Number of key/value heads (for GQA). 
                               Defaults to total_num_heads.
            bias: Whether to include bias terms.
        """
        super().__init__()
        total_num_kv_heads = total_num_kv_heads or total_num_heads

        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_kv_heads

        output_size = (self.num_heads + 2 * self.num_kv_heads) * self.head_size
        self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input to Q, K, V tensors.
        
        Args:
            x: Input tensor [B, L, hidden_size].
            
        Returns:
            q: Query tensor [B, num_heads, L, head_size].
            k: Key tensor [B, num_kv_heads, L, head_size].
            v: Value tensor [B, num_kv_heads, L, head_size].
        """
        if x.dim() != 3:
            raise ValueError(f"QKVLinear expects 3D input [B, L, D], got {x.shape}")

        bsz, seqlen, _ = x.shape
        
        out = F.linear(x, self.weight, self.bias)

        total_heads = self.num_heads + 2 * self.num_kv_heads
        out = out.view(bsz, seqlen, total_heads, self.head_size)
        out = out.permute(0, 2, 1, 3).contiguous()

        q = out[:, : self.num_heads]
        k = out[:, self.num_heads : self.num_heads + self.num_kv_heads]
        v = out[:, self.num_heads + self.num_kv_heads :]
        
        return q, k, v


class MergedColumnLinear(nn.Module):
    """Fused column-parallel linear layer for MLP.
    
    Combines multiple linear projections (e.g., gate and up in SwiGLU)
    into a single matrix multiplication:
    
        [gate, up] = x @ W^T  where W = [W_gate; W_up]
    
    The outputs are split along the last dimension.
    """
    
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        """Initialize merged linear layer.
        
        Args:
            input_size: Input dimension.
            output_sizes: List of output dimensions for each split.
            bias: Whether to include bias terms.
        """
        super().__init__()
        self.input_size = input_size
        self.output_sizes = list(output_sizes)
        
        output_size = sum(self.output_sizes)
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Project and split input.
        
        Args:
            x: Input tensor [..., input_size].
            
        Returns:
            Tuple of tensors, one for each output_size.
        """
        out = F.linear(x, self.weight, self.bias)
        return torch.split(out, self.output_sizes, dim=-1)
