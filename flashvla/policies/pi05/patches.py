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
"""RMSNorm with per-token adaRMS conditioning.

``FlashVLARMSNorm`` extends ``lerobot.policies.pi_gemma.PiGemmaRMSNorm`` to
support both per-sample conditioning ``[B, D]`` and per-token conditioning
``[B, L, D]`` (the latter is used by action streaming). The streaming policies
swap this class onto every ``PiGemmaRMSNorm`` instance in ``from_pretrained``
(in-place ``__class__`` assignment; ``weight`` / ``dense`` / ``eps`` /
``cond_dim`` are reused, no state_dict change).
"""

import torch

from lerobot.policies.pi_gemma import PiGemmaRMSNorm


class FlashVLARMSNorm(PiGemmaRMSNorm):
    """``PiGemmaRMSNorm`` with per-token adaRMS conditioning support."""

    def forward(self, x, cond=None):
        dtype = x.dtype
        normed_inputs = self._norm(x)

        if cond is None or self.dense is None:
            normed_inputs = normed_inputs * (1.0 + self.weight.float())
            return normed_inputs.to(dtype), None

        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dimension {self.cond_dim}, got {cond.shape[-1]}")

        modulation = self.dense(cond.to(self.dense.weight.dtype))
        if len(x.shape) == 3 and modulation.ndim == 2:
            modulation = modulation.unsqueeze(1)

        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)

        return normed_inputs.to(dtype), gate.to(dtype)

    def extra_repr(self):
        repr_str = f"dim={self.dim}, eps={self.eps}"
        if self.dense is not None:
            repr_str += f", adaptive=True, cond_dim={self.cond_dim}, per_token=True"
        return repr_str
