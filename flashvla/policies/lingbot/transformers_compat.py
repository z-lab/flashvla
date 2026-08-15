#!/usr/bin/env python

# Copyright 2025 FlashVLA team. All rights reserved.
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
"""Transformers-version shims for the vendored Qwen2.5-VL / Qwen2 modules.

``qwenvl_in_vla.py`` and ``modeling_lingbot_vla.py`` are vendored copies of
upstream LingBot-VLA, which were written against Transformers 4.x. LeRobot
0.5.1 pins ``transformers==5.3.0``, where a handful of the imported symbols
moved or disappeared and the Qwen2.5-VL config gained a nested ``text_config``.
This module isolates every one of those differences so the vendored files stay
byte-comparable with upstream apart from their import lines.
"""

from __future__ import annotations

import copy

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _UPSTREAM_ROPE_INIT_FUNCTIONS

try:  # Transformers < 5
    from transformers.cache_utils import SlidingWindowCache
except ImportError:  # Transformers >= 5 folded sliding windows into the layer types

    class SlidingWindowCache:
        """Never-matching placeholder for ``isinstance(cache, SlidingWindowCache)``.

        Qwen2.5-VL ships ``use_sliding_window=False``, so upstream's sliding
        window branches are dead for LingBot. Keeping a type that no cache can
        be an instance of preserves the vendored control flow exactly.
        """

        def __init__(self, *args, **kwargs):
            raise TypeError(
                "SlidingWindowCache was removed in Transformers 5 and is not used by LingBot-VLA"
            )


try:  # Transformers < 5
    from transformers.utils import LossKwargs
except ImportError:  # Renamed to TransformersKwargs in Transformers 5
    from transformers.utils import TransformersKwargs as LossKwargs


def _compute_default_rope_parameters(config, device=None, seq_len=None, **rope_kwargs):
    """Plain (non-scaled) RoPE inverse frequencies.

    Transformers 4.x exposed this as ``ROPE_INIT_FUNCTIONS["default"]``;
    Transformers 5 folded it into ``RotaryEmbeddingConfigMixin`` and left only
    the scaled variants in the registry. Both Qwen2.5-VL and the LingBot action
    expert resolve to ``rope_type="default"``, so the vendored rotary modules
    still need it under that key. Kept identical to the 4.x implementation.
    """
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    base = getattr(config, "rope_theta", None)
    if base is None:
        base = rope_parameters["rope_theta"]
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)

    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0  # (inv_freq, attention_scaling)


ROPE_INIT_FUNCTIONS = dict(_UPSTREAM_ROPE_INIT_FUNCTIONS)
ROPE_INIT_FUNCTIONS.setdefault("default", _compute_default_rope_parameters)


# Attributes the vendored Qwen2.5-VL text stack reads directly off the top-level
# config. Transformers 5 moved them under ``config.text_config``.
_TEXT_CONFIG_ATTRS = (
    "attention_dropout",
    "bos_token_id",
    "eos_token_id",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "layer_types",
    "max_position_embeddings",
    "max_window_layers",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "pad_token_id",
    "rms_norm_eps",
    "rope_scaling",
    "rope_theta",
    "sliding_window",
    "use_cache",
    "use_sliding_window",
    "vocab_size",
)


def flatten_qwen_vl_config(config: PretrainedConfig) -> PretrainedConfig:
    """Return a Qwen2.5-VL config whose text fields are readable at the top level.

    Transformers 5 nests the language-model fields under ``text_config``; the
    vendored ``Qwen2_5_VLModel`` reads ``config.hidden_size`` and friends
    directly. Copy the nested values up rather than rewriting ~1000 lines of
    vendored code, and keep ``text_config`` intact so Transformers' own
    ``get_text_config()`` machinery still works.

    Configs that are already flat (Transformers 4.x) are returned unchanged.
    """
    text_config = getattr(config, "text_config", None)
    if text_config is None or hasattr(config, "hidden_size"):
        return config

    flat = copy.deepcopy(config)
    for name in _TEXT_CONFIG_ATTRS:
        if hasattr(text_config, name):
            setattr(flat, name, getattr(text_config, name))
    return flat


__all__ = [
    "LossKwargs",
    "ROPE_INIT_FUNCTIONS",
    "SlidingWindowCache",
    "flatten_qwen_vl_config",
]
