# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
#
# Portions derived from LingBot-VLA, Copyright Robbyant Team.
# Source: https://github.com/Robbyant/lingbot-vla (commit 4eb34b7).
# Modified by the FlashVLA team. Licensed under Apache-2.0.

import einops
import torch
from collections import deque
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from torch import Tensor, nn
from typing import List, Optional, Tuple, Union, Callable, Dict, Any
from functools import partial
from transformers import (
    AutoConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING
from dataclasses import dataclass
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.cache_utils import Cache, StaticCache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel, ALL_ATTENTION_FUNCTIONS
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    can_return_tuple,
    is_torch_flex_attn_available,
)
from flashvla.policies.lingbot.transformers_compat import (
    ROPE_INIT_FUNCTIONS,
    LossKwargs,
    SlidingWindowCache,
    flatten_qwen_vl_config,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs, is_flash_attn_available
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.processing_utils import Unpack
import torch.distributed._tensor as dt
from flashvla.policies.lingbot.qwenvl_in_vla import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel, Qwen2_5_VLPreTrainedModel
from flashvla.policies.lingbot.lingbot_utils import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    resize_with_pad,
    sample_beta,
    apply_rope,
    our_eager_attention_forward,
)
from flashvla.policies.lingbot.flex_attention import flex_attention_forward
from flashvla.policies.lingbot.configuration_lingbot import LingBotConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Qwen2Config"


# torch.compile / dynamo cannot trace cuda.Event ops, so wrap them in
# dynamo-disabled functions: a compiled sample path still works under
# profile=True. Mirrors the pi05 baseline so the latency benchmark reports an
# identical encode/prefill/action breakdown.
@torch._dynamo.disable
def _ev_record(ev):
    ev.record()


@torch._dynamo.disable
def _ev_elapsed_ms(start, end):
    return start.elapsed_time(end)


@torch._dynamo.disable
def _cuda_sync():
    torch.cuda.synchronize()


def robotwin_raw_to_lingbot_layout(values: Tensor, config) -> Tensor:
    """Map normalized raw ALOHA joints to LingBot's padded typed layout.

    Raw:      [left_arm6, left_grip, right_arm6, right_grip] (14)
    LingBot:  [arm12, arm_pad2, effector2, remaining_pad] (max dim)
    """
    if not getattr(config, "robotwin_feature_layout", False):
        target_dim = max(values.shape[-1], config.max_action_dim)
        return F.pad(values, (0, target_dim - values.shape[-1]))

    arm_dim = int(config.robotwin_arm_dim)
    arm_padded_dim = int(config.robotwin_arm_padded_dim)
    effector_dim = int(config.robotwin_effector_dim)
    raw_dim = arm_dim + effector_dim
    if values.shape[-1] != raw_dim:
        raise ValueError(
            f"RoboTwin LingBot layout expects {raw_dim} raw joints, got {values.shape[-1]}"
        )
    if arm_dim != 12 or effector_dim != 2:
        raise ValueError("Only the RoboTwin ALOHA 12-arm + 2-effector layout is supported")

    arms = torch.cat([values[..., :6], values[..., 7:13]], dim=-1)
    effectors = torch.stack([values[..., 6], values[..., 13]], dim=-1)
    arm_pad = values.new_zeros(*values.shape[:-1], arm_padded_dim - arm_dim)
    typed = torch.cat([arms, arm_pad, effectors], dim=-1)
    target_dim = config.max_state_dim if values.ndim == 2 else config.max_action_dim
    return F.pad(typed, (0, target_dim - typed.shape[-1]))


def lingbot_layout_to_robotwin_raw(values: Tensor, config) -> Tensor:
    """Inverse of :func:`robotwin_raw_to_lingbot_layout` for model outputs."""
    if not getattr(config, "robotwin_feature_layout", False):
        return values[..., : config.action_dim]

    arm_padded_dim = int(config.robotwin_arm_padded_dim)
    arms = values[..., : int(config.robotwin_arm_dim)]
    effectors = values[
        ..., arm_padded_dim : arm_padded_dim + int(config.robotwin_effector_dim)
    ]
    return torch.cat(
        [arms[..., :6], effectors[..., :1], arms[..., 6:12], effectors[..., 1:2]],
        dim=-1,
    )


def lingbot_valid_action_mask(config, device: torch.device) -> Tensor:
    """Return the model-coordinate dimensions supervised by upstream LingBot."""
    mask = torch.zeros(config.max_action_dim, dtype=torch.bool, device=device)
    if getattr(config, "robotwin_feature_layout", False):
        mask[: int(config.robotwin_arm_dim)] = True
        start = int(config.robotwin_arm_padded_dim)
        mask[start : start + int(config.robotwin_effector_dim)] = True
    else:
        mask[: config.action_dim] = True
    return mask


def lingbot_flow_matching_loss(
    target_velocity: Tensor,
    predicted_velocity: Tensor,
    loss_type: str,
) -> Tensor:
    """Return an elementwise LingBot flow-matching loss.

    FSDP bf16 keeps the sampled actions/velocity target in fp32 while the
    action expert predicts bf16 velocities. ``mse_loss`` does not support that
    mixed-dtype backward path, so compute its numerically sensitive square in
    fp32 and let autograd cast the gradient back through ``Tensor.float()``.
    The released RoboTwin L1 path is intentionally left unchanged.
    """

    if loss_type == "fm":
        return F.mse_loss(
            predicted_velocity.float(),
            target_velocity.float(),
            reduction="none",
        )
    if loss_type == "L1_fm":
        return F.l1_loss(target_velocity, predicted_velocity, reduction="none")
    raise ValueError(f"Unsupported LingBot flow-matching loss_type={loss_type!r}")


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class FixQwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        FixQwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.norm_qkv:
            self.q_layernorm = Qwen2RMSNorm(self.self_attn.head_dim, eps=config.rms_norm_eps)
            self.k_layernorm = Qwen2RMSNorm(self.self_attn.head_dim, eps=config.rms_norm_eps)

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        att_output: Optional[torch.Tensor] = None,
        start: Optional[int] = 0,
        end: Optional[int] = 0,
        compute_kqv: bool = False,
        norm_qkv: bool = False,
        old_adanorm: bool = False,
        output_atten: bool = False,
        ada_cond: Optional[torch.Tensor] = None,
        gate: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if compute_kqv:
            if ada_cond is not None:
                if old_adanorm:
                    hidden_states = self.input_layernorm(hidden_states, ada_cond)
                    gate = None
                else:
                    hidden_states, gate = self.input_layernorm(hidden_states, ada_cond)
            else:
                hidden_states = self.input_layernorm(hidden_states)
                gate = None
            hidden_shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)

            query_state = self.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = self.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)
            if norm_qkv:
                query_state = self.q_layernorm(query_state)
                key_state = self.k_layernorm(key_state)

            return query_state, key_state, value_state, gate

        elif output_atten:
            if att_output.dtype != self.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(self.self_attn.o_proj.weight.dtype)
            out_emb = self.self_attn.o_proj(att_output[:, start:end])

            # first residual
            if gate is not None:
                out_emb = out_emb * gate + hidden_states
            else:
                out_emb += hidden_states
            after_first_residual = out_emb.clone()
            if ada_cond is not None:
                if old_adanorm:
                    out_emb = self.post_attention_layernorm(out_emb, ada_cond)
                    after_gate= None
                else:
                    out_emb, after_gate = self.post_attention_layernorm(out_emb, ada_cond)
            else:
                out_emb = self.post_attention_layernorm(out_emb)
                after_gate = None
            out_emb = self.mlp(out_emb)

            # second residual
            if after_gate is not None:
                out_emb = out_emb * after_gate + after_first_residual
            else:
                out_emb += after_first_residual

            return out_emb

        else:
            raise ValueError(f"Invaild Operation compute_kqv={compute_kqv} and output_atten={output_atten} with Qwen2DecoderLayer in LingBot-VLA")

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


QWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

QWEN2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""

@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config, eval=False):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = FixQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        if eval:
            self._init_weights = lambda module: None
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2Config,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...
class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    # Transformers 5 expects {tied_key: source_key} instead of 4.x's flat list.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, eval):
        super().__init__(config)
        self.model = Qwen2Model(config, eval)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class QwenvlWithExpertConfig(PretrainedConfig):
    model_type = "QwenvlWithExpertModel"
    sub_configs = {"qwenvl_config": AutoConfig, "qwen_expert_config": AutoConfig}

    def __init__(
        self,
        qwenvl_config: dict | None = None,
        qwen_expert_config: dict | None = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        vocab_size: int = 257152,
        use_lm_head: bool = False,
        attention_implementation: str = "eager",
        vision_attention_implementation: str = "sdpa",
        tokenizer_path: str | None = None,
        enable_expert_vision: bool = False,
        expert_vision_type: str | None = None,
        use_cache: bool = True,
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        self.vision_attention_implementation = vision_attention_implementation
        self.tokenizer_path = tokenizer_path
        self.enable_expert_vision = enable_expert_vision
        self.expert_vision_type = expert_vision_type
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.use_lm_head = use_lm_head
        if qwenvl_config is None:
            self.qwenvl_config = CONFIG_MAPPING["qwen2_5_vl"](
                attention_dropout=0.0,
                bos_token_id=151643,
                eos_token_id=151645,
                vision_start_token_id=151652,
                vision_end_token_id=151653,
                vision_token_id=151654,
                image_token_id=151655,
                video_token_id=151656,
                hidden_act="silu",
                hidden_size=2048,
                initializer_range=0.02,
                intermediate_size=11008,
                max_position_embeddings=128000,
                max_window_layers=70,
                model_type="qwen2_5_vl",
                num_attention_heads=16,
                num_hidden_layers=36,
                num_key_value_heads=2,
                rms_norm_eps=1e-06,
                rope_theta=1000000.0,
                sliding_window=32768,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                transformers_version="4.41.2",
                use_cache=use_cache,
                use_sliding_window=False,
                vision_config={
                    "depth": 32,
                    "hidden_act": "silu",
                    "hidden_size": 1280,
                    "intermediate_size": 3420,
                    "num_heads": 16,
                    "in_chans": 3,
                    "out_hidden_size": 2048,
                    "patch_size": 14,
                    "spatial_merge_size": 2,
                    "spatial_patch_size": 14,
                    "window_size": 112,
                    "fullatt_block_indexes": [
                        7,
                        15,
                        23,
                        31
                    ],
                    "tokens_per_second": 2,
                    "temporal_patch_size": 2
                },
                rope_scaling={
                                "type": "mrope",
                                "mrope_section": [
                                    16,
                                    24,
                                    24
                                ]
                                },
                vocab_size=151936,
            )
        elif isinstance(qwenvl_config, dict):
            if "model_type" not in qwenvl_config:
                qwenvl_config["model_type"] = "qwen2_5_vl"

            cfg_cls = CONFIG_MAPPING[qwenvl_config["model_type"]]
            self.qwenvl_config = cfg_cls(**qwenvl_config)
        elif isinstance(qwenvl_config, PretrainedConfig):
            self.qwenvl_config = qwenvl_config
        else:
            raise TypeError(
                "qwenvl_config must be a dict, PretrainedConfig, or None; got "
                f"{type(qwenvl_config).__name__}"
            )

        if qwen_expert_config is None:
            self.qwen_expert_config = CONFIG_MAPPING["qwen2"](
                attention_dropout=0.0,
                bos_token_id=151643,
                eos_token_id=151645,
                hidden_act="silu",
                hidden_size=768,
                head_dim=128,
                initializer_range=0.02,
                intermediate_size=2752,
                max_position_embeddings=32768,
                max_window_layers=21,
                model_type="qwen2",
                num_attention_heads=16,
                num_hidden_layers=36,
                num_key_value_heads=2,
                rms_norm_eps=1e-06,
                rope_theta=1000000.0,
                sliding_window=32768,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                transformers_version="4.43.1",
                use_cache=use_cache,
                use_sliding_window=False,
                vocab_size=151936,
            )
        elif isinstance(qwen_expert_config, dict):
            if "model_type" not in qwen_expert_config:
                qwen_expert_config["model_type"] = "qwen2"

            cfg_cls = CONFIG_MAPPING[qwen_expert_config["model_type"]]
            self.qwen_expert_config = cfg_cls(**qwen_expert_config)
        elif isinstance(qwen_expert_config, PretrainedConfig):
            self.qwen_expert_config = qwen_expert_config
        else:
            raise TypeError(
                "qwen_expert_config must be a dict, PretrainedConfig, or None; got "
                f"{type(qwen_expert_config).__name__}"
            )

        super().__init__(**kwargs)

        # PretrainedConfig is not a dataclass and never calls __post_init__.
        # Validate here so these constraints are active for both direct and
        # deserialized nested configs.
        if self.train_expert_only and not self.freeze_vision_encoder:
            raise ValueError(
                "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
            )

        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). Expected 'eager', 'fa2' or 'flex'."
            )
        if self.vision_attention_implementation not in [
            "eager",
            "sdpa",
            "flash_attention_2",
        ]:
            raise ValueError(
                "Wrong value provided for vision_attention_implementation "
                f"({self.vision_attention_implementation})."
            )

class OldAdaRMSNorm(nn.Module):
    def __init__(self, hidden_size, cond_dim, eps=1e-6):
        """
        AdaRMSNorm: RMSNorm + FiLM
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.gamma = nn.Linear(cond_dim, hidden_size)
        self.beta = nn.Linear(cond_dim, hidden_size)

        # DiT style init: gamma.weight=0, gamma.bias=1; beta.weight=0, beta.bias=0
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.constant_(module.weight, 0.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, hidden_states, cond):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        hidden_states = self.weight * hidden_states
        gamma = self.gamma(cond)
        beta  = self.beta(cond)
        if gamma.ndim == 2:  # cond is [B, D] → broadcast over seq_len
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        # if ndim == 3: cond is [B, seq_len, D] → already per-token
        hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        return hidden_states.to(input_dtype)

    # def extra_repr(self):
    #     return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class AdaRMSNorm(nn.Module):
    def __init__(self, hidden_size, cond_dim, split_gate_liner, no_split_gate_liner, eps=1e-6):
        """
        AdaRMSNorm: RMSNorm + FiLM
        """
        super().__init__()
        if not (split_gate_liner or no_split_gate_liner):
            self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.use_gate = split_gate_liner or no_split_gate_liner
        if not no_split_gate_liner:
            self.gamma = nn.Linear(cond_dim, hidden_size)
            self.beta = nn.Linear(cond_dim, hidden_size)
            if self.use_gate:
                self.gate = nn.Linear(cond_dim, hidden_size)
                nn.init.zeros_(self.gate.weight)
                nn.init.zeros_(self.gate.bias)

            # DiT style init: gamma.weight=0, gamma.bias=1; beta.weight=0, beta.bias=0
            nn.init.zeros_(self.gamma.weight)
            nn.init.zeros_(self.gamma.bias)
            nn.init.zeros_(self.beta.weight)
            nn.init.zeros_(self.beta.bias)
        else:
            self.gamma_beta_gate = nn.Linear(cond_dim, hidden_size * 3, bias=True)
            nn.init.zeros_(self.gamma_beta_gate.weight)
            nn.init.zeros_(self.gamma_beta_gate.bias)
        self.no_split_gate_liner = no_split_gate_liner
        self.split_gate_liner = split_gate_liner

    def forward(self, hidden_states, cond):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if not (self.split_gate_liner or self.no_split_gate_liner):
            hidden_states = self.weight * hidden_states
        if not self.no_split_gate_liner:
            gamma = self.gamma(cond)
            beta = self.beta(cond)
            if gamma.ndim == 2:
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
            if self.use_gate:
                gate = self.gate(cond)
                if gate.ndim == 2:
                    gate = gate.unsqueeze(1)
            else:
                gate = None
            hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        else:
            modulation = self.gamma_beta_gate(cond)
            if modulation.ndim == 2 and hidden_states.ndim == 3:
                modulation = modulation.unsqueeze(1)
            gamma, beta, gate = torch.chunk(modulation, 3, dim=-1)
            hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        return hidden_states.to(input_dtype), gate

    # def extra_repr(self):
    #     return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class FixAdaRMSNorm(nn.Module):
    def __init__(self, hidden_size, cond_dim, split_gate_liner, no_split_gate_liner, eps=1e-6):
        """
        AdaRMSNorm: RMSNorm + FiLM
        """
        super().__init__()
        if not (split_gate_liner or no_split_gate_liner):
            self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.use_gate = split_gate_liner or no_split_gate_liner
        if not no_split_gate_liner:
            self.gamma = nn.Linear(cond_dim, hidden_size)
            self.beta = nn.Linear(cond_dim, hidden_size)
            if self.use_gate:
                self.gate = nn.Linear(cond_dim, hidden_size)
                nn.init.zeros_(self.gate.weight)
                nn.init.zeros_(self.gate.bias)

            # DiT style init: gamma.weight=0, gamma.bias=1; beta.weight=0, beta.bias=0
            nn.init.zeros_(self.gamma.weight)
            nn.init.zeros_(self.gamma.bias)
            nn.init.zeros_(self.beta.weight)
            nn.init.zeros_(self.beta.bias)
        else:
            self.gamma_beta_gate = nn.Linear(cond_dim, hidden_size * 3, bias=True)
            nn.init.zeros_(self.gamma_beta_gate.weight)
            nn.init.zeros_(self.gamma_beta_gate.bias)
        self.no_split_gate_liner = no_split_gate_liner
        self.split_gate_liner = split_gate_liner

    def forward(self, hidden_states, cond):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if not (self.split_gate_liner or self.no_split_gate_liner):
            hidden_states = self.weight * hidden_states
        if not self.no_split_gate_liner:
            gamma = self.gamma(cond)
            beta = self.beta(cond)
            if gamma.ndim == 2:
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
            if self.use_gate:
                gate = self.gate(cond)
                if gate.ndim == 2:
                    gate = gate.unsqueeze(1)
            else:
                gate = None
            hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        else:
            modulation = self.gamma_beta_gate(cond)
            if modulation.ndim == 2 and hidden_states.ndim == 3:
                modulation = modulation.unsqueeze(1)
            gamma, beta, gate = torch.chunk(modulation, 3, dim=-1)
            hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        return hidden_states.to(input_dtype), gate

# HACK: show directly use this norm during initialization
def replace_lnorm_with_adanorm(module, hidden_size, cond_dim, split_gate_liner, no_split_gate_liner, final_norm_adanorm, old_adanorm):
    if old_adanorm:
        for name, child in module.named_children():
            if isinstance(child, Qwen2RMSNorm):
                if 'q_layernorm' not in name and 'k_layernorm' not in name:
                    setattr(module, name, OldAdaRMSNorm(hidden_size, cond_dim))
            else:
                replace_lnorm_with_adanorm(child, hidden_size, cond_dim, split_gate_liner, no_split_gate_liner, final_norm_adanorm, old_adanorm)
    else:
        for name, child in module.named_children():
            if final_norm_adanorm:
                if isinstance(child, Qwen2RMSNorm):
                    if 'q_layernorm' not in name and 'k_layernorm' not in name:
                        setattr(module, name, AdaRMSNorm(hidden_size, cond_dim, split_gate_liner, no_split_gate_liner))
                elif isinstance(child, FixQwen2RMSNorm):
                    if 'q_layernorm' not in name and 'k_layernorm' not in name:
                        setattr(module, name, FixAdaRMSNorm(hidden_size, cond_dim, split_gate_liner, no_split_gate_liner))
                else:
                    replace_lnorm_with_adanorm(child, hidden_size, cond_dim, split_gate_liner, no_split_gate_liner, final_norm_adanorm, old_adanorm)
            else:
                if isinstance(child, Qwen2RMSNorm):
                    if 'q_layernorm' not in name and 'k_layernorm' not in name:
                        setattr(module, name, AdaRMSNorm(hidden_size, cond_dim, split_gate_liner, no_split_gate_liner))
                else:
                    replace_lnorm_with_adanorm(child, hidden_size, cond_dim, split_gate_liner, no_split_gate_liner, final_norm_adanorm, old_adanorm)

class QwenvlWithExpertModel(PreTrainedModel):
    config_class = QwenvlWithExpertConfig

    def __init__(self, config: QwenvlWithExpertConfig, eval=False):
        super().__init__(config=config)
        self.config = config
        # Transformers 5 nests the Qwen2.5-VL language fields under
        # ``text_config``; the vendored VLM below reads them at the top level.
        vlm_config = flatten_qwen_vl_config(
            AutoConfig.from_pretrained(self.config.tokenizer_path)
        )
        vlm_config.vision_config._attn_implementation = (
            self.config.vision_attention_implementation
        )
        vlm_config.vision_config.initializer_range = 0.02
        vlm_config.norm_qkv = self.config.norm_qkv
        if self.config.vocab_size != 0 and self.config.vocab_size != 257152 and vlm_config.vocab_size != self.config.vocab_size:
            vlm_config.vocab_size = self.config.vocab_size
        self.qwenvl = Qwen2_5_VLForConditionalGeneration._from_config(vlm_config)
        if self.config.use_lm_head:
            self.qwenvl.tie_weights()
        self.config.qwen_expert_config.norm_qkv = self.config.norm_qkv
        self.qwen_expert = Qwen2ForCausalLM._from_config(self.config.qwen_expert_config, eval=eval)

        # Vision rotary/window metadata depends only on the fixed patch grid.
        # Cache it after the first call so compiled steady streaming contains
        # no Python ``.item()``/``.tolist()`` CUDA synchronizations.
        self._vision_grid_key = None
        self._vision_rotary_pos_emb = None
        self._vision_window_index = None
        self._vision_cu_window_seqlens = None
        self._vision_cu_seqlens = None

        if getattr(self.config, 'adanorm_time', False):
            replace_lnorm_with_adanorm(self.qwen_expert, self.config.qwen_expert_config.hidden_size, self.config.qwen_expert_config.hidden_size, config.split_gate_liner, config.no_split_gate_liner, config.final_norm_adanorm, config.old_adanorm)
        # Remove unused embed_tokens
        del self.qwen_expert.model.embed_tokens
        if self.config.enable_expert_vision:
            # The upstream DINOv3 expert-vision tower is not part of this
            # release; LingBotFlashVLAConfig rejects enable_expert_vision=True.
            raise NotImplementedError(
                "enable_expert_vision is not supported in the open-source FlashVLA release"
            )
        self.attention_interface = self.get_attention_interface()

        self.set_requires_grad()

    def set_requires_grad(self):
        """sets the requires_grad attribute of the model parameters based on the configuration.
        If `freeze_vision_encoder` is True, the vision tower parameters are frozen.
        If `train_expert_only` is True, the entire Qwenvl model is frozen.
        """
        if self.config.freeze_vision_encoder:
            self.qwenvl.visual.eval()
            for params in self.qwenvl.visual.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.qwenvl.eval()
            for params in self.qwenvl.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.qwenvl.visual.eval()
        if self.config.train_expert_only:
            self.qwenvl.eval()

    @torch._dynamo.disable
    @torch.no_grad()
    def _cache_vision_grid_metadata(self, image_grid_thw, grid_key) -> None:
        """Create persistent vision metadata outside compiled CUDA graphs.

        ``preprcess_grid_thw`` contains host-visible shape logic and returns
        CUDA tensors.  If it runs while a streaming path is being captured,
        saving those graph-owned outputs on ``self`` leaves them backed by a
        reusable CUDAGraph output slot.  A later replay then overwrites the
        supposedly persistent cache.  This disabled boundary plus explicit
        clones gives the cache ordinary, stable storage.
        """
        metadata = self.qwenvl.visual.preprcess_grid_thw(image_grid_thw)
        (
            self._vision_rotary_pos_emb,
            self._vision_window_index,
            self._vision_cu_window_seqlens,
            self._vision_cu_seqlens,
        ) = tuple(
            value.detach().to(device=image_grid_thw.device).clone()
            for value in metadata
        )
        self._vision_grid_key = grid_key

    @torch._dynamo.disable
    @torch.no_grad()
    def prepare_vision_metadata(self, images: torch.Tensor) -> None:
        """Pre-fill fixed-grid Qwen vision metadata before graph capture."""
        if images.ndim == 4:  # [B, cameras, patches, patch_dim]
            num_images = int(images.shape[0] * images.shape[1])
            num_patches = int(images.shape[2])
        elif images.ndim == 3:  # [images, patches, patch_dim]
            num_images = int(images.shape[0])
            num_patches = int(images.shape[1])
        else:
            raise ValueError(
                "LingBot vision metadata expects patchified images with rank "
                f"3 or 4, got shape={tuple(images.shape)}"
            )

        grid_size = int(num_patches**0.5)
        if grid_size * grid_size != num_patches:
            raise ValueError(
                "LingBot vision patch grid must be square, got "
                f"num_patches={num_patches}"
            )
        grid_key = (
            num_images,
            num_patches,
            images.device.type,
            images.device.index,
        )
        if self._vision_grid_key == grid_key:
            return

        image_grid_thw = torch.tensor(
            [[1, grid_size, grid_size]],
            dtype=torch.long,
            device=images.device,
        ).expand(num_images, -1).contiguous()
        self._cache_vision_grid_metadata(image_grid_thw, grid_key)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        grid_key = (
            image_grid_thw.shape[0],
            pixel_values.shape[1],
            image_grid_thw.device.type,
            image_grid_thw.device.index,
        )
        if self._vision_grid_key != grid_key:
            self._cache_vision_grid_metadata(image_grid_thw, grid_key)

        image_embeds = self.qwenvl.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            rotary_pos_emb=self._vision_rotary_pos_emb,
            window_index=self._vision_window_index,
            cu_window_seqlens=self._vision_cu_window_seqlens,
            cu_seqlens=self._vision_cu_seqlens,
        )
        # FlashVLA preprocessing uses one fixed grid for every image, so the
        # merged token count is uniform. Reshape avoids the upstream CUDA
        # ``split_sizes.tolist()`` host synchronization after vision forward.
        num_images = image_grid_thw.shape[0]
        return image_embeds.reshape(num_images, image_embeds.shape[0] // num_images, -1)

    def embed_image(self, image: torch.Tensor, patch_size=14, temporal_patch_size=2):
        h = w = int(image.shape[1] ** 0.5)
        image_grid_thw = torch.tensor([[1, h, w]]*image.shape[0], device=image.device)
        image_embeds = self.get_image_features(image, image_grid_thw=image_grid_thw)
        return image_embeds
        # return torch.randn(72, 64, 2048).to(device=image.device, dtype=torch.bfloat16)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.qwenvl.model.embed_tokens(tokens)

    def handle_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
    ):
        if use_cache:
            if past_key_values is None:
                past_key_values = {}

            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                key_states = torch.cat(
                    [past_key_values[layer_idx]["key_states"], key_states], dim=1
                )
                value_states = torch.cat(
                    [past_key_values[layer_idx]["value_states"], value_states],
                    dim=1,
                )
        return key_states, value_states, past_key_values

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        vlm_position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
        ada_cond: List[torch.FloatTensor] = None,
        use_ki: bool = False,
        norm_qkv: bool = False,
    ):
        """
        Args:
            attention_mask (Optional[torch.Tensor], optional):
                Attention mask with shape (b, seq_len, seq_len). Defaults to None.
            position_ids (Optional[torch.LongTensor], optional):
                Position indices for applying RoPE. Defaults to None.
            past_key_values (Optional[Union[List[torch.FloatTensor], Cache]], optional):
                Optional kv cache. Defaults to None.
            inputs_embeds (List[torch.FloatTensor], optional):
                Input embeddings. Defaults to None.
            use_cache (Optional[bool], optional):
                Whether to use kv cache. Defaults to None.
            fill_kv_cache (Optional[bool], optional):
                Whether to return kv tensors in this forward pass as cache. Defaults to None.

        Returns:
            outputs_embeds (torch.Tensor): Output embeddings.
            past_key_values (Optional[Union[List[torch.FloatTensor], Cache]]):
                Optional kv cache.
        """
        models = [self.qwenvl.model, self.qwen_expert.model]

        # RMSNorm
        num_layers = self.qwenvl.config.num_hidden_layers # 36
        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            gates = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                if i == 1: # For action expert
                    query_state, key_state, value_state, gate = models[i].layers[layer_idx](hidden_states, compute_kqv=True, ada_cond = ada_cond, norm_qkv=norm_qkv, old_adanorm=self.config.old_adanorm)
                else:   # For VLM
                    query_state, key_state, value_state = models[i].layers[layer_idx](hidden_states, compute_kqv=True, norm_qkv=norm_qkv)
                    gate = None
                    if use_ki:
                        query_state, key_state, value_state = query_state.detach(), key_state.detach(), value_state.detach()

                if query_state.dtype != torch.float32:
                    query_state, key_state, value_state = query_state.to(torch.float32), key_state.to(torch.float32), value_state.to(torch.float32)
                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)
                gates.append(gate)

            # B,L,H,D with L sequence length (img, lang, state, action), H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            key_states, value_states, past_key_values = self.handle_kv_cache(
                key_states,
                value_states,
                layer_idx,
                past_key_values=past_key_values,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
            )
            att_output = self.attention_interface(query_states, key_states, value_states, attention_mask)

            # first part of att_output is prefix (up to sequence length, [:, 0:prefix_seq_len])
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is not None:
                    end = start + hidden_states.shape[1]
                    if i == 1:
                        out_emb = models[i].layers[layer_idx](hidden_states, att_output, start, end, output_atten=True, ada_cond = ada_cond, gate=(gates[0] if len(gates) == 1 else gates[i]), old_adanorm=self.config.old_adanorm)
                    else:
                        out_emb = models[i].layers[layer_idx](hidden_states, att_output, start, end, output_atten=True)
                    outputs_embeds.append(out_emb)
                    start = end
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                if self.config.final_norm_adanorm:
                    if i == 1:
                        out_emb, _ = models[i].norm(hidden_states, ada_cond)
                    else:
                        out_emb = models[i].norm(hidden_states)
                else:
                    out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        if self.config.attention_implementation == "fa2":
            raise NotImplementedError("FA2 is not implemented (yet)")
        elif self.config.attention_implementation == "flex":
            attention_interface = flex_attention_forward
        elif self.config.attention_implementation == "eager":
            attention_interface = our_eager_attention_forward
        elif self.config.attention_implementation == "xformer":
            # attention_interface = xformer_attention_forward
            raise NotImplementedError("Xformer attention is not implemented (yet)")
        else:
            raise ValueError(
                f"Invalid attention implementation: {self.config.attention_implementation}. "
                "Expected one of ['fa2', 'flex', 'eager', 'xformer']."
            )
        return attention_interface

class LingbotVlaPolicy(PreTrainedPolicy):
    config_class = LingBotConfig
    name = "lingbot"
    _no_split_modules = ["Qwen2DecoderLayer", "FixQwen2RMSNorm", "FixAdaRMSNorm"]
    fsdp_wrap_class_names = (
        "Qwen2_5_VLPatchMerger",
        "Qwen2_5_VLVisionBlock",
        "Qwen2_5_VLDecoderLayer",
        "Qwen2DecoderLayer",
        "Embedding",
    )
    fsdp_wrap_name_suffixes = ()
    fsdp_fp32_class_names = ()
    fsdp_fp32_name_suffixes = ()
    fsdp_fp32_output_name_suffixes = ()

    def __init__(self, config, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        tokenizer_path = getattr(config, "tokenizer_path", None)
        if not tokenizer_path:
            raise ValueError("config.tokenizer_path must be set")
        self.model = FlowMatching(config, eval=not self.training)

        if not getattr(self.config, "use_lm_head", False):
            del self.model.qwenvl_with_expert.qwenvl.lm_head
        del self.model.qwenvl_with_expert.qwen_expert.lm_head

        self.reset()
        torch.set_float32_matmul_precision("high")

    def prepare_for_fsdp(self, *, compute_dtype: torch.dtype) -> None:
        """Validate LingBot's module boundaries before policy-owned FSDP2 wrapping."""
        if getattr(self.config, "use_lm_head", False):
            raise ValueError(
                "LingBot FSDP2 requires use_lm_head=False because the tied VLM "
                "embedding/lm_head parameter would cross two communication groups."
            )
        joint_model = self.model.qwenvl_with_expert
        if hasattr(joint_model.qwenvl, "lm_head") or hasattr(
            joint_model.qwen_expert,
            "lm_head",
        ):
            raise RuntimeError("LingBot FSDP2 requires both unused lm_head modules to be removed")
        if getattr(self.config, "compile_model", False):
            raise ValueError("Compile LingBot only for inference, after FSDP2 training/export.")
        self._fsdp_compute_dtype = compute_dtype

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        strict: bool = True,
        **kwargs,
    ):
        """Load FlashVLA or sharded upstream-compatible LingBot weights.

        Upstream LingBot releases keep essential runtime fields outside
        ``config.json``, so callers loading one of those checkpoints directly
        must pass an explicit :class:`LingBotConfig`. FlashVLA-saved
        checkpoints reconstruct it from their own registered ``config.json``.
        """
        del resume_download, proxies
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        if not isinstance(config, LingBotConfig):
            raise TypeError(
                "LingbotVlaPolicy requires an explicit LingBotConfig: upstream "
                "LingBot checkpoints keep their runtime fields outside config.json."
            )

        kwargs.pop("dataset_stats", None)
        instance = cls(config=config, **kwargs)
        from flashvla.policies.lingbot.checkpoint import (
            load_lingbot_weights,
            move_lingbot_for_runtime,
        )

        load_lingbot_weights(
            instance,
            pretrained_name_or_path,
            strict=strict,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
        )
        move_lingbot_for_runtime(instance)
        if getattr(config, "compile_model", False):
            instance.model.qwenvl_with_expert = torch.compile(
                instance.model.qwenvl_with_expert,
                mode=getattr(config, "compile_mode", "max-autotune"),
            )
        instance.eval()
        return instance

    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, profile: bool = False
    ) -> Tensor:
        """Predict one normalized action chunk from a LeRobot-style batch.

        With ``profile=True`` this returns ``(actions, profile_results)``, where
        ``profile_results`` is the encode/prefill/action cuda.Event breakdown
        from ``sample_actions``. The layout conversion and ``.float()`` below
        deliberately stay outside the timed region — they are not model compute
        — matching the pi05 baseline.
        """
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions, profile_results = self.model.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            expert_imgs=None,
            noise=noise,
            profile=profile,
        )
        actions = actions[:, : self.config.n_action_steps]
        actions = lingbot_layout_to_robotwin_raw(actions, self.config)
        if profile:
            return actions.float(), profile_results
        return actions.float()

    @torch.no_grad()
    def select_action(
        self, observation: dict[str, Tensor], noise: Tensor | None = None
    ):
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(observation, noise=noise)
            self._action_queue.extend(
                actions.transpose(0, 1)[: self.config.n_action_steps]
            )
        return self._action_queue.popleft()

    @staticmethod
    def _patchify_image(img: Tensor, patch_size: int = 14, temporal_patch_size: int = 2, merge_size: int = 2) -> Tensor:
        """Convert [B, C, H, W] image (normalized) to Qwen2.5-VL patch format [B, num_patches, patch_dim].

        Replicates the patchification logic of Qwen2_5_VLImageProcessor:
        temporal duplicate → spatial split with merge-size interleaving → flatten.
        """
        B, C, H, W = img.shape
        # Temporal duplicate: [B, C, H, W] -> [B, T, C, H, W]
        img = img.unsqueeze(1).expand(-1, temporal_patch_size, -1, -1, -1)

        grid_t = 1  # single frame → 1 temporal grid cell
        grid_h = H // patch_size
        grid_w = W // patch_size
        gh_m = grid_h // merge_size
        gw_m = grid_w // merge_size

        # [B, T, C, gh_m, ms, pH, gw_m, ms, pW]
        img = img.reshape(B, grid_t, temporal_patch_size, C, gh_m, merge_size, patch_size, gw_m, merge_size, patch_size)
        # -> [B, grid_t, gh_m, gw_m, ms_h, ms_w, C, T, pH, pW]
        img = img.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        # -> [B, num_patches, patch_dim]
        num_patches = grid_t * grid_h * grid_w
        patch_dim = C * temporal_patch_size * patch_size * patch_size
        return img.reshape(B, num_patches, patch_dim)

    def prepare_images(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Extract images from batch and convert to Qwen2.5-VL patch format.

        Returns:
            images: [B, n_cameras, num_patches, patch_dim] stacked patches
            img_masks: [B, n_cameras] boolean masks
        """
        patches_list: list[Tensor] = []
        img_masks: list[Tensor] = []

        configured_img_keys = list(self.config.image_features)
        if getattr(self.config, "robotwin_feature_layout", False):
            preferred = [
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            ]
            missing_configured = [
                key for key in preferred if key not in configured_img_keys
            ]
            unexpected = [key for key in configured_img_keys if key not in preferred]
            if missing_configured or unexpected:
                raise ValueError(
                    "Released RoboTwin LingBot weights require exactly the high, "
                    "left-wrist, and right-wrist RGB features; "
                    f"missing={missing_configured}, unexpected={unexpected}"
                )
            configured_img_keys = preferred
            absent_required = [key for key in preferred if key not in batch]
            if absent_required:
                raise ValueError(
                    "RoboTwin LingBot requires fixed camera slots in "
                    f"high/left/right order; missing {absent_required}"
                )

        present_img_keys = [key for key in configured_img_keys if key in batch]
        missing_img_keys = [key for key in configured_img_keys if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                "All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # CLIP normalization (matching Qwen2.5-VL processor)
        _CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        _CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

        for key in present_img_keys:
            img = batch[key]  # [B, C, H, W] in [0, 1]
            # PyTorch's antialiased bilinear resize is not implemented for
            # bfloat16 (CPU or CUDA in the supported stack).  Upstream also
            # performs image preprocessing in fp32 and lets the Qwen patch
            # embedding cast to its weight dtype.  Keep that boundary here so
            # bf16 policy inference does not fail before the vision tower.
            img = img.to(dtype=torch.float32)
            if getattr(self.config, "image_resize_mode", "stretch") == "stretch":
                img = F.interpolate(
                    img,
                    size=self.config.image_resolution,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            else:
                img = resize_with_pad(img, *self.config.image_resolution, pad_value=0)
            # Normalize with CLIP mean/std
            mean = _CLIP_MEAN.to(device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
            std = _CLIP_STD.to(device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
            img = (img - mean) / std
            # Patchify to Qwen2.5-VL format: [B, num_patches, patch_dim]
            patches_list.append(self._patchify_image(img))
            bsz = img.shape[0]
            img_masks.append(torch.ones(bsz, dtype=torch.bool, device=img.device))

        for _ in range(min(len(missing_img_keys), self.config.empty_cameras)):
            patches_list.append(torch.zeros_like(patches_list[-1]))
            img_masks.append(torch.zeros(bsz, dtype=torch.bool, device=img.device))

        # Stack cameras: [B, n_cameras, num_patches, patch_dim]
        images = torch.stack(patches_list, dim=1)
        # [B, n_cameras]
        img_masks = torch.stack(img_masks, dim=1)
        return images, img_masks

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        """Pad state to max_state_dim."""
        state = batch[OBS_STATE]
        if getattr(self.config, "robotwin_feature_layout", False):
            return robotwin_raw_to_lingbot_layout(state, self.config)
        if state.shape[-1] < self.config.max_state_dim:
            state = F.pad(state, (0, self.config.max_state_dim - state.shape[-1]))
        return state

    def prepare_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Pad action to max_action_dim."""
        actions = batch[ACTION]
        if getattr(self.config, "robotwin_feature_layout", False):
            return robotwin_raw_to_lingbot_layout(actions, self.config)
        if actions.shape[-1] < self.config.max_action_dim:
            actions = F.pad(actions, (0, self.config.max_action_dim - actions.shape[-1]))
        return actions

    def forward(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, time: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Training forward pass.

        Args:
            batch: Training batch from preprocessor.
            noise: Optional noise for reproducibility.
            time: Optional timestep for reproducibility.

        Returns:
            Tuple of (loss, loss_dict).
        """
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        loss_dict: dict[str, Tensor | float] = {}

        losses = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions,
            expert_imgs=None, noise=noise, time=time,
            loss_type=self.config.loss_type,
        )

        valid_dims = lingbot_valid_action_mask(self.config, losses.device)
        losses = losses[:, :, valid_dims]
        # Select rather than multiply padded timesteps so short episode tails
        # do not dilute the mean with zeros.
        if actions_is_pad is not None:
            losses = losses[~actions_is_pad]

        loss_vla = losses.mean()
        loss_dict["loss"] = loss_vla.item()
        return loss_vla, loss_dict

class FlowMatching(nn.Module):
    def __init__(self, config, eval):
        super().__init__()
        self.config = config

        # qwenvl with action expert
        qwenvl_with_export_config = QwenvlWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            vocab_size=getattr(self.config,"vocab_size", 0),
            use_lm_head=getattr(self.config,"use_lm_head", False),
            attention_implementation=self.config.attention_implementation,
            vision_attention_implementation=self.config.vision_attention_implementation,
            tokenizer_path=self.config.tokenizer_path,
            enable_expert_vision=self.config.enable_expert_vision,
            expert_vision_type=self.config.expert_vision_type,
            use_cache=getattr(self.config, "use_cache", True),
        )
        qwenvl_with_export_config.adanorm_time = getattr(config, "adanorm_time", False)
        qwenvl_with_export_config.split_gate_liner = getattr(config, "split_gate_liner", False)
        qwenvl_with_export_config.no_split_gate_liner = getattr(config, "nosplit_gate_liner", False)
        qwenvl_with_export_config.separate_time_proj = getattr(config, "separate_time_proj", False)
        qwenvl_with_export_config.old_adanorm = getattr(config, "old_adanorm", False)
        qwenvl_with_export_config.final_norm_adanorm = getattr(config, "final_norm_adanorm", False)
        qwenvl_with_export_config.norm_qkv = getattr(config, "norm_qkv", False)
        self.qwenvl_with_expert = QwenvlWithExpertModel(
            qwenvl_with_export_config, eval
        )
        self.config.proj_width = qwenvl_with_export_config.qwen_expert_config.hidden_size
        self.config.initializer_range = getattr(qwenvl_with_export_config.qwen_expert_config, "initializer_range", None)
        # projection layers
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(
            self.config.max_action_dim, self.config.proj_width
        )
        self.action_out_proj = nn.Linear(
            self.config.proj_width, self.config.max_action_dim
        )
        if getattr(config, "separate_time_proj", False):
            self.time_mlp_in = nn.Linear(self.config.proj_width, self.config.proj_width)
            self.time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)
        else:
            self.action_time_mlp_in = nn.Linear(
                self.config.proj_width * 2, self.config.proj_width
            )
            self.action_time_mlp_out = nn.Linear(
                self.config.proj_width, self.config.proj_width
            )
        # The upstream depth-alignment auxiliary head is not part of this
        # release; LingBotFlashVLAConfig rejects a non-empty align_params.
        self.use_depth_align = False

        self.set_requires_grad()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            bsize,
            device,
        )
        time = (
            time_beta * self.config.time_sampling_scale
            + self.config.time_sampling_offset
        )
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, vlm_causal
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsize = images.shape[0]
        device = images.device
        dtype = images.dtype

        # embed image
        if images.ndim == 5:
            images = einops.rearrange(images, "b n c h w -> (b n) c h w")
        elif images.ndim == 4:
            images = einops.rearrange(images, "b n l d -> (b n) l d")
        elif images.ndim == 3: # For inference bs=1
            bsize = 1
        img_emb = self.qwenvl_with_expert.embed_image(images)
        num_patch = img_emb.shape[1]
        img_emb = einops.rearrange(img_emb, "(b n) l d -> b (n l) d", b=bsize) # bsize = 24
        num_img_embs = img_emb.shape[1]
        if img_masks.ndim ==1: # For inference bs=1
            img_masks = img_masks.unsqueeze(0)
        img_masks = einops.repeat(img_masks, "b n -> b (n l)", l=num_patch)

        # embed language
        lang_emb = self.qwenvl_with_expert.embed_language_tokens(lang_tokens)
        num_lang_embs = lang_emb.shape[1]

        # assemble embeddings
        embs = torch.cat([img_emb, lang_emb], dim=1)
        pad_masks = torch.cat([img_masks, lang_masks], dim=1)

        # (see `make_att_2d_masks` to understand why zeros means bidirection)
        if not vlm_causal:
            att_masks = torch.zeros(
                (img_emb.size(0), num_img_embs + num_lang_embs), device=device, dtype=torch.bool
            )
        else:
            att_masks = torch.ones(
                (img_emb.size(0), num_img_embs + num_lang_embs), device=device, dtype=torch.bool
            )
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep, expert_imgs=None):
        bsize = state.shape[0] # state_bs = img_bs
        device = state.device
        # LeRobot processors intentionally keep normalized observations in
        # fp32, while released LingBot inference commonly runs the projection
        # layers in bf16. Match the projection boundary explicitly (the
        # streaming suffix does the same) instead of relying on autocast.
        proj_dtype = self.state_proj.weight.dtype
        state = state.to(dtype=proj_dtype)
        noisy_actions = noisy_actions.to(dtype=proj_dtype)
        dtype = proj_dtype
        # embed state
        state_emb = self.state_proj(state) # torch.Size([state_bs, 1024])

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding( # 1, 1024
            timestep, # torch.Size([1]))
            self.config.proj_width, # 1024
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb_ori = time_emb

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions) # torch.Size([1, state_bs*50, 1024])
        if getattr(self.config, "separate_time_proj", False):
            time_emb = self.time_mlp_in(time_emb)
            time_emb = F.silu(time_emb)
            time_emb_ori = F.silu(self.time_mlp_out(time_emb)) # [1, 1024]
            action_time_emb = action_emb
        else:
            time_emb = einops.repeat(time_emb, "b d -> b n d", n=action_emb.shape[1]) # [1, 1024] -> [1, state_bs*50, 1024]
            action_time_emb = torch.cat([action_emb, time_emb], dim=-1) # [1, state_bs*50, 2048]

            action_time_emb = self.action_time_mlp_in(action_time_emb)
            action_time_emb = F.silu(action_time_emb)  # swish == silu
            action_time_emb = self.action_time_mlp_out(action_time_emb) # [1, state_bs*50, 1024]
        action_time_dim = action_time_emb.shape[1]

        if expert_imgs is not None:
            if expert_imgs.ndim == 5:
                expert_imgs = einops.rearrange(expert_imgs, "b n c h w -> (b n) c h w")
            elif expert_imgs.ndim == 4:
                bsize=1
            expert_img_emb = self.qwenvl_with_expert.expert_visual.forward_features(expert_imgs)["x_norm_clstoken"].unsqueeze(1)
            expert_img_emb = self.qwenvl_with_expert.expert_visual_mlp(expert_img_emb)
            expert_img_emb = einops.rearrange(expert_img_emb, "(b n) l d -> b (n l) d", b=bsize) # bsize = 24
            embs = torch.cat([expert_img_emb, state_emb[:, None], action_time_emb], dim=1)
            num_expert_img_emb = expert_img_emb.shape[1]
            pad_masks = torch.ones(
                (bsize, action_time_dim + 1 + num_expert_img_emb), device=device, dtype=torch.bool
            )
            att_masks = torch.zeros(
                (bsize, action_time_dim + 1 + num_expert_img_emb), device=device, dtype=torch.bool
            )
            att_masks[:, [0, num_expert_img_emb, num_expert_img_emb + 1]] = True

        else:
            embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)
            pad_masks = torch.ones(
                (bsize, action_time_dim + 1), device=device, dtype=torch.bool
            )

            # Set attention masks for suffix tokens so that prefix tokens cannot attend to suffix tokens.
            # And state token cannot attend action tokens.
            # Action tokens use a bidirectional attention.
            att_masks = torch.zeros(
                (bsize, action_time_dim + 1), device=device, dtype=torch.bool
            )
            att_masks[:, :2] = True

        return time_emb_ori, embs, pad_masks, att_masks

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        expert_imgs,
        noise=None,
        time=None,
        vlm_causal=False,
        loss_type='fm',
        use_ki=False,
        norm_qkv=False
    ) -> Tensor:
        dtype = state.dtype
        device = state.device
        if noise is None:
            # actions_shape = (
            #     bsize,
            #     self.config.n_action_steps, # 50
            #     self.config.max_action_dim, # 32
            # )
            noise = torch.randn(actions.shape, device=device, dtype=dtype)

        if time is None:
            time = self.sample_time(actions.size(0), device).to(dtype)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, vlm_causal
        ) # 1,bs_img*(768+48),2048  1,bs_img*(768+48)  1,bs_img*(768+48)
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, time, expert_imgs
        ) # [1, state_bs*(50+1), 1024], [1, state_bs*(50+1)], [1, state_bs*(50+1)]   state_bs=bs_img

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # pad_masks = pad_masks.reshape(state.size(0), -1)
        # att_masks = att_masks.reshape(state.size(0), -1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        vlm_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # prefix_embs = prefix_embs.reshape(state.size(0), -1, prefix_embs.size(-1))
        # suffix_embs = suffix_embs.reshape(state.size(0), -1, suffix_embs.size(-1))
        (outputs_embeds, suffix_out), _ = self.qwenvl_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            vlm_position_ids=vlm_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            # Full-sequence training consumes prefix and suffix together; a KV
            # cache is only required by the suffix-only inference loop.
            use_cache=False,
            fill_kv_cache=False,
            ada_cond = time_embs if getattr(self.config, 'adanorm_time', False) else None,
            use_ki=use_ki,
            norm_qkv=norm_qkv
        )
        # Training may use the full chunk even when deployment later executes
        # a shorter sub-chunk. Match the velocity target's actual horizon.
        suffix_out = suffix_out[:, -actions.shape[1] :]
        if suffix_out.dtype != self.action_out_proj.weight.dtype:
            suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out)
        # u_t = u_t.reshape(images.size(0), -1, u_t.size(-1))
        losses = lingbot_flow_matching_loss(u_t, v_t, loss_type)

        return losses

    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, expert_imgs=None, vlm_causal=False, noise=None, profile=False
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors).

        With ``profile=True`` the second return value is a per-stage cuda.Event
        breakdown ``{"encode", "prefill", "action", "total"}`` in ms: encode =
        VLM image/language embed, prefill = prefix KV-cache fill, action = the
        full ``num_steps`` Euler denoise loop. Mirrors the pi05 baseline so the
        latency benchmark reports the same three-segment split. Without
        ``profile`` the second value is None; callers unpack ``actions, _``.
        """
        if not self.config.use_cache:
            raise ValueError(
                "LingBot suffix-only denoising inference requires use_cache=True"
            )
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        _evs = (
            [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            if profile
            else None
        )

        if noise is None:
            actions_shape = (
                bsize,
                self.config.n_action_steps,
                self.config.max_action_dim,
            )
            noise = torch.randn(actions_shape, device=device, dtype=dtype)

        # Encode: embed VLM images + language into the prefix sequence.
        if profile:
            _ev_record(_evs[0])
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, vlm_causal
        )
        if profile:
            _ev_record(_evs[1])
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks) # bs, prefix_len, prefix_len
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Prefill: compute and cache the prefix (image + language) KV.
        _, past_key_values = self.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        # Under torch.compile with CUDA graphs (max-autotune) the prefill KV
        # cache is an OUTPUT of the cudagraph static pool. The denoise loop below
        # re-invokes the SAME compiled ``qwenvl_with_expert``, whose replay
        # overwrites that pool and corrupts the cached prefix K/V ("accessing
        # tensor output of CUDAGraphs that has been overwritten"). Clone the
        # cache off the pool so every replay reads stable tensors. Numerically
        # transparent, and only taken when compiled — the streaming policy
        # compiles prefill plus one denoise step as a single callable, so its
        # cache never crosses a replay boundary.
        if getattr(self.config, "compile_model", False) and isinstance(past_key_values, dict):
            past_key_values = {
                layer_idx: {name: t.clone() for name, t in layer.items()}
                for layer_idx, layer in past_key_values.items()
            }
        if profile:
            _ev_record(_evs[2])

        # Action: num_steps Euler denoise steps over the cached prefix KV.
        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=dtype, device=device)
        count = 0
        while time >= -dt / 2:
            count += 1
            expanded_time = time.expand(bsize)

            v_t = self.predict_velocity(
                state, prefix_pad_masks, past_key_values, x_t, expert_imgs, expanded_time
            )

            # Euler step
            x_t += dt * v_t
            time += dt
        if profile:
            _ev_record(_evs[3])
            _cuda_sync()
            profile_results = {
                "encode": _ev_elapsed_ms(_evs[0], _evs[1]),
                "prefill": _ev_elapsed_ms(_evs[1], _evs[2]),
                "action": _ev_elapsed_ms(_evs[2], _evs[3]),
                "total": _ev_elapsed_ms(_evs[0], _evs[3]),
            }
        else:
            profile_results = None
        return x_t, profile_results

    def predict_velocity(self, state, prefix_pad_masks, past_key_values, x_t, expert_imgs, timestep):
        """predict velocity at time t using the suffix model."""
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, timestep, expert_imgs
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2) # bs, suffix_len, prefix_len+suffix_len

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.qwenvl_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            ada_cond = time_embs if getattr(self.config, 'adanorm_time', False) else None,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -x_t.shape[1] :]
        v_t = self.action_out_proj(suffix_out)
        return v_t

ModelClass = LingbotVlaPolicy

__all__ = ["LingbotVlaPolicy", "Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VLModel", "Qwen2ForCausalLM", "Qwen2_5_VLPreTrainedModel"]
