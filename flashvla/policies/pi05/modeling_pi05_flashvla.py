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
"""PI0.5 FlashVLA: Padded Cold Start with Shared Observation Training.

Uses a fixed chunk_size=10, buffer of N=5 slots, and padding-based warmup
instead of full ODE cold start. During cold start the buffer gradually fills
with real slots while padded slots are masked. Training uses shared observation
(all N buffer configs trained simultaneously per observation).

Buffer states during cold start (0=cleanest, 4=noisiest, P=padding):
  Step 0: [4, P, P, P, P]  -> 1 real slot
  Step 1: [3, 4, P, P, P]  -> 2 real slots
  Step 2: [2, 3, 4, P, P]  -> 3 real slots
  Step 3: [1, 2, 3, 4, P]  -> 4 real slots
  Step 4: [0, 1, 2, 3, 4]  -> full buffer -> extract first chunk
  Step 5+: steady state streaming (single denoise step per call)
"""
import builtins
import logging
import math
import os
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from lerobot.policies.pi_gemma import (
    PiGemmaForCausalLM as GemmaForCausalLM,
    PaliGemmaForConditionalGenerationWithPiGemma as PaliGemmaForConditionalGeneration,
    _gated_residual,
)
from lerobot.configs.policies import PreTrainedConfig, T
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from flashvla.policies.pi05.configuration_pi05 import PI05FlashVLAConfig
from flashvla.policies.pi05.utils import (
    ColdStartStats,
    build_attention_mask_and_position_ids,
    compute_cold_start_action,
    create_sinusoidal_pos_embedding_for_blocks,
    load_cold_start_stats,
    pad_vector,
    resize_with_pad,
)
from flashvla.layers.attention import Attention
from flashvla.layers.linear import QKVLinear, MergedColumnLinear
from flashvla.layers.rope import RotaryEmbedding


# torch.compile / dynamo can't trace cuda.Event ops; wrap them in a
# disabled function so compiled sample_actions still works under profile=True.
@torch._dynamo.disable
def _ev_record(ev):
    ev.record()


@torch._dynamo.disable
def _ev_elapsed_ms(start, end):
    return start.elapsed_time(end)


@torch._dynamo.disable
def _cuda_sync():
    torch.cuda.synchronize()

logger = logging.getLogger(__name__)

def _rope_theta(text_cfg) -> float:
    """RoPE base frequency (config.rope_theta or config.rope_parameters)."""
    theta = getattr(text_cfg, "rope_theta", None)
    if theta is None:
        theta = (getattr(text_cfg, "rope_parameters", None) or {}).get("rope_theta", 10000.0)
    return theta



# ============================================================
# Reused classes (identical structure to baseline modeling_pi05.py)
# ============================================================

class PI05PrefixEmbedder(nn.Module):
    """Embed images and language tokens into prefix sequence."""

    def __init__(self, config: PI05FlashVLAConfig, vlm: PaliGemmaForConditionalGeneration):
        super().__init__()
        self.config = config
        # Held in a list so it is not registered as a submodule.
        self._paligemma_model = [vlm.model]
        self.lang_embedder = vlm.language_model.embed_tokens

    def forward(self, images, img_masks, tokens, masks):
        embs = []
        pad_masks = []
        att_masks = []
        bsz = tokens.shape[0]

        for img, img_mask in zip(images, img_masks, strict=True):
            pg = self._paligemma_model[0]
            # Run the SigLIP tower directly and use the unscaled projector
            # output. Cast the float32 input embeddings to the encoder dtype.
            vt = pg.vision_tower.vision_model
            hidden = vt.embeddings(img)
            enc_dtype = vt.encoder.layers[0].self_attn.q_proj.weight.dtype
            hidden = hidden.to(enc_dtype)
            hidden = vt.encoder(inputs_embeds=hidden).last_hidden_state
            img_feats = vt.post_layernorm(hidden)
            img_emb = pg.multi_modal_projector(img_feats)
            bsz, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsz, num_img_embs))
            att_masks += [0] * num_img_embs

        lang_emb = self.lang_embedder(tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        embs.append(lang_emb)
        pad_masks.append(masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        bsz = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsz, len(att_masks))

        return embs, pad_masks, att_masks


class PI05Attention(nn.Module):
    """Joint attention over VLM and action expert hidden states."""

    def __init__(self, config, vlm_attention, action_expert_attention):
        super().__init__()
        self.config = config
        self.vlm_attention = vlm_attention
        self.action_expert_attention = action_expert_attention
        text_cfg = config.vlm_config.text_config

        self.rotary_emb = RotaryEmbedding(
            head_size=text_cfg.head_dim,
            rotary_dim=text_cfg.head_dim,
            max_position_embeddings=text_cfg.max_position_embeddings,
            base=_rope_theta(text_cfg),
        )
        self.attn = Attention(scale=vlm_attention.scaling)
        self.num_heads = text_cfg.num_attention_heads
        self.head_dim = text_cfg.head_dim

    def forward(self, hidden_states, attention_mask, position_ids, use_cache: bool = False):
        attns = [self.vlm_attention, self.action_expert_attention]

        q_states = []
        k_states = []
        v_states = []
        for attn, hs in zip(attns, hidden_states):
            if hs is None or hs.shape[1] == 0:
                continue
            if hasattr(attn, "qkv_proj"):
                q, k, v = attn.qkv_proj(hs)
            else:
                bsz, seqlen, _ = hs.shape
                q = attn.q_proj(hs).view(bsz, seqlen, -1, self.head_dim).permute(0, 2, 1, 3).contiguous()
                k = attn.k_proj(hs).view(bsz, seqlen, -1, self.head_dim).permute(0, 2, 1, 3).contiguous()
                v = attn.v_proj(hs).view(bsz, seqlen, -1, self.head_dim).permute(0, 2, 1, 3).contiguous()
            q_states.append(q)
            k_states.append(k)
            v_states.append(v)

        q = torch.cat(q_states, dim=2)
        k = torch.cat(k_states, dim=2)
        v = torch.cat(v_states, dim=2)

        q, k = self.rotary_emb(position_ids, q, k)

        bsz = q.shape[0]
        attn_outputs = self.attn(q, k, v, attention_mask, use_cache=use_cache)

        attn_outputs = attn_outputs.transpose(1, 2).contiguous()
        attn_outputs = attn_outputs.view(bsz, -1, self.num_heads * self.head_dim)

        outputs = []
        start_pos = 0
        for attn, hs in zip(attns, hidden_states):
            if hs is None or hs.shape[1] == 0:
                outputs.append(None)
                continue
            end_pos = start_pos + hs.shape[1]
            out_emb = attn.o_proj(attn_outputs[:, start_pos:end_pos])
            outputs.append(out_emb)
            start_pos = end_pos
        return outputs


class PI05MLP(nn.Module):
    """Joint MLP for VLM and action expert."""

    def __init__(self, config, vlm_mlp, action_expert_mlp):
        super().__init__()
        self.config = config
        self.vlm_mlp = vlm_mlp
        self.action_expert_mlp = action_expert_mlp

    def forward(self, hidden_states):
        mlps = [self.vlm_mlp, self.action_expert_mlp]
        outputs = []
        for mlp, hs in zip(mlps, hidden_states):
            if hs is None or hs.shape[1] == 0:
                outputs.append(hs)
                continue
            if hasattr(mlp, "gate_up_proj"):
                gate, up = mlp.gate_up_proj(hs)
            else:
                gate = mlp.gate_proj(hs)
                up = mlp.up_proj(hs)
            x = mlp.act_fn(gate) * up
            x = mlp.down_proj(x)
            outputs.append(x)
        return outputs


# ============================================================
# Suffix embedder and layer classes
# ============================================================

class PI05SuffixEmbedder(nn.Module):
    """Embed noisy actions with per-token time conditioning for FlashVLA.

    Key difference from baseline: takes per-token time [B, N*C] instead of
    scalar [B]. The adarms_cond is [B, N*C, D] (per-token), not [B, D].
    Padded slots are zeroed out in embeddings.
    """

    def __init__(self, config: PI05FlashVLAConfig):
        super().__init__()
        self.config = config
        self.N = config.num_buffer_slots
        self.C = config.chunk_size

        # Action projection
        self.action_in_proj = nn.Linear(config.max_action_dim, config.action_expert_config.hidden_size)

        # Time MLP for flow matching timestep
        self.time_mlp_in = nn.Linear(config.action_expert_config.hidden_size, config.action_expert_config.hidden_size)
        self.time_mlp_out = nn.Linear(config.action_expert_config.hidden_size, config.action_expert_config.hidden_size)

        # Optional state conditioning MLP
        if config.state_cond:
            self.state_proj = nn.Linear(config.max_state_dim, config.action_expert_config.hidden_size)
            self.state_mlp_in = nn.Linear(config.action_expert_config.hidden_size, config.action_expert_config.hidden_size)
            self.state_mlp_out = nn.Linear(config.action_expert_config.hidden_size, config.action_expert_config.hidden_size)
            nn.init.zeros_(self.state_mlp_out.weight)
            nn.init.zeros_(self.state_mlp_out.bias)

    def forward(self, state, noisy_actions, time, padding_mask=None):
        """Embed noisy actions with time and state conditioning.

        ``time`` is per-token ``[B, L]`` (or ``[B, L, 1]``); time_mlp runs on all
        L positions.

        Args:
            state: Robot state ``[B, state_dim]``.
            noisy_actions: Noisy action sequence x_t ``[B, L, action_dim]``.
            time: Per-token times ``[B, L]`` (or ``[B, L, 1]``).
            padding_mask: ``[B, L]`` boolean per-token. True = real, False = padded.

        Returns:
            suffix_embs: ``[B, L, D]``.
            pad_masks: ``[B, L]``.
            att_masks: ``[B, L]``.
            adarms_cond: ``[B, L, D]``.
        """
        C = self.C
        bsz, L = noisy_actions.shape[:2]
        num_slots = L // C
        device = noisy_actions.device

        if time.dim() == 3:
            time = time.squeeze(-1)

        time_emb = create_sinusoidal_pos_embedding_for_blocks(
            time,
            self.config.action_expert_config.hidden_size,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=device,
        )
        time_emb = time_emb.to(dtype=noisy_actions.dtype)

        time_emb = self.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = F.silu(time_emb)

        adarms_cond = time_emb

        suffix_embs = self.action_in_proj(noisy_actions)  # [B, L, D]

        if self.config.state_cond:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)
            state_emb = self.state_proj(state)
            state_emb = self.state_mlp_in(state_emb)
            state_emb = F.silu(state_emb)
            state_emb = self.state_mlp_out(state_emb)
            state_emb = F.silu(state_emb)
            adarms_cond = adarms_cond + state_emb.unsqueeze(1)

        # Block-causal: mark first token of each slot as boundary
        att_row = torch.zeros(num_slots, C, dtype=suffix_embs.dtype, device=device)
        att_row[:, 0] = 1
        att_masks = att_row.flatten(0, 1).unsqueeze(0).expand(bsz, -1)  # [B, L]

        pad_masks = torch.ones(bsz, L, dtype=torch.bool, device=device)

        if padding_mask is not None:
            pad_masks = padding_mask
            suffix_embs = suffix_embs * padding_mask.unsqueeze(-1).to(suffix_embs.dtype)
            adarms_cond = adarms_cond * padding_mask.unsqueeze(-1).to(adarms_cond.dtype)

        return suffix_embs, pad_masks, att_masks, adarms_cond


class PI05ModelLayer(nn.Module):
    """Single transformer layer for PI0.5."""

    def __init__(self, config, vlm_layer, action_expert_layer):
        super().__init__()
        self.config = config
        self.input_layernorm = nn.ModuleList([vlm_layer.input_layernorm, action_expert_layer.input_layernorm])
        self.post_attention_layernorm = nn.ModuleList(
            [vlm_layer.post_attention_layernorm, action_expert_layer.post_attention_layernorm]
        )
        self.self_attn = PI05Attention(config, vlm_layer.self_attn, action_expert_layer.self_attn)
        self.mlp = PI05MLP(config, vlm_layer.mlp, action_expert_layer.mlp)

    def forward(self, hidden_states, attention_mask, position_ids, conds, use_cache: bool = False):
        residuals = [hs.clone() if hs is not None else None for hs in hidden_states]
        gates = []
        for i in range(len(hidden_states)):
            hs = hidden_states[i]
            if hs is None:
                gates.append(None)
                continue
            hidden_states[i], gate = self.input_layernorm[i](hs, conds[i])
            gates.append(gate)

        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids, use_cache=use_cache)

        for i in range(len(hidden_states)):
            hs = hidden_states[i]
            if hs is None:
                continue
            hidden_states[i] = _gated_residual(residuals[i], hs, gates[i])

        residuals = [hs.clone() if hs is not None else None for hs in hidden_states]
        gates = []
        for i in range(len(hidden_states)):
            hs = hidden_states[i]
            if hs is None:
                gates.append(None)
                continue
            hidden_states[i], gate = self.post_attention_layernorm[i](hs, conds[i])
            gates.append(gate)

        hidden_states = self.mlp(hidden_states)

        for i in range(len(hidden_states)):
            hs = hidden_states[i]
            if hs is None:
                continue
            hidden_states[i] = _gated_residual(residuals[i], hs, gates[i])
        return hidden_states


class PI05ModelLayer(PI05ModelLayer):
    """Transformer layer with per-token adaRMS conditioning for FlashVLA.

    Overrides forward_shared_observation to handle per-token adaRMS cond
    [B, num_offsets, suffix_length, D] instead of per-offset cond [B, num_offsets, D].
    The regular forward method inherits from PI05ModelLayer and works because the
    patched GemmaRMSNorm handles both 2D and 3D cond.
    """

    def forward(self, hidden_states, attention_mask, position_ids, conds=None, use_cache: bool = False, *, suffix_adarms_conds=None, num_offsets: int | None = None, suffix_length: int | None = None):
        if suffix_adarms_conds is not None:
            if num_offsets is None or suffix_length is None:
                raise ValueError("num_offsets and suffix_length are required when suffix_adarms_conds is provided.")
            return self.forward_shared_observation(hidden_states, attention_mask, position_ids, suffix_adarms_conds, num_offsets, suffix_length, use_cache=use_cache)
        if conds is None:
            raise ValueError("conds is required for the regular PI05ModelLayer forward path.")
        return super().forward(hidden_states, attention_mask, position_ids, conds, use_cache=use_cache)

    def forward_shared_observation(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        suffix_adarms_conds,  # [B, num_offsets, suffix_length, D] -- per-token
        num_offsets: int,
        suffix_length: int,
        use_cache: bool = False,
    ):
        batch_size = hidden_states[0].shape[0]

        # ============ Pre-attention layernorm ============
        residuals = [hs.clone() if hs is not None else None for hs in hidden_states]
        gates = []

        # Prefix: VLM layernorm without conditioning
        prefix = hidden_states[0]
        prefix_normed, prefix_gate = self.input_layernorm[0](prefix, cond=None)
        hidden_states[0] = prefix_normed
        gates.append(prefix_gate)

        # Suffix: per-token conditioning (parallelized)
        # Reshape: [B, num_offsets * suffix_length, D] -> [B * num_offsets, suffix_length, D]
        suffix = hidden_states[1]
        hidden_dim = suffix.shape[-1]
        suffix_flat = suffix.view(batch_size * num_offsets, suffix_length, hidden_dim)

        # Reshape per-token cond: [B, num_offsets, suffix_length, D] -> [B * num_offsets, suffix_length, D]
        cond_flat = suffix_adarms_conds.view(batch_size * num_offsets, suffix_length, -1) if suffix_adarms_conds is not None else None

        suffix_normed_flat, suffix_gate_flat = self.input_layernorm[1](suffix_flat, cond=cond_flat)

        # Reshape back
        suffix_normed = suffix_normed_flat.view(batch_size, num_offsets * suffix_length, hidden_dim)
        hidden_states[1] = suffix_normed

        # Gate is already per-token [B * num_offsets, suffix_length, D]
        # Reshape to [B, num_offsets * suffix_length, D]
        suffix_gates = suffix_gate_flat.view(batch_size, num_offsets * suffix_length, -1)
        gates.append(suffix_gates)

        # ============ Self-attention ============
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids, use_cache=use_cache)

        # ============ Gated residual connection ============
        for i in range(len(hidden_states)):
            hs = hidden_states[i]
            if hs is None:
                continue
            hidden_states[i] = _gated_residual(residuals[i], hs, gates[i])

        # ============ Pre-MLP layernorm ============
        residuals = [hs.clone() if hs is not None else None for hs in hidden_states]
        gates = []

        prefix = hidden_states[0]
        prefix_normed, prefix_gate = self.post_attention_layernorm[0](prefix, cond=None)
        hidden_states[0] = prefix_normed
        gates.append(prefix_gate)

        suffix = hidden_states[1]
        hidden_dim = suffix.shape[-1]
        suffix_flat = suffix.view(batch_size * num_offsets, suffix_length, hidden_dim)

        cond_flat = suffix_adarms_conds.view(batch_size * num_offsets, suffix_length, -1) if suffix_adarms_conds is not None else None
        suffix_normed_flat, suffix_gate_flat = self.post_attention_layernorm[1](suffix_flat, cond=cond_flat)

        suffix_normed = suffix_normed_flat.view(batch_size, num_offsets * suffix_length, hidden_dim)
        hidden_states[1] = suffix_normed

        suffix_gates = suffix_gate_flat.view(batch_size, num_offsets * suffix_length, -1)
        gates.append(suffix_gates)

        # ============ MLP ============
        hidden_states = self.mlp(hidden_states)

        # ============ Gated residual connection ============
        for i in range(len(hidden_states)):
            hs = hidden_states[i]
            if hs is None:
                continue
            hidden_states[i] = _gated_residual(residuals[i], hs, gates[i])

        return hidden_states


# ============================================================
# Core Model
# ============================================================

class PI05FlashVLAModel(nn.Module):
    """Core model for FlashVLA with padded cold start."""

    def __init__(self, config: PI05FlashVLAConfig):
        super().__init__()
        self.config = config
        self.N = config.num_buffer_slots
        self.C = config.chunk_size

        # Initialize backbone models
        self.vlm = PaliGemmaForConditionalGeneration(config.vlm_config)
        self.action_expert = GemmaForCausalLM(config.action_expert_config)

        # Note: patch_for_torch_compile is called after weight loading in from_pretrained,
        # because the patched GemmaAttention fuses q/k/v proj which changes state_dict keys.

        # Embedders
        self.prefix_embedder = PI05PrefixEmbedder(config, self.vlm)
        self.suffix_embedder = PI05SuffixEmbedder(config)

        # Shared transformer layers with per-token adaRMS
        num_hidden_layers = config.vlm_config.text_config.num_hidden_layers
        self.layers = nn.ModuleList([
            PI05ModelLayer(
                config,
                self.vlm.model.language_model.layers[i],
                self.action_expert.model.layers[i],
            )
            for i in range(num_hidden_layers)
        ])

        # Output projection
        self.action_out_proj = nn.Linear(config.action_expert_config.hidden_size, config.max_action_dim)

        # Convert to bfloat16 for efficiency
        self.to_bfloat16_for_selected_params(getattr(config, "dtype", "float32"))

        # Persistent 0-d int64 tensor holding the current cold-start step. By
        # passing the SAME tensor (fixed memory address) into _cold_start and
        # in-place updating with .fill_() each call, dynamo+cudagraphs can
        # capture ONE graph that handles every step_counter value via tensor
        # arithmetic — instead of N-1 specialized graphs.
        self.register_buffer(
            "_cold_step_t",
            torch.zeros((), dtype=torch.int64),
            persistent=False,
        )

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            # Both branches compiled. _cold_start uses a persistent 0-d tensor
            # for step_counter (see _cold_step_t above) so dynamo produces a
            # single graph instead of recompiling per int value. With dynamic
            # ints, cudagraphs would skip capture (~300ms/call); with the
            # fixed-address tensor input, cudagraph trees can capture once
            # and replay across all N-1 cold-start calls per episode.
            self._steady_streaming = torch.compile(self._steady_streaming, mode=config.compile_mode)
            self._cold_start = torch.compile(self._cold_start, mode=config.compile_mode)

    def detach_backbone_layer_aliases(self):
        num_layers = len(self.layers)
        self.vlm.model.language_model.layers = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
        self.action_expert.model.layers = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

    def backbone_dtype(self):
        # bf16 compute dtype from a Linear that survives mlp fusion (only
        # gate_proj/up_proj deleted) AND detach_backbone_layer_aliases. Do NOT
        # read a layernorm weight: FlashVLA keeps layernorms float32.
        return self.layers[0].mlp.vlm_mlp.down_proj.weight.dtype

    def to_bfloat16_for_selected_params(self, precision: str = "bfloat16") -> None:
        modules = [self.vlm, self.action_expert]
        params_to_keep_float32 = [
            # Keep the vision input embeddings, layernorms, and final norm in float32.
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]
        if precision == "bfloat16":
            for m in modules:
                for name, param in m.named_parameters():
                    if any(selector in name for selector in params_to_keep_float32):
                        continue
                    param.data = param.data.to(dtype=torch.bfloat16)
        elif precision == "float32":
            for m in modules:
                m.to(dtype=torch.float32)
        else:
            raise ValueError(f"Invalid precision: {precision}")

    def init_qkv_fusion_from_existing(self) -> None:
        backbones = [self.vlm.model.language_model, self.action_expert.model]
        for backbone in backbones:
            num_layers = backbone.config.num_hidden_layers
            for idx in range(num_layers):
                layer = backbone.layers[idx]
                attn = layer.self_attn
                q_proj = attn.q_proj
                k_proj = attn.k_proj
                v_proj = attn.v_proj
                hidden_size = q_proj.in_features
                head_dim = attn.head_dim
                num_heads = self.vlm.model.language_model.config.num_attention_heads
                num_kv_heads = self.vlm.model.language_model.config.num_key_value_heads
                qkv = QKVLinear(
                    hidden_size=hidden_size, head_size=head_dim,
                    total_num_heads=num_heads, total_num_kv_heads=num_kv_heads,
                    bias=q_proj.bias is not None,
                )
                attn.qkv_proj = qkv
                qkv.to(device=q_proj.weight.device, dtype=q_proj.weight.dtype)
                with torch.no_grad():
                    q_span = num_heads * head_dim
                    kv_span = num_kv_heads * head_dim
                    qkv.weight[:q_span].copy_(q_proj.weight)
                    qkv.weight[q_span:q_span + kv_span].copy_(k_proj.weight)
                    qkv.weight[q_span + kv_span:].copy_(v_proj.weight)
                    if qkv.bias is not None:
                        qkv.bias[:q_span].copy_(q_proj.bias)
                        qkv.bias[q_span:q_span + kv_span].copy_(k_proj.bias)
                        qkv.bias[q_span + kv_span:].copy_(v_proj.bias)
                delattr(attn, "q_proj")
                delattr(attn, "k_proj")
                delattr(attn, "v_proj")

    def init_mlp_fusion_from_existing(self) -> None:
        backbones = [self.vlm.model.language_model, self.action_expert.model]
        for backbone in backbones:
            num_layers = backbone.config.num_hidden_layers
            for idx in range(num_layers):
                layer = backbone.layers[idx]
                mlp = layer.mlp
                hidden_size = mlp.hidden_size
                intermediate_size = mlp.intermediate_size
                gate_up = MergedColumnLinear(hidden_size, [intermediate_size, intermediate_size], bias=False)
                mlp.gate_up_proj = gate_up
                gate_up.to(device=mlp.gate_proj.weight.device, dtype=mlp.gate_proj.weight.dtype)
                with torch.no_grad():
                    gate_up.weight[:intermediate_size].copy_(mlp.gate_proj.weight)
                    gate_up.weight[intermediate_size:].copy_(mlp.up_proj.weight)
                delattr(mlp, "gate_proj")
                delattr(mlp, "up_proj")

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def _build_per_chunk_time(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Per-chunk independent time sampling.

        Each (config k_idx, slot s) pair draws its own Beta sample, so two
        configs at the same slot level get DIFFERENT t.

        Returns:
            time_per_token: ``[B, N * num_slots_per_config * C]``. All C
            tokens within a single (config, slot) pair share one t (this is
            required by flow-matching: a chunk denoises with one t).
        """
        N, C = self.N, self.C
        num_slots_per_config = N

        alpha = torch.tensor(
            self.config.time_sampling_beta_alpha, device=device, dtype=torch.float32,
        )
        beta = torch.tensor(
            self.config.time_sampling_beta_beta, device=device, dtype=torch.float32,
        )
        beta_dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
        # Sample one Beta per (B, config, slot) — independent across configs.
        samples = beta_dist.sample(
            (batch_size, N, num_slots_per_config),
        ).to(device=device, dtype=torch.float32)

        # Build static segment-start matrix and is_real mask
        # shared across batch elements.
        seg_start = torch.zeros(N, num_slots_per_config, device=device, dtype=torch.float32)
        is_real = torch.zeros(N, num_slots_per_config, device=device, dtype=torch.bool)
        for k_idx in range(N):
            num_real = k_idx + 1
            for i in range(num_real):
                seg_start[k_idx, i] = (N - num_real + i) / N
                is_real[k_idx, i] = True

        # time = seg_start + sample/N, then scale+offset; padding=1.
        time_per_slot = seg_start[None, :, :] + samples / N
        time_per_slot = (
            time_per_slot * self.config.time_sampling_scale
            + self.config.time_sampling_offset
        )
        time_per_slot = torch.where(
            is_real[None, :, :], time_per_slot, torch.ones_like(time_per_slot),
        )
        # [B, N, num_slots_per_config] → expand to per-token [B, N*S*C]
        time_per_token = (
            time_per_slot.unsqueeze(-1)
            .expand(-1, -1, -1, C)
            .reshape(batch_size, -1)
        )
        return time_per_token

    def _build_shared_obs_mask(
        self,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        suffix_pad_masks_per_offset: torch.Tensor,
        suffix_att_masks: torch.Tensor,
        num_offsets: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build attention mask with per-config suffix padding.

        Args:
            prefix_pad_masks: [B, prefix_length].
            prefix_att_masks: [B, prefix_length].
            suffix_pad_masks_per_offset: [B, num_offsets, suffix_length].
            suffix_att_masks: [B, suffix_length] for one suffix (same structure for all).
            num_offsets: Number of offset branches.
            dtype: Output dtype.

        Returns:
            attention_mask: [B, 1, total_length, total_length].
            position_ids: [B, total_length].
        """
        batch_size = prefix_pad_masks.shape[0]
        prefix_length = prefix_pad_masks.shape[1]
        suffix_length = suffix_pad_masks_per_offset.shape[2]
        total_length = prefix_length + suffix_length * num_offsets
        device = prefix_pad_masks.device
        mask_value = torch.finfo(dtype).min

        # Build full pad_masks with per-offset suffix padding
        full_pad_masks = torch.zeros(batch_size, total_length, dtype=torch.bool, device=device)
        full_att_masks = torch.zeros(batch_size, total_length, dtype=prefix_att_masks.dtype, device=device)

        full_pad_masks[:, :prefix_length] = prefix_pad_masks
        full_att_masks[:, :prefix_length] = prefix_att_masks

        suffix_pad_tiled = suffix_pad_masks_per_offset.reshape(batch_size, -1)
        suffix_att_tiled = suffix_att_masks.unsqueeze(1).expand(-1, num_offsets, -1).reshape(batch_size, -1)

        full_pad_masks[:, prefix_length:] = suffix_pad_tiled
        full_att_masks[:, prefix_length:] = suffix_att_tiled

        # Block-causal via cumsum
        cumsum = torch.cumsum(full_att_masks, dim=1)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]

        # Apply padding
        pad_2d_masks = full_pad_masks[:, None, :] & full_pad_masks[:, :, None]
        att_2d_masks = att_2d_masks & pad_2d_masks

        # Block cross-offset attention in suffix
        suffix_positions = torch.arange(num_offsets * suffix_length, device=device)
        offset_ids = suffix_positions // suffix_length
        cross_offset_mask = offset_ids.unsqueeze(1) == offset_ids.unsqueeze(0)
        suffix_start = prefix_length
        att_2d_masks[:, suffix_start:, suffix_start:] = att_2d_masks[:, suffix_start:, suffix_start:] & cross_offset_mask

        # Position IDs
        position_ids = torch.zeros(batch_size, total_length, dtype=torch.long, device=device)
        prefix_pos = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
        position_ids[:, :prefix_length] = prefix_pos
        last_prefix_pos = prefix_pos[:, -1]

        suffix_pos_per_offset = torch.cumsum(suffix_pad_masks_per_offset.long(), dim=2)
        suffix_pos_tiled = suffix_pos_per_offset.reshape(batch_size, -1)
        position_ids[:, prefix_length:] = last_prefix_pos[:, None] + suffix_pos_tiled

        # Convert to additive mask
        attention_mask = torch.where(
            att_2d_masks,
            torch.zeros_like(att_2d_masks, dtype=dtype),
            torch.full_like(att_2d_masks, mask_value, dtype=dtype),
        )
        attention_mask = attention_mask.unsqueeze(1)

        return attention_mask, position_ids
    
    def forward_shared_observation(
        self, images, img_masks, tokens, masks,
        states, actions, action_is_pad,
        noise=None,
    ):
        """Training forward pass with shared observation across all N buffer configs.

        Args:
            images: List of image tensors [B, C, H, W].
            img_masks: List of validity masks [B].
            tokens: Language token IDs [B, L_text].
            masks: Language attention masks [B, L_text].
            states: Robot states [B, state_dim].
            actions: Target actions [B, N * H, action_dim]. H = N*C.
            action_is_pad: [B, N * H] boolean, True for padded positions.
            noise: Optional noise [B, N * H, action_dim].

        Returns:
            Per-element MSE loss.
        """
        batch_size = states.shape[0]
        N, C = self.N, self.C
        num_offsets = N
        suffix_length = H = N * C
        device = states.device

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        time_per_token = self._build_per_chunk_time(batch_size, device)

        # x_t needs per-token times (each token gets its own noise sample).
        time_expanded = time_per_token.unsqueeze(-1)  # [B, N*H, 1]
        # For noisy slots: x_t = t*noise + (1-t)*actions (standard flow matching)
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embed shared prefix once
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedder(
            images, img_masks, tokens, masks
        )

        # action_is_pad: True = padded; suffix embedder expects True = real
        real_mask = ~action_is_pad if action_is_pad is not None else None
        suffix_embs, suffix_pad_masks, suffix_att_masks, suffix_adarms_cond = self.suffix_embedder(
            states, x_t, time_per_token,
            padding_mask=real_mask,
        )
        hidden_dim = suffix_embs.shape[-1]

        # Per-config suffix pad masks for attention: [B, N, H]
        suffix_pad_masks_per_offset = suffix_pad_masks.view(batch_size, N, H)

        # Representative att_masks (same block structure for all configs): [B, H]
        suffix_att_masks = suffix_att_masks[:, :H]

        # adarms_cond reshaped for layers: [B, N, H, D]
        suffix_adarms_cond = suffix_adarms_cond.view(batch_size, N, H, -1)

        # Compute dtype from a Linear weight (layernorms are kept float32).
        backbone_dtype = self.backbone_dtype()
        prefix_embs = prefix_embs.to(dtype=backbone_dtype)
        suffix_embs = suffix_embs.to(dtype=backbone_dtype)

        attention_mask, position_ids = self._build_shared_obs_mask(
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            suffix_pad_masks_per_offset=suffix_pad_masks_per_offset,
            suffix_att_masks=suffix_att_masks,
            num_offsets=num_offsets,
            dtype=prefix_embs.dtype,
        )

        # Forward through transformer layers
        hidden_states = [prefix_embs, suffix_embs]

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask,
                position_ids,
                suffix_adarms_conds=suffix_adarms_cond,
                num_offsets=num_offsets,
                suffix_length=suffix_length,
            )

        # Final layer norm with per-token conditioning
        norms = [self.vlm.language_model.norm, self.action_expert.model.norm]

        prefix_out = hidden_states[0]
        prefix_out, _ = norms[0](prefix_out, cond=None)

        suffix_out = hidden_states[1]  # [B, N*H, D]
        suffix_out = suffix_out.view(batch_size * N, H, hidden_dim)
        cond = suffix_adarms_cond.view(batch_size * N, H, -1)
        suffix_out, _ = norms[1](suffix_out, cond=cond)
        suffix_out = suffix_out.view(batch_size, N * H, hidden_dim)

        # Project to action space
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out)  # [B, N*H, action_dim]

        # Compute MSE loss
        losses = F.mse_loss(u_t, v_t, reduction="none")  # [B, N*H, action_dim]

        # Build loss mask: exclude padded positions
        if action_is_pad is not None:
            loss_mask = ~action_is_pad  # [B, N*H]
        else:
            loss_mask = torch.ones(batch_size, N * H, dtype=torch.bool, device=device)

        losses = losses[loss_mask]

        return losses

    @torch.no_grad()
    def denoise_step(
        self,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        x_t,
        timestep,
        padding_mask=None,
    ):
        """Single denoising step with KV-cached prefix.

        Args:
            prefix_pad_masks: Cached prefix padding masks [B, L_prefix].
            prefix_att_masks: Cached prefix attention masks [B, L_prefix].
            state: Robot state [B, state_dim].
            x_t: Current buffer [B, buf_len, action_dim]. buf_len = N*C.
            timestep: Per-token time [B, buf_len].
            padding_mask: [B, buf_len] boolean (per-token).

        Returns:
            Predicted velocity v_t [B, buf_len, action_dim].
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, suffix_adarms_cond = self.suffix_embedder(
            state, x_t, timestep, padding_mask=padding_mask
        )

        # Compute dtype from a Linear weight (layernorms are kept float32).
        backbone_dtype = self.backbone_dtype()
        suffix_embs = suffix_embs.to(dtype=backbone_dtype)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        full_attention_mask, full_position_ids = build_attention_mask_and_position_ids(
            pad_masks, att_masks, suffix_embs.dtype,
        )

        L_suf = suffix_embs.shape[1]
        attention_mask = full_attention_mask[:, :, -L_suf:, :]
        position_ids = full_position_ids[:, -L_suf:]

        hidden_states = [None, suffix_embs]
        conds = [None, suffix_adarms_cond]

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids, conds, use_cache=True)

        suffix_hidden = hidden_states[1]
        suffix_hidden, _ = self.action_expert.model.norm(suffix_hidden, cond=suffix_adarms_cond)

        suffix_out = suffix_hidden.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)

    @torch.no_grad()
    def _cold_start(self, images, img_masks, tokens, masks, state, buffer, step_counter, profile=False):
        """One denoise step during cold-start (buffer not yet full).

        step_counter is a 0-d int64 tensor (NOT a Python int) so that
        dynamo produces a single graph parametrized by tensor value — no
        per-int recompiles — and cudagraph trees can capture once and
        replay across all step_counter values. All output shapes are
        static (constant in N, C, bsz); step_counter only enters through
        pure tensor arithmetic (torch.arange + comparison + torch.where).

        Returns:
            (updated_buffer, breakdown_dict_or_None)
        """
        N, C = self.N, self.C
        buf_slots = N
        buf_len = buf_slots * C
        bsz = tokens.shape[0]
        device = tokens.device

        _evs = (
            [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            if profile else None
        )

        # Encode: compute and cache prefix KV
        if profile:
            _ev_record(_evs[0])
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedder(
            images, img_masks, tokens, masks
        )
        if profile:
            _ev_record(_evs[1])

        # Prefill: compute and cache prefix KV
        for layer in self.layers:
            layer.self_attn.attn.reset_cache()

        prefix_attention_mask, prefix_position_ids = build_attention_mask_and_position_ids(
            prefix_pad_masks, prefix_att_masks, prefix_embs.dtype,
        )

        hidden_states_prefill = [prefix_embs, None]
        conds_prefill = [None, None]
        for layer in self.layers:
            hidden_states_prefill = layer(
                hidden_states_prefill, prefix_attention_mask, prefix_position_ids,
                conds_prefill, use_cache=True,
            )
        if profile:
            _ev_record(_evs[2])

        # ===== Vectorized mask/time construction (no Python control flow) =====
        # step_counter is a Python int; num_real specializes the graph per value.
        num_real = step_counter + 1

        slot_idx = torch.arange(buf_slots, device=device)              # [buf_slots]
        slot_idx_buf = torch.arange(buf_len, device=device) // C       # [buf_len]

        # slot 0..num_real-1 = noisy real, time = (N - num_real + 1 + slot_idx)/N
        # slot num_real..N-1 = padding, time = 1.0
        is_real = slot_idx < num_real
        time_slot = (N - num_real + 1 + slot_idx).to(torch.float32) / N
        time_slot = torch.where(is_real, time_slot, torch.ones_like(time_slot))

        padding_mask = is_real.unsqueeze(0).unsqueeze(2).expand(bsz, -1, C).reshape(bsz, buf_len)
        time = time_slot.unsqueeze(0).unsqueeze(2).expand(bsz, -1, C).reshape(bsz, buf_len)

        v_t = self.denoise_step(
            prefix_pad_masks, prefix_att_masks, state,
            buffer, time, padding_mask=padding_mask,
        )

        dt = 1.0 / N
        # New buffer assembled via masks instead of slice assignment.
        new_noise = self.sample_noise((bsz, buf_len, self.config.max_action_dim), device)
        updated_buffer = buffer - dt * v_t
        keep_mask = slot_idx_buf < num_real            # slot 0..num_real-1
        noise_mask = slot_idx_buf == num_real          # slot num_real

        keep_mask = keep_mask.view(1, buf_len, 1)
        noise_mask = noise_mask.view(1, buf_len, 1)
        buffer = torch.where(
            keep_mask,
            updated_buffer,
            torch.where(noise_mask, new_noise, torch.zeros_like(updated_buffer)),
        )

        if profile:
            _ev_record(_evs[3])
            _cuda_sync()
            breakdown = {
                "encode": _ev_elapsed_ms(_evs[0], _evs[1]),
                "prefill": _ev_elapsed_ms(_evs[1], _evs[2]),
                "action": _ev_elapsed_ms(_evs[2], _evs[3]),
                "total": _ev_elapsed_ms(_evs[0], _evs[3]),
            }
        else:
            breakdown = None

        return buffer, breakdown

    @torch.no_grad()
    def _steady_streaming(self, images, img_masks, tokens, masks, state, buffer, profile=False):
        """One denoise step during steady streaming (buffer full).

        Static shapes throughout — this is the function compiled into a
        single CUDA graph for non-blocking dispatch. profile parameter:
        per-segment breakdown is meaningless under cuda graph replay; the
        outer caller measures wall-clock total instead.
        """
        N, C = self.N, self.C
        buf_slots = N
        buf_len = buf_slots * C
        bsz = tokens.shape[0]
        device = tokens.device

        # Wall-clock total measurement (only meaningful under compile=True
        # for the steady branch; cold-start also reports breakdown).
        _evs_total = (
            [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            if profile else None
        )
        
        # Encode: compute and cache prefix KV
        if profile:
            _ev_record(_evs_total[0])
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedder(
            images, img_masks, tokens, masks
        )
        if profile:
            _ev_record(_evs_total[1])
        # Prefill: compute and cache prefix KV
        for layer in self.layers:
            layer.self_attn.attn.reset_cache()

        prefix_attention_mask, prefix_position_ids = build_attention_mask_and_position_ids(
            prefix_pad_masks, prefix_att_masks, prefix_embs.dtype,
        )

        hidden_states_prefill = [prefix_embs, None]
        conds_prefill = [None, None]
        for layer in self.layers:
            hidden_states_prefill = layer(
                hidden_states_prefill, prefix_attention_mask, prefix_position_ids,
                conds_prefill, use_cache=True,
            )
        if profile:
            _ev_record(_evs_total[2])
        # Steady-state time vector (all slots real)
        time_per_slot = torch.arange(1, N + 1, device=device, dtype=torch.float32) / N
        time = time_per_slot.unsqueeze(0).unsqueeze(2).expand(bsz, -1, C).reshape(bsz, buf_len)

        v_t = self.denoise_step(
            prefix_pad_masks, prefix_att_masks, state,
            buffer, time, padding_mask=None,
        )

        dt = 1.0 / N
        buffer = buffer - dt * v_t
        actions_to_execute = buffer[:, :C, :self.config.max_action_dim]
        new_buffer = self.sample_noise((bsz, buf_len, self.config.max_action_dim), device)
        new_buffer[:, :-C, :] = buffer[:, C:, :]
        buffer = new_buffer

        if profile:
            _ev_record(_evs_total[3])
            _cuda_sync()
            breakdown = {
                "encode": _ev_elapsed_ms(_evs_total[0], _evs_total[1]),
                "prefill": _ev_elapsed_ms(_evs_total[1], _evs_total[2]),
                "action": _ev_elapsed_ms(_evs_total[2], _evs_total[3]),
                "total": _ev_elapsed_ms(_evs_total[0], _evs_total[3]),
            }
        else:
            breakdown = None

        return actions_to_execute, buffer, breakdown

    @torch.no_grad()
    def sample_actions(self, images, img_masks, tokens, masks, state, buffer, step_counter, profile=False):
        """Top-level dispatch: cold-start (eager) or steady streaming (compiled).

        Buffer layout is [noisy_slot_0(C), ..., noisy_slot_{N-1}(C)].

        Args:
            images: Input images.
            img_masks: Image validity masks.
            tokens: Language tokens.
            masks: Language masks.
            state: Robot state [B, state_dim].
            buffer: Action buffer or None for first step.
            step_counter: Current step (0-indexed).
            profile: when True, return wall-clock and (in cold-start only)
                encode/prefill/action breakdown.

        Returns:
            (actions_to_execute [B, C, action_dim] or None, updated_buffer,
             new_step_counter, profile_results_or_None).
        """
        N, C = self.N, self.C
        buf_slots = N
        buf_len = buf_slots * C
        bsz = tokens.shape[0]
        device = tokens.device

        # Initialize buffer on first call
        if buffer is None:
            buffer = self.sample_noise((bsz, buf_len, self.config.max_action_dim), device)

        if step_counter < N - 1:
            # Cold start. Write the current step into our persistent 0-d
            # buffer so the compiled graph reads from a fixed address (enables
            # cudagraph replay across all step_counter values).
            self._cold_step_t.fill_(step_counter)
            buffer, breakdown = self._cold_start(
                images, img_masks, tokens, masks, state, buffer, self._cold_step_t,
                profile=profile,
            )
            actions_to_execute = None
        else:
            # Steady streaming: compiled, single CUDA graph, non-blocking
            actions_to_execute, buffer, breakdown = self._steady_streaming(
                images, img_masks, tokens, masks, state, buffer,
                profile=profile,
            )

        if profile:
            profile_results = breakdown
        else:
            profile_results = None

        return actions_to_execute, buffer, step_counter + 1, profile_results


# ============================================================
# Policy wrapper
# ============================================================

class PI05FlashVLAPolicy(PreTrainedPolicy):
    """PI0.5 FlashVLA Policy wrapper."""

    config_class = PI05FlashVLAConfig
    name = "pi05-flashvla"

    def __init__(
        self,
        config: PI05FlashVLAConfig,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Normalization is handled by the external processor pipeline.

        self.model = PI05FlashVLAModel(config)

        # Pre-allocate streaming buffer
        buf_len = config.total_buffer_length
        self.register_buffer(
            "action_buffer",
            torch.zeros(1, buf_len, config.max_action_dim),
            persistent=False,
        )
        self._step_counter = 0

        # Cold-start normalization stats are filled in by from_pretrained().
        # Default to an empty ColdStartStats so locally-constructed policies
        # (e.g. unit tests) still have a valid attribute to read.
        self._cold_start_stats = ColdStartStats()

        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> T:
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        kwargs.pop("dataset_stats", None)
        instance = cls(config, **kwargs)

        from safetensors.torch import load_file
        from transformers.utils import cached_file

        original_state_dict: dict[str, Tensor] | None = None
        if os.path.isdir(pretrained_name_or_path):
            model_file = os.path.join(pretrained_name_or_path, "model.safetensors")
            if not os.path.isfile(model_file):
                raise FileNotFoundError(f"No 'model.safetensors' found in directory: {model_file}")
            original_state_dict = load_file(model_file)
        else:
            resolved_file = cached_file(
                pretrained_name_or_path, "model.safetensors",
                cache_dir=cache_dir, force_download=force_download,
                resume_download=resume_download, proxies=proxies,
                token=token, revision=revision, local_files_only=local_files_only,
            )
            if resolved_file is None:
                raise FileNotFoundError(f"Could not resolve 'model.safetensors' for {pretrained_name_or_path}")
            original_state_dict = load_file(resolved_file)

        # Weight key mapping from OpenPI / FlashVLA base format
        prefix_rules: list[tuple[str, str]] = [
            ("model.action_in_proj.", "model.suffix_embedder.action_in_proj."),
            ("model.action_out_proj.", "model.action_out_proj."),
            ("model.time_mlp_in.", "model.suffix_embedder.time_mlp_in."),
            ("model.time_mlp_out.", "model.suffix_embedder.time_mlp_out."),
            ("model.state_proj.", "model.suffix_embedder.state_proj."),
            ("model.state_mlp_in.", "model.suffix_embedder.state_mlp_in."),
            ("model.state_mlp_out.", "model.suffix_embedder.state_mlp_out."),
            ("model.paligemma_with_expert.gemma_expert.", "model.action_expert."),
            ("model.paligemma_with_expert.paligemma.", "model.vlm."),
            ("action_in_proj.", "model.suffix_embedder.action_in_proj."),
            ("action_out_proj.", "model.action_out_proj."),
            ("time_mlp_in.", "model.suffix_embedder.time_mlp_in."),
            ("time_mlp_out.", "model.suffix_embedder.time_mlp_out."),
            ("state_proj.", "model.suffix_embedder.state_proj."),
            ("state_mlp_in.", "model.suffix_embedder.state_mlp_in."),
            ("state_mlp_out.", "model.suffix_embedder.state_mlp_out."),
            ("paligemma_with_expert.gemma_expert.", "model.action_expert."),
            ("paligemma_with_expert.paligemma.", "model.vlm."),
        ]

        def map_key(key: str) -> str | None:
            for src, dst in prefix_rules:
                if key.startswith(src):
                    return dst + key[len(src):]
            return key

        target_sd = instance.state_dict()
        mapped_sd: dict[str, Tensor] = {}
        for old_key, value in original_state_dict.items():
            new_key = map_key(old_key)
            if (new_key in target_sd) and (target_sd[new_key].shape != value.shape):
                continue
            mapped_sd[new_key] = value

        incompatible = instance.load_state_dict(mapped_sd, strict=False)
        _missing_keys, unexpected_keys = incompatible.missing_keys, incompatible.unexpected_keys

        unexpected_fatal = list(unexpected_keys)
        if unexpected_fatal:
            raise RuntimeError(
                "Checkpoint loading failed.\n"
                f"Unexpected keys: {unexpected_fatal}"
            )

        # Only patch GemmaRMSNorm (in-place class swap, no state_dict change).
        # Skip GemmaAttention patching — PI05Attention accesses q/k/v_proj directly.
        from flashvla.policies.pi05.patches import FlashVLARMSNorm as PatchedGemmaRMSNorm
        from lerobot.policies.pi_gemma import PiGemmaRMSNorm as OriginalGemmaRMSNorm
        for module in instance.model.modules():
            if type(module) is OriginalGemmaRMSNorm:
                module.__class__ = PatchedGemmaRMSNorm

        instance.to(config.device)
        instance.eval()

        if getattr(config, "fuse_qkv", True):
            instance.model.init_qkv_fusion_from_existing()
        if getattr(config, "fuse_gate_up", True):
            instance.model.init_mlp_fusion_from_existing()

        instance.model.detach_backbone_layer_aliases()

        # Load action + state normalization stats for cold-start action
        # computation (see flashvla.policies.pi05.utils.load_cold_start_stats).
        instance._cold_start_stats = load_cold_start_stats(pretrained_name_or_path)

        return instance

    def reset(self):
        """Reset buffer and step counter. Call when environment resets.

        Buffer layout [noise, P, P, P, P]: slot 0 is noise, rest zeros (padding).
        """
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        C = self.config.chunk_size
        self.action_buffer.zero_()
        self.action_buffer[:, :C, :].normal_()
        self._step_counter = 0
        self._cold_start_action = None  # will be set on first select_action

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], profile=False) -> Tensor | None:
        """Predict a chunk of actions for inference.

        Returns:
            Action chunk [B, C, action_dim] or None during cold start.
        """
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions, buffer, new_step, profile_results = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            buffer=self.action_buffer,
            step_counter=self._step_counter,
            profile=profile,
        )

        self.action_buffer.copy_(buffer)
        self._step_counter = new_step

        if actions is None:
            # Cold-start: chunk not yet ready. Still pass profile data through if requested
            # (encode + prefill + cold-start denoise still consumed time worth measuring).
            if profile:
                return None, profile_results
            return None

        original_action_dim = self.config.action_feature.shape[0]
        if profile:
            return actions[:, :, :original_action_dim].float(), profile_results
        else:
            return actions[:, :, :original_action_dim].float()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action using action chunking.

        During cold start (buffer not yet full) this returns a "hold the
        robot still" action computed by
        :func:`flashvla.policies.pi05.utils.compute_cold_start_action`, with the
        strategy selected by ``config.cold_start_mode`` — ``"zero_delta"``
        for delta-action envs, ``"current_state"`` for absolute-qpos envs
        like RoboTwin.
        """
        mode = getattr(self.config, "cold_start_mode", "zero_delta")
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, profile=False)
            if actions is None:
                cold_action = compute_cold_start_action(
                    mode=mode,
                    stats=self._cold_start_stats,
                    action_dim=self.config.action_feature.shape[0],
                    device=self.action_buffer.device,
                    state_normalized=batch[OBS_STATE] if mode == "current_state" else None,
                    cache=self._cold_start_action if mode == "zero_delta" else None,
                )
                # Cache the state-independent zero-delta action across calls.
                if mode == "zero_delta":
                    self._cold_start_action = cold_action
                for _ in range(self.config.n_action_steps):
                    self._action_queue.append(cold_action)
            else:
                self._action_queue.extend(actions.transpose(0, 1)[:self.config.n_action_steps])
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor], noise=None) -> tuple[Tensor, dict[str, Tensor]]:
        """Training forward pass."""
        # Batch arrives already normalized and tokenized by the preprocessor.
        images, img_masks = self.prepare_images(batch)
        states = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        action_is_pad = batch.get("action_is_pad")

        loss_dict: dict[str, Tensor | float] = {}

        losses = self.model.forward_shared_observation(
            images, img_masks, lang_tokens, lang_masks,
            states, actions, action_is_pad, noise,
        )

        losses = losses[..., :self.config.max_action_dim]

        loss = losses.mean() if losses.numel() > 0 else losses.sum()
        loss_dict["loss"] = loss.item()

        return loss, loss_dict

    def prepare_images(self, batch):
        images: list[Tensor] = []
        img_masks: list[Tensor] = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                "All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key]
            img = resize_with_pad(img, *self.config.image_resolution, pad_value=0)
            img = img * 2.0 - 1.0
            bsz = img.shape[0]
            device = img.device
            mask = torch.ones(bsz, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_state(self, batch):
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions
