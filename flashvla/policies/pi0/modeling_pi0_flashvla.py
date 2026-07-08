#!/usr/bin/env python

# Copyright 2025 FlashVLA team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""PI0 FlashVLA: Padded Cold Start with Shared Observation Training.

Port of pi05's FlashVLA
(``flashvla/policies/pi05/modeling_pi05_flashvla.py``) to pi0's
non-adaRMS architecture. Buffer of N=5 slots × C=10 actions per slot,
one-step inference in steady state, padded cold start, shared-observation
training with PerSeg time sampling.

Differences from pi05 streaming:
  * Per-slot time is injected via pi0's existing concat path
    (``cat([action_emb, time_emb]) -> action_time_mlp_in/out``) instead of
    pi05's adaRMS FiLM. ``PI0ModelLayer.forward(..., conds=[None, None])``
    works unchanged — no patches, no per-token cond reshape gymnastics.
  * The state token sits at suffix index 0 of every buffer config (pi0
    keeps state-as-token; pi05 bakes state into the prompt via
    ``state_cond=False`` discretization). The action buffer therefore
    contains only action slots; state is freshly fed as an argument
    every call.

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
import os
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from lerobot.policies.pi_gemma import (
    PiGemmaForCausalLM as GemmaForCausalLM,
    PaliGemmaForConditionalGenerationWithPiGemma as PaliGemmaForConditionalGeneration,
)

from lerobot.configs.policies import PreTrainedConfig, T
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

from flashvla.policies.pi0.configuration_pi0 import PI0FlashVLAConfig
from flashvla.policies.pi0.modeling_pi0 import (
    PI0PrefixEmbedder,
    PI0ModelLayer,
)
from flashvla.policies.pi0.utils import (
    build_attention_mask_and_position_ids,
    build_flashvla_attention_mask_and_position_ids,
    pad_vector,
    resize_with_pad,
)
from flashvla.policies.pi05.utils import (
    ColdStartStats,
    compute_cold_start_action,
    create_sinusoidal_pos_embedding_for_blocks,
    load_cold_start_stats,
)


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




class PI0StreamingSuffixEmbedder(nn.Module):
    """Embed state + noisy actions with per-slot time conditioning.

    Two operating modes selected by ``config.use_adarms_time_cond``:

    * **False (default, concat path)** — pi0's native time injection:
      ``cat([action_emb, time_emb]) → action_time_mlp_in → silu →
      action_time_mlp_out`` produces the suffix **token embedding**.
      State token at suffix index 0 (just ``state_proj(state)``, no time).
      Returns ``adarms_cond=None``.

    * **True (adaRMS path, additive on top of baseline pi0)** — keep all
      baseline weights for their original purpose AND add pi05-style FiLM
      cond paths in parallel. Architecture is purely additive: no pi0_base
      weight is repurposed or dropped.
      - Token embedding: SAME as baseline pi0 (action_in_proj +
        action_time_mlp_in/out concat path), preserving pi0_base init.
      - Time cond: ``time_mlp_in → silu → time_mlp_out → silu`` (new).
      - State cond: ``state_proj → state_mlp_in → silu → state_mlp_out →
        silu`` (state_mlp_out zero-init for DiT-style start). State
        **suffix token is dropped** (it now contributes only via cond).
      - ``adarms_cond = time_cond + state_cond.unsqueeze(1)`` (per-token).
      - All new FiLM layers (dense in patched RMSNorm, plus state_mlp_out)
        are zero-init → at step 0 the model is mathematically identical to
        baseline pi0 streaming with concat-time; FiLM is no-op until
        training pushes the weights off zero.
    """

    def __init__(self, config: PI0FlashVLAConfig):
        super().__init__()
        self.config = config

        width = config.action_expert_config.hidden_size
        self.action_in_proj = nn.Linear(config.max_action_dim, width)
        self.state_proj = nn.Linear(config.max_state_dim, width)
        self.action_time_mlp_in = nn.Linear(width * 2, width)
        self.action_time_mlp_out = nn.Linear(width, width)

        if config.use_adarms_time_cond:
            self.time_mlp_in = nn.Linear(width, width)
            self.time_mlp_out = nn.Linear(width, width)
            self.state_mlp_in = nn.Linear(width, width)
            self.state_mlp_out = nn.Linear(width, width)
            nn.init.zeros_(self.state_mlp_out.weight)
            nn.init.zeros_(self.state_mlp_out.bias)

        self.C = config.chunk_size

    def forward(
        self,
        state: Tensor,
        noisy_actions: Tensor,
        time_per_slot: Tensor,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Embed state + noisy actions with per-slot time.

        Args:
            state: Robot state ``[B, state_dim]``.
            noisy_actions: ``[B, L_act, max_action_dim]`` with
                ``L_act = num_slots * C``.
            time_per_slot: Per-slot times ``[B, num_slots]``. Expanded to
                per-token internally by ``repeat_interleave(C)``.
            padding_mask: Optional ``[B, L_act]`` boolean. True = real
                action token, False = padded slot.

        Returns:
            * concat mode: embs ``[B, 1+L_act, D]`` (state at idx 0),
              pad_masks ``[B, 1+L_act]``, att_masks ``[B, 1+L_act]``,
              ``adarms_cond=None``.
            * adaRMS mode: embs ``[B, L_act, D]`` (no state token),
              pad_masks ``[B, L_act]``, att_masks ``[B, L_act]``,
              ``adarms_cond [B, L_act, D]`` per-token FiLM modulation.
        """
        bsize = state.shape[0]
        device = state.device
        proj_dtype = self.state_proj.weight.dtype
        C = self.C
        L_act = noisy_actions.shape[1]
        num_slots = L_act // C
        assert num_slots * C == L_act, (
            f"L_act={L_act} not divisible by C={C}"
        )

        state = state.to(dtype=proj_dtype)
        noisy_actions = noisy_actions.to(dtype=proj_dtype)

        time_per_token = time_per_slot.repeat_interleave(C, dim=1)
        time_emb = create_sinusoidal_pos_embedding_for_blocks(
            time_per_token,
            self.config.action_expert_config.hidden_size,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=device,
        )
        time_emb = time_emb.to(dtype=proj_dtype)

        action_emb = self.action_in_proj(noisy_actions)

        action_time_emb = torch.cat([action_emb, time_emb], dim=-1)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        if self.config.use_adarms_time_cond:
            time_cond = self.time_mlp_in(time_emb)
            time_cond = F.silu(time_cond)
            time_cond = self.time_mlp_out(time_cond)
            time_cond = F.silu(time_cond)

            state_emb = self.state_proj(state)
            state_emb = self.state_mlp_in(state_emb)
            state_emb = F.silu(state_emb)
            state_emb = self.state_mlp_out(state_emb)
            state_emb = F.silu(state_emb)

            adarms_cond = time_cond + state_emb.unsqueeze(1)

            embs = action_time_emb

            if padding_mask is not None:
                pad_f = padding_mask.unsqueeze(-1).to(embs.dtype)
                embs = embs * pad_f
                adarms_cond = adarms_cond * pad_f
                pad_masks = padding_mask
            else:
                pad_masks = torch.ones(bsize, L_act, dtype=torch.bool, device=device)

            slot_att = torch.zeros(num_slots, C, dtype=embs.dtype, device=device)
            slot_att[:, 0] = 1
            att_masks = slot_att.flatten(0, 1).unsqueeze(0).expand(bsize, -1)

            return embs, pad_masks, att_masks, adarms_cond

        if padding_mask is not None:
            action_time_emb = action_time_emb * padding_mask.unsqueeze(-1).to(action_time_emb.dtype)

        state_emb = self.state_proj(state)[:, None, :]
        embs = torch.cat([state_emb, action_time_emb], dim=1)

        if padding_mask is not None:
            action_pad = padding_mask
        else:
            action_pad = torch.ones(bsize, L_act, dtype=torch.bool, device=device)
        state_pad = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks = torch.cat([state_pad, action_pad], dim=1)

        slot_att = torch.zeros(num_slots, C, dtype=embs.dtype, device=device)
        slot_att[:, 0] = 1
        action_att = slot_att.flatten(0, 1)
        state_att = torch.ones(1, dtype=embs.dtype, device=device)
        full_att = torch.cat([state_att, action_att], dim=0)
        att_masks = full_att.unsqueeze(0).expand(bsize, -1)

        return embs, pad_masks, att_masks, None




class PI0FlashVLAModel(nn.Module):
    """Core PI0 FlashVLA model — buffer + cold start + steady state."""

    def __init__(self, config: PI0FlashVLAConfig):
        super().__init__()
        self.config = config
        self.N = config.num_buffer_slots
        self.C = config.chunk_size

        if config.use_adarms_time_cond:
            config.action_expert_config.use_adarms = True
            config.action_expert_config.adarms_cond_dim = config.action_expert_config.hidden_size
        self.vlm = PaliGemmaForConditionalGeneration(config.vlm_config)
        self.action_expert = GemmaForCausalLM(config.action_expert_config)

        self.prefix_embedder = PI0PrefixEmbedder(config, self.vlm)
        self.suffix_embedder = PI0StreamingSuffixEmbedder(config)

        num_hidden_layers = config.vlm_config.text_config.num_hidden_layers
        self.layers = nn.ModuleList([
            PI0ModelLayer(
                config,
                self.vlm.model.language_model.layers[i],
                self.action_expert.model.layers[i],
            )
            for i in range(num_hidden_layers)
        ])

        self.action_out_proj = nn.Linear(config.action_expert_config.hidden_size, config.max_action_dim)

        self.to_bfloat16_for_selected_params(getattr(config, "dtype", "float32"))

        self.register_buffer("_cold_step_t", torch.zeros((), dtype=torch.int64), persistent=False)

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self._cold_start = torch.compile(self._cold_start, mode=config.compile_mode)
            self._steady_streaming = torch.compile(self._steady_streaming, mode=config.compile_mode)


    def to_bfloat16_for_selected_params(self, precision: str = "bfloat16") -> None:
        modules = [self.vlm, self.action_expert]
        params_to_keep_float32 = [
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
        from flashvla.layers.linear import QKVLinear

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
        from flashvla.layers.linear import MergedColumnLinear

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
    ) -> tuple[Tensor, Tensor]:
        """Per-chunk independent time sampling.

        Each (config k_idx, slot s) draws its own Beta sample. Real slots get
        sampled t; padded slots get t=1.0.
        """
        N, C = self.N, self.C
        num_slots_per_config = N

        alpha = torch.tensor(self.config.time_sampling_beta_alpha, device=device, dtype=torch.float32)
        beta = torch.tensor(self.config.time_sampling_beta_beta, device=device, dtype=torch.float32)
        beta_dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
        samples = beta_dist.sample((batch_size, N, num_slots_per_config)).to(device=device, dtype=torch.float32)

        seg_start = torch.zeros(N, num_slots_per_config, device=device, dtype=torch.float32)
        is_real = torch.zeros(N, num_slots_per_config, device=device, dtype=torch.bool)
        for k_idx in range(N):
            num_real = k_idx + 1
            for i in range(num_real):
                seg_start[k_idx, i] = (N - num_real + i) / N
                is_real[k_idx, i] = True

        time_per_slot = seg_start[None, :, :] + samples / N
        time_per_slot = (
            time_per_slot * self.config.time_sampling_scale
            + self.config.time_sampling_offset
        )
        time_per_slot = torch.where(
            is_real[None, :, :], time_per_slot, torch.ones_like(time_per_slot),
        )
        time_per_slot = time_per_slot.reshape(batch_size, -1)
        time_per_token = time_per_slot.repeat_interleave(C, dim=1)
        return time_per_slot, time_per_token


    def forward_shared_observation(
        self, images, img_masks, tokens, masks,
        states, actions, action_is_pad,
        noise=None,
    ):
        """Shared-observation training: N buffer configs side-by-side, one prefix.

        Args:
            states: ``[B, state_dim]`` (single state shared across all N configs).
            actions: ``[B, N * H_cfg, action_dim]`` flattened in config-major
                order. ``H_cfg = N*C``.
            action_is_pad: ``[B, N * H_cfg]`` boolean.
        """
        batch_size = states.shape[0]
        N, C = self.N, self.C
        H_cfg = N * C
        device = states.device

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        time_per_slot_all, time_per_token = self._build_per_chunk_time(
            batch_size, device,
        )

        time_expanded = time_per_token.unsqueeze(-1)
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedder(
            images, img_masks, tokens, masks,
        )

        num_slots_per_config = N
        states_flat = states.unsqueeze(1).expand(batch_size, N, -1).reshape(batch_size * N, -1)
        x_t_flat = x_t.view(batch_size * N, H_cfg, -1)
        time_per_slot_flat = time_per_slot_all.view(batch_size * N, num_slots_per_config)
        if action_is_pad is not None:
            real_mask_flat = (~action_is_pad).view(batch_size * N, H_cfg)
        else:
            real_mask_flat = None

        embs_flat, pad_flat, att_flat, cond_flat = self.suffix_embedder(
            states_flat, x_t_flat, time_per_slot_flat, padding_mask=real_mask_flat,
        )
        L_suf = embs_flat.shape[1]
        use_adarms = self.config.use_adarms_time_cond

        suffix_embs = embs_flat.view(batch_size, N, L_suf, -1).reshape(batch_size, N * L_suf, -1)
        suffix_pad_per_offset = pad_flat.view(batch_size, N, L_suf)
        suffix_att_masks = att_flat[:batch_size]

        if cond_flat is not None:
            suffix_adarms_cond = cond_flat.view(batch_size, N, L_suf, -1).reshape(batch_size, N * L_suf, -1)
        else:
            suffix_adarms_cond = None

        backbone_dtype = self.vlm.model.language_model.layers[0].self_attn.o_proj.weight.dtype
        prefix_embs = prefix_embs.to(dtype=backbone_dtype)
        suffix_embs = suffix_embs.to(dtype=backbone_dtype)
        if suffix_adarms_cond is not None:
            suffix_adarms_cond = suffix_adarms_cond.to(dtype=backbone_dtype)

        attention_mask, position_ids = build_flashvla_attention_mask_and_position_ids(
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            suffix_pad_masks_per_offset=suffix_pad_per_offset,
            suffix_att_masks=suffix_att_masks,
            num_offsets=N,
            dtype=prefix_embs.dtype,
        )

        hidden_states = [prefix_embs, suffix_embs]
        conds = [None, suffix_adarms_cond]
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids, conds, use_cache=False)

        _, _ = self.vlm.language_model.norm(hidden_states[0], cond=None)
        suffix_out, _ = self.action_expert.model.norm(hidden_states[1], cond=suffix_adarms_cond)

        suffix_out = suffix_out.view(batch_size, N, L_suf, -1)
        if use_adarms:
            action_out = suffix_out
        else:
            action_out = suffix_out[:, :, 1:, :]
        action_out = action_out.reshape(batch_size, N * H_cfg, -1)
        action_out = action_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(action_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")

        if action_is_pad is not None:
            loss_mask = ~action_is_pad
        else:
            loss_mask = torch.ones(batch_size, N * H_cfg, dtype=torch.bool, device=device)

        losses = losses[loss_mask]
        return losses


    @torch.no_grad()
    def denoise_step(
        self,
        prefix_pad_masks,
        prefix_att_masks,
        state,
        x_t,
        time_per_slot,
        padding_mask=None,
    ):
        """Single denoising step with KV-cached prefix.

        Returns predicted velocity ``v_t [B, buf_len, action_dim]`` —
        already with the state token sliced off.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, suffix_adarms_cond = self.suffix_embedder(
            state, x_t, time_per_slot, padding_mask=padding_mask,
        )

        backbone_dtype = self.vlm.model.language_model.layers[0].self_attn.o_proj.weight.dtype
        suffix_embs = suffix_embs.to(dtype=backbone_dtype)
        if suffix_adarms_cond is not None:
            suffix_adarms_cond = suffix_adarms_cond.to(dtype=backbone_dtype)

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

        if self.config.use_adarms_time_cond:
            action_hidden = suffix_hidden
        else:
            action_hidden = suffix_hidden[:, 1:, :]
        action_hidden = action_hidden.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(action_hidden)

    @torch.no_grad()
    def _cold_start(self, images, img_masks, tokens, masks, state, buffer, step_counter, profile=False):
        """One denoise step during cold-start (buffer not yet full).

        step_counter is a 0-d int64 tensor — see _cold_step_t docs above.
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

        if profile:
            _ev_record(_evs[0])
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedder(
            images, img_masks, tokens, masks,
        )
        if profile:
            _ev_record(_evs[1])

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

        num_real = step_counter + 1

        slot_idx = torch.arange(buf_slots, device=device)
        slot_idx_buf = torch.arange(buf_len, device=device) // C

        is_real = slot_idx < num_real
        time_slot = (N - num_real + 1 + slot_idx).to(torch.float32) / N
        time_slot = torch.where(is_real, time_slot, torch.ones_like(time_slot))

        time_per_slot = time_slot.unsqueeze(0).expand(bsz, -1)
        padding_mask = is_real.unsqueeze(0).unsqueeze(2).expand(bsz, -1, C).reshape(bsz, buf_len)

        v_t = self.denoise_step(
            prefix_pad_masks, prefix_att_masks, state,
            buffer, time_per_slot, padding_mask=padding_mask,
        )

        dt = 1.0 / N
        new_noise = self.sample_noise((bsz, buf_len, self.config.max_action_dim), device)
        updated_buffer = buffer - dt * v_t
        keep_mask = slot_idx_buf < num_real
        noise_mask = slot_idx_buf == num_real

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
    def _steady_streaming(self, images, img_masks, tokens, masks, state, buffer):
        """One denoise step during steady streaming (buffer full).

        Static shapes throughout — compiled into a single CUDA graph.
        """
        N, C = self.N, self.C
        buf_slots = N
        buf_len = buf_slots * C
        bsz = tokens.shape[0]
        device = tokens.device

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.prefix_embedder(
            images, img_masks, tokens, masks,
        )

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

        time_slot = torch.arange(1, N + 1, device=device, dtype=torch.float32) / N
        time_per_slot = time_slot.unsqueeze(0).expand(bsz, -1)

        v_t = self.denoise_step(
            prefix_pad_masks, prefix_att_masks, state,
            buffer, time_per_slot, padding_mask=None,
        )

        dt = 1.0 / N
        buffer = buffer - dt * v_t
        actions_to_execute = buffer[:, :C, :self.config.max_action_dim]
        new_buffer = self.sample_noise((bsz, buf_len, self.config.max_action_dim), device)
        new_buffer[:, :-C, :] = buffer[:, C:, :]
        buffer = new_buffer

        return actions_to_execute, buffer

    @torch.no_grad()
    def sample_actions(self, images, img_masks, tokens, masks, state, buffer, step_counter, profile=False):
        """Top-level dispatch: cold-start (eager) or steady streaming (compiled)."""
        N, C = self.N, self.C
        buf_slots = N
        buf_len = buf_slots * C
        bsz = tokens.shape[0]
        device = tokens.device

        if buffer is None:
            buffer = self.sample_noise((bsz, buf_len, self.config.max_action_dim), device)

        _evs_total = (
            [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            if profile else None
        )
        if profile:
            _ev_record(_evs_total[0])

        if step_counter < N - 1:
            self._cold_step_t.fill_(step_counter)
            buffer, breakdown = self._cold_start(
                images, img_masks, tokens, masks, state, buffer, self._cold_step_t,
                profile=profile,
            )
            actions_to_execute = None
        else:
            actions_to_execute, buffer = self._steady_streaming(
                images, img_masks, tokens, masks, state, buffer,
            )
            breakdown = None

        if profile:
            _ev_record(_evs_total[1])
            _cuda_sync()
            wall_total = _ev_elapsed_ms(_evs_total[0], _evs_total[1])
            if breakdown is not None:
                profile_results = breakdown
                profile_results["wall_total"] = wall_total
            else:
                profile_results = {"total": wall_total}
        else:
            profile_results = None

        return actions_to_execute, buffer, step_counter + 1, profile_results




class PI0FlashVLAPolicy(PreTrainedPolicy):
    """PI0 FlashVLA Policy wrapper."""

    config_class = PI0FlashVLAConfig
    name = "pi0-flashvla"

    def __init__(
        self,
        config: PI0FlashVLAConfig,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = PI0FlashVLAModel(config)

        buf_len = config.total_buffer_length
        self.register_buffer(
            "action_buffer",
            torch.zeros(1, buf_len, config.max_action_dim),
            persistent=False,
        )
        self._step_counter = 0
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

        prefix_rules: list[tuple[str, str]] = [
            ("model.action_in_proj.", "model.suffix_embedder.action_in_proj."),
            ("model.action_out_proj.", "model.action_out_proj."),
            ("model.action_time_mlp_in.", "model.suffix_embedder.action_time_mlp_in."),
            ("model.action_time_mlp_out.", "model.suffix_embedder.action_time_mlp_out."),
            ("model.state_proj.", "model.suffix_embedder.state_proj."),
            ("model.paligemma_with_expert.gemma_expert.", "model.action_expert."),
            ("model.paligemma_with_expert.paligemma.", "model.vlm."),
            ("action_in_proj.", "model.suffix_embedder.action_in_proj."),
            ("action_out_proj.", "model.action_out_proj."),
            ("action_time_mlp_in.", "model.suffix_embedder.action_time_mlp_in."),
            ("action_time_mlp_out.", "model.suffix_embedder.action_time_mlp_out."),
            ("state_proj.", "model.suffix_embedder.state_proj."),
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
            if new_key not in target_sd:
                continue
            if target_sd[new_key].shape != value.shape:
                continue
            mapped_sd[new_key] = value

        if getattr(config, "use_adarms_time_cond", False):
            from flashvla.policies.pi05.patches import FlashVLARMSNorm as PatchedGemmaRMSNorm
            from lerobot.policies.pi_gemma import (
                PiGemmaRMSNorm as OriginalGemmaRMSNorm,
            )
            n_zeroed = 0
            n_kept = 0
            ae_prefix = "model.action_expert."
            with torch.no_grad():
                for name, module in instance.model.action_expert.named_modules():
                    if type(module) is OriginalGemmaRMSNorm:
                        module.__class__ = PatchedGemmaRMSNorm
                    if isinstance(module, OriginalGemmaRMSNorm) and getattr(module, "dense", None) is not None:
                        dense_w_key = f"{ae_prefix}{name}.dense.weight"
                        if dense_w_key in mapped_sd:
                            n_kept += 1
                        else:
                            module.dense.weight.zero_()
                            if module.dense.bias is not None:
                                module.dense.bias.zero_()
                            n_zeroed += 1
                se = instance.model.suffix_embedder
                if hasattr(se, "state_mlp_out"):
                    if "model.suffix_embedder.state_mlp_out.weight" not in mapped_sd:
                        se.state_mlp_out.weight.zero_()
                        if se.state_mlp_out.bias is not None:
                            se.state_mlp_out.bias.zero_()
            logger.info(
                f"adaRMS mode: zero-init'd {n_zeroed} fresh FiLM dense layers "
                f"(DiT identity), {n_kept} layers will be loaded from ckpt"
            )

        incompatible = instance.load_state_dict(mapped_sd, strict=False)
        _missing_keys, unexpected_keys = incompatible.missing_keys, incompatible.unexpected_keys
        unexpected_fatal = list(unexpected_keys)
        if unexpected_fatal:
            raise RuntimeError(
                "Checkpoint loading failed.\n"
                f"Unexpected keys: {unexpected_fatal}"
            )

        instance.to(config.device)
        instance.eval()

        if getattr(config, "fuse_qkv", True):
            instance.model.init_qkv_fusion_from_existing()
        if getattr(config, "fuse_gate_up", True):
            instance.model.init_mlp_fusion_from_existing()

        instance._cold_start_stats = load_cold_start_stats(pretrained_name_or_path)

        return instance

    def reset(self):
        """Reset buffer and step counter. Call when environment resets.

        [noise, P, P, P, P]: slot 0 is noise, rest zeros (padding).
        """
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        C = self.config.chunk_size
        self.action_buffer.zero_()
        self.action_buffer[:, :C, :].normal_()
        self._step_counter = 0
        self._cold_start_action = None

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], profile=False) -> Tensor | None:
        """Predict a chunk of actions for inference (None during cold start)."""
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
            if profile:
                return None, profile_results
            return None

        original_action_dim = self.config.action_feature.shape[0]
        if profile:
            return actions[:, :, :original_action_dim].float(), profile_results
        return actions[:, :, :original_action_dim].float()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action. Returns cold-start fallback during warm-up."""
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
                if mode == "zero_delta":
                    self._cold_start_action = cold_action
                for _ in range(self.config.n_action_steps):
                    self._action_queue.append(cold_action)
            else:
                self._action_queue.extend(actions.transpose(0, 1)[:self.config.n_action_steps])
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor], noise=None) -> tuple[Tensor, dict[str, Tensor]]:
        """Training forward pass."""
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
        loss = losses.mean()
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
        return pad_vector(batch[OBS_STATE], self.config.max_state_dim)

    def prepare_action(self, batch):
        return pad_vector(batch[ACTION], self.config.max_action_dim)
