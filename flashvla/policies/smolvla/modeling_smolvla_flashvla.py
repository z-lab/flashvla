#!/usr/bin/env python

# Copyright 2025 FlashVLA team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""SmolVLA FlashVLA.

Port of pi0/pi05 streaming buffer + shared-observation training to smolvla.

Architecture adaptations vs pi0 streaming:
  * State stays in the prefix (smolvla's native layout). One prefix encoding
    is shared across all N buffer configs in training — no per-config state
    token duplication like pi0/lingbot streaming.
  * Suffix uses pi05/pi0-style block-causal across slots: only the first
    token of each slot is a boundary (``att_masks=[1]+[0]*(C-1)`` per slot).
    Within a slot (C action tokens): full attention. Across slots: causal
    at the block level. Diverges from baseline smolvla's per-token AR within
    the action chunk (``modeling_smolvla.py:717``) — empirical results on
    pi05 ([[project_flashvla_outlier_amplification]]) recommend
    this pattern for streaming.
  * Cross-attention mode (default in smolvla): the action expert cross-attends
    to VLM KV. Cross-config blocking only meaningfully constrains self-attn
    layers (every ``self_attn_every_n_layers``). The same full attention mask
    works for both since cross-attn layers slice only suffix→prefix entries
    which are unaffected by cross-config blocking.

Buffer layout (N=5 slots, C=10 actions per slot, no action prefix):
  cold-start step 0: [4, P, P, P, P]
  cold-start step 4: [0, 1, 2, 3, 4]  → buffer full, first chunk extracted
  steady streaming:  one denoise step per call, slot 0 popped, slots shifted left.
"""
from __future__ import annotations

import builtins
import logging
import os
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from lerobot.configs.policies import PreTrainedConfig, T
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from flashvla.policies.smolvla.configuration_smolvla import SmolVLAFlashVLAConfig
from flashvla.policies.smolvla.modeling_smolvla import (
    VLAFlowMatching,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad,
)
# Cross-import policy-agnostic cold-start helpers + per-token sinusoidal embed
# (already used by pi0/pi05 streaming; same postprocessor safetensors format).
from flashvla.policies.pi05.utils import (
    ColdStartStats,
    compute_cold_start_action,
    create_sinusoidal_pos_embedding_for_blocks,
    load_cold_start_stats,
)
from flashvla.policies.pi0.utils import (
    build_flashvla_attention_mask_and_position_ids,
)


logger = logging.getLogger(__name__)


# ============================================================
# adaRMS for the Llama-based action expert
# ============================================================
#
# SmolVLA's action expert is SmolLM2 (Llama 3-style architecture), so its
# RMSNorm class is ``transformers.models.llama.modeling_llama.LlamaRMSNorm``,
# NOT Gemma's. Unlike pi05's setup — where the Gemma decoder is flashvla-patched
# at install time to bake ``use_adarms`` / ``cond_dim`` / ``dense`` into the
# constructor — we leave ``transformers`` untouched and SWAP RMSNorm instances
# in ``SmolVLAFlashVLAPolicy.from_pretrained`` (right before
# ``load_state_dict``).
#
# Llama RMSNorm formula: ``normed * weight`` (weight init 1.0). Our
# ``LlamaAdaRMSNorm`` preserves the original weight (so smolvla_base's learned
# scale survives) and stacks FiLM ``(1 + scale) + shift`` modulation on top:
#
#     out = normed * weight * (1 + scale) + shift
#
# With zero-init ``dense`` (scale = shift = 0 at step 0), this collapses to
# the baseline Llama RMSNorm — i.e. DiT identity at training start.
#
# No gate: ``SmolVLMWithExpertModel.forward``'s residual connection is a plain
# ``+=`` (smolvlm_with_expert.py:487, 497) with no per-block gating, so a gate
# would be discarded. ``dense`` is sized ``2*D`` (scale + shift) instead of
# pi05's ``3*D``.


class LlamaAdaRMSNorm(nn.Module):
    """Llama RMSNorm + FiLM (scale, shift) modulation.

    Args:
        weight: ``nn.Parameter`` from the original ``LlamaRMSNorm`` (reused,
            not copied — load_state_dict on the model still finds it under
            the same key path).
        variance_epsilon: ``self.variance_epsilon`` from the original module.
        cond_dim: dim of the conditioning vector (typically the expert's
            hidden size; matches pi0's design where time_mlp output is D→D).
    """

    def __init__(self, weight: nn.Parameter, variance_epsilon: float, cond_dim: int):
        super().__init__()
        self.weight = weight  # reused nn.Parameter from baseline LlamaRMSNorm
        self.variance_epsilon = variance_epsilon
        self.cond_dim = cond_dim
        dim = weight.numel()
        # 2*D: scale + shift (no gate; see module-level note above).
        self.dense = nn.Linear(cond_dim, dim * 2, bias=True)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def _norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # fp32 norm (matches LlamaRMSNorm.forward and pi05's GemmaRMSNorm._norm).
        h = hidden_states.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        return h * torch.rsqrt(variance + self.variance_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normed = self._norm(hidden_states)
        normed = normed * self.weight.float()  # standard Llama scale

        if cond is None:
            return normed.to(input_dtype)

        if cond.shape[-1] != self.cond_dim:
            raise ValueError(
                f"LlamaAdaRMSNorm: expected cond dim {self.cond_dim}, got {cond.shape[-1]}"
            )

        modulation = self.dense(cond.to(self.dense.weight.dtype))
        # Cond may be per-sample [B, D] or per-token [B, L, D]. Broadcast the
        # per-sample case to match the token axis of normed [B, L, D].
        if hidden_states.ndim == 3 and modulation.ndim == 2:
            modulation = modulation.unsqueeze(1)
        scale, shift = torch.chunk(modulation, 2, dim=-1)
        normed = normed * (1.0 + scale.to(torch.float32)) + shift.to(torch.float32)
        return normed.to(input_dtype)


# ============================================================
# Core model
# ============================================================


class VLAFlowMatchingFlashVLA(VLAFlowMatching):
    """SmolVLA flow-matching backbone with per-slot time + buffer streaming."""

    def __init__(self, config: SmolVLAFlashVLAConfig):
        super().__init__(config)
        self.N = config.num_buffer_slots
        self.C = config.chunk_size

        # Persistent 0-d int64 buffer for cold-start step counter — same
        # trick as pi05/pi0 streaming. Enables single CUDA-graph capture
        # across all cold-start step values.
        self.register_buffer("_cold_step_t", torch.zeros((), dtype=torch.int64), persistent=False)

        # adaRMS — purely additive on top of smolvla_base's concat-time path.
        # See ``SmolVLAFlashVLAConfig.use_adarms_time_cond`` docstring.
        if config.use_adarms_time_cond:
            D = self.vlm_with_expert.expert_hidden_size
            self.time_mlp_in = nn.Linear(D, D)
            self.time_mlp_out = nn.Linear(D, D)
        
        compile_model = config.compile_model
        
        if compile_model:
            self._cold_start = torch.compile(self._cold_start, mode=config.compile_mode)
            self._steady_streaming = torch.compile(self._steady_streaming, mode=config.compile_mode)
    # ------------------------------------------------------------------
    # Suffix embedder: per-slot time, pi05/pi0-style block-causal across slots
    # ------------------------------------------------------------------

    def embed_suffix(  # type: ignore[override]
        self,
        noisy_actions: Tensor,
        time_per_slot: Tensor,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Embed action tokens with per-slot time.

        Args:
            noisy_actions: ``[B, L_act, max_action_dim]`` where ``L_act`` is a
                multiple of ``C``.
            time_per_slot: ``[B, num_slots]`` per-slot times (``num_slots = L_act // C``).
                Broadcast to per-token via ``repeat_interleave(C)`` internally.
            padding_mask: optional ``[B, L_act]`` boolean. True = real action
                token, False = padded slot — padded embeddings are zeroed.

        Returns:
            embs: ``[B, L_act, expert_hidden_size]`` — token embedding
                (same in baseline & adaRMS modes — keeps smolvla_base's
                concat-time path intact).
            pad_masks: ``[B, L_act]``
            att_masks: ``[B, L_act]`` — pi05/pi0-style block-causal: only the
                first token of each slot is a boundary (1), the remaining C-1
                tokens are non-boundary (0). Within a slot: full attention;
                across slots: causal at the block level.
            adarms_cond: ``[B, L_act, D]`` per-token FiLM conditioning when
                ``config.use_adarms_time_cond=True``; ``None`` otherwise.
                Option (A) — time-only, state stays in prefix.
        """
        bsize = noisy_actions.shape[0]
        device = noisy_actions.device
        C = self.C
        L_act = noisy_actions.shape[1]
        num_slots = L_act // C
        assert num_slots * C == L_act, f"L_act={L_act} not divisible by C={C}"

        # Per-token time via repeat_interleave
        time_per_token = time_per_slot.repeat_interleave(C, dim=1)  # [B, L_act]

        # Sinusoidal embedding (per-token, 2D input)
        time_emb = create_sinusoidal_pos_embedding_for_blocks(
            time_per_token,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        action_emb = self.action_in_proj(noisy_actions)
        dtype = action_emb.dtype
        time_emb = time_emb.to(dtype=dtype)

        # ===== Token embedding (same in BOTH modes — smolvla baseline path) =====
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Zero out padded action tokens; pad_masks reflects that
        if padding_mask is not None:
            action_time_emb = action_time_emb * padding_mask.unsqueeze(-1).to(action_time_emb.dtype)
            pad_masks = padding_mask
        else:
            pad_masks = torch.ones(bsize, L_act, dtype=torch.bool, device=device)

        embs = action_time_emb  # [B, L_act, expert_hidden_size]

        # pi05/pi0-style block-causal across slots: only the first token of each
        # slot is a block boundary (att_mask=1), the remaining C-1 tokens are
        # non-boundary (att_mask=0). Combined with ``make_att_2d_masks`` /
        # ``build_flashvla_attention_mask_and_position_ids`` which both
        # do ``cumsum -> cumsum[None,:] <= cumsum[:,None]``:
        #   * within a slot (C tokens) — same cumsum value, full attention.
        #   * across slots — slot k+1 has higher cumsum, sees slot k (block-causal).
        # Departs from baseline smolvla's per-token AR within the action chunk
        # (``modeling_smolvla.py:717``) but matches the established pi05 streaming
        # recipe ([[project_flashvla_outlier_amplification]]).
        slot_att = torch.zeros(num_slots, C, dtype=torch.bool, device=device)
        slot_att[:, 0] = True
        att_masks = slot_att.flatten(0, 1).unsqueeze(0).expand(bsize, -1).contiguous()

        # ===== adaRMS cond (option A: time-only) =====
        adarms_cond = None
        if self.config.use_adarms_time_cond:
            time_cond = self.time_mlp_in(time_emb)
            time_cond = F.silu(time_cond)
            time_cond = self.time_mlp_out(time_cond)
            time_cond = F.silu(time_cond)  # [B, L_act, D]
            if padding_mask is not None:
                time_cond = time_cond * padding_mask.unsqueeze(-1).to(time_cond.dtype)
            adarms_cond = time_cond

        return embs, pad_masks, att_masks, adarms_cond

    # ------------------------------------------------------------------
    # Shared-obs time sampling (PerSeg / per-chunk) — same as pi0 streaming
    # ------------------------------------------------------------------

    def _sample_global_slot_times(self, bsize: int, device: torch.device) -> Tensor:
        """Sample one global t per slot level, shared across all N configs."""
        N = self.N
        beta_dist = torch.distributions.Beta(
            concentration1=torch.tensor(1.5, device=device, dtype=torch.float32),
            concentration0=torch.tensor(1.0, device=device, dtype=torch.float32),
        )
        samples = beta_dist.sample((bsize, N)).to(device=device, dtype=torch.float32)
        level_indices = torch.arange(N, device=device, dtype=torch.float32)
        seg_starts = level_indices / N
        global_time = seg_starts[None, :] + samples / N
        return global_time * 0.999 + 0.001  # [B, N]

    def _build_per_sample_time(
        self, batch_size: int, device: torch.device, use_prefix: bool,
    ) -> tuple[Tensor, Tensor]:
        """PerSeg time: one global t per slot level shared across N configs.

        Returns:
            time_per_slot_all_configs: ``[B, N * num_slots_per_config]`` flat
                in config-major order. Padded slots get t=1.0 (placeholder,
                will be masked by padding_mask downstream).
            time_per_token: ``[B, N * num_slots_per_config * C]`` per-token.
        """
        N, C = self.N, self.C
        num_slots_per_config = (N + 1) if use_prefix else N

        global_time = self._sample_global_slot_times(batch_size, device)  # [B, N]
        if use_prefix:
            prefix_time = torch.zeros(batch_size, 1, device=device, dtype=global_time.dtype)
            time_per_segment = torch.cat([prefix_time, global_time], dim=1)  # [B, N+1]
        else:
            time_per_segment = global_time

        # lookup[k_idx, s] = which segment maps to config k's slot s
        lookup = torch.zeros(N, num_slots_per_config, dtype=torch.long, device=device)
        if use_prefix:
            for k_idx in range(N):
                k = k_idx + 1
                lookup[k_idx, 0] = 0  # prefix segment (t=0)
                lookup[k_idx, 1:1 + k] = torch.arange(N - k + 1, N + 1, device=device)
        else:
            for k_idx in range(N):
                k = k_idx + 1
                lookup[k_idx, :k] = torch.arange(N - k, N, device=device)

        segment_lookup = lookup.reshape(-1).unsqueeze(0).expand(batch_size, -1).contiguous()
        time_per_slot = torch.gather(time_per_segment, 1, segment_lookup)
        time_per_token = time_per_slot.repeat_interleave(C, dim=1)
        return time_per_slot, time_per_token

    def _build_per_chunk_time(
        self, batch_size: int, device: torch.device, use_prefix: bool,
    ) -> tuple[Tensor, Tensor]:
        """Per-chunk: each config independently samples its t. Ablation only."""
        N, C = self.N, self.C
        num_slots_per_config = (N + 1) if use_prefix else N

        beta_dist = torch.distributions.Beta(
            concentration1=torch.tensor(1.5, device=device, dtype=torch.float32),
            concentration0=torch.tensor(1.0, device=device, dtype=torch.float32),
        )
        samples = beta_dist.sample(
            (batch_size, N, num_slots_per_config),
        ).to(device=device, dtype=torch.float32)

        seg_start = torch.zeros(N, num_slots_per_config, device=device, dtype=torch.float32)
        is_real = torch.zeros(N, num_slots_per_config, device=device, dtype=torch.bool)
        is_prefix = torch.zeros(N, num_slots_per_config, device=device, dtype=torch.bool)
        for k_idx in range(N):
            num_real = k_idx + 1
            offset = 1 if use_prefix else 0
            if use_prefix:
                is_prefix[k_idx, 0] = True
            for i in range(num_real):
                seg_start[k_idx, offset + i] = (N - num_real + i) / N
                is_real[k_idx, offset + i] = True

        time_per_slot = seg_start[None, :, :] + samples / N
        time_per_slot = time_per_slot * 0.999 + 0.001
        time_per_slot = torch.where(is_real[None, :, :], time_per_slot, torch.ones_like(time_per_slot))
        time_per_slot = torch.where(is_prefix[None, :, :], torch.zeros_like(time_per_slot), time_per_slot)
        time_per_slot = time_per_slot.reshape(batch_size, -1)  # [B, N*S]
        time_per_token = time_per_slot.repeat_interleave(C, dim=1)
        return time_per_slot, time_per_token

    # ------------------------------------------------------------------
    # Training: shared-observation forward (N configs side-by-side)
    # ------------------------------------------------------------------

    def forward_shared_observation(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        action_is_pad,
        noise=None,
    ):
        """N buffer configs side-by-side, one shared prefix.

        Args:
            state: ``[B, max_state_dim]`` (shared across all N configs).
            actions: ``[B, N * H_cfg, action_dim]`` config-major.
                H_cfg = (N+1)*C with prefix, N*C without.
            action_is_pad: ``[B, N * H_cfg]`` boolean.
        """
        bsize = state.shape[0]
        N, C = self.N, self.C
        use_prefix = self.config.use_action_prefix
        H_cfg = (N + 1) * C if use_prefix else N * C
        device = state.device

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        mode = getattr(self.config, "timestep_sample_mode", "per-sample")
        if mode == "per-sample":
            time_per_slot_all, time_per_token = self._build_per_sample_time(bsize, device, use_prefix)
        elif mode == "per-chunk":
            time_per_slot_all, time_per_token = self._build_per_chunk_time(bsize, device, use_prefix)
        else:
            raise ValueError(f"Unknown timestep_sample_mode={mode!r}")

        time_expanded = time_per_token.unsqueeze(-1)
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Shared prefix — encode once
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state,
        )

        # Per-config suffix: flatten (B, N) into batch dim for the embedder
        num_slots_per_config = (N + 1) if use_prefix else N
        x_t_flat = x_t.view(bsize * N, H_cfg, -1)
        time_per_slot_flat = time_per_slot_all.view(bsize * N, num_slots_per_config)
        if action_is_pad is not None:
            real_mask_flat = (~action_is_pad).view(bsize * N, H_cfg)
        else:
            real_mask_flat = None

        embs_flat, pad_flat, att_flat, cond_flat = self.embed_suffix(
            x_t_flat, time_per_slot_flat, padding_mask=real_mask_flat,
        )
        # embs_flat: [B*N, H_cfg, D]; reshape to [B, N*H_cfg, D]
        suffix_embs = embs_flat.view(bsize, N, H_cfg, -1).reshape(bsize, N * H_cfg, -1)
        suffix_pad_per_offset = pad_flat.view(bsize, N, H_cfg)
        suffix_att_masks = att_flat[:bsize]  # representative (same pattern for all configs)
        # adaRMS cond — flatten [B*N, H_cfg, D] → [B, N*H_cfg, D] mirroring suffix_embs.
        if cond_flat is not None:
            suffix_adarms_cond = cond_flat.view(bsize, N, H_cfg, -1).reshape(bsize, N * H_cfg, -1)
        else:
            suffix_adarms_cond = None

        # Attention mask: pi0 streaming helper handles cumsum-based block-causal
        # + cross-config blocking. With our att=[1,0,0,...]_C pattern (boundary
        # only at each slot's first token), cumsum gives block-causal across
        # slots while keeping full intra-slot attention — pi05/pi0 streaming
        # semantics, replicated N times with cross-config isolation.
        # Cast prefix_att_masks to long (smolvla emits bool; helper does cumsum on int).
        # pi0 helper returns an additive mask [B, 1, L, L] with 0 (attendable)
        # or -inf (blocked) — needs a float dtype for torch.finfo.min. smolvla's
        # eager_attention_forward expects bool [B, L, L] via torch.where; convert.
        prefix_att_long = prefix_att_masks.to(dtype=torch.long)
        suffix_att_long = suffix_att_masks.to(dtype=torch.long)
        additive_mask, position_ids = build_flashvla_attention_mask_and_position_ids(
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_long,
            suffix_pad_masks_per_offset=suffix_pad_per_offset,
            suffix_att_masks=suffix_att_long,
            num_offsets=N,
            dtype=torch.float32,
        )
        attention_mask = (additive_mask == 0).squeeze(1)  # [B, total_len, total_len] bool

        # Run joint VLM+expert
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
            adarms_cond=suffix_adarms_cond,
        )

        # Slice action outputs per config: suffix_out [B, N*H_cfg, D]
        suffix_out = suffix_out.view(bsize, N, H_cfg, -1)
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)  # [B, N, H_cfg, action_dim]
        v_t = v_t.reshape(bsize, N * H_cfg, -1)

        losses = F.mse_loss(u_t, v_t, reduction="none")

        # Loss mask: exclude padded slots and (if use_prefix) the prefix tokens
        if action_is_pad is not None:
            loss_mask = ~action_is_pad
        else:
            loss_mask = torch.ones(bsize, N * H_cfg, dtype=torch.bool, device=device)
        if use_prefix:
            prefix_exclude = torch.zeros(bsize, N * H_cfg, dtype=torch.bool, device=device)
            for k_idx in range(N):
                start = k_idx * H_cfg
                prefix_exclude[:, start:start + C] = True
            loss_mask = loss_mask & ~prefix_exclude

        losses = losses[loss_mask]
        return losses

    # ------------------------------------------------------------------
    # Inference: denoise_step + cold_start + steady_streaming
    # ------------------------------------------------------------------

    @torch.no_grad()
    def denoise_step(  # type: ignore[override]
        self,
        prefix_pad_masks: Tensor,
        past_key_values,
        x_t: Tensor,
        time_per_slot: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Single denoise step with KV-cached prefix. Returns velocity v_t."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, suffix_adarms_cond = self.embed_suffix(
            x_t, time_per_slot, padding_mask=padding_mask,
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        # Suffix can attend to all real prefix tokens
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        # Within suffix: AR causal masked by padding
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            adarms_cond=suffix_adarms_cond,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)  # [B, suffix_len, action_dim]

    @torch.no_grad()
    def _encode_and_prefill(self, images, img_masks, lang_tokens, lang_masks, state):
        """Run prefix forward once, populate KV cache.

        Returns (prefix_pad_masks, past_key_values).
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        return prefix_pad_masks, past_key_values

    @torch.no_grad()
    def _cold_start(
        self, images, img_masks, lang_tokens, lang_masks, state, buffer, step_counter,
    ):
        """One denoise step during cold-start (buffer not yet full).

        step_counter is a 0-d int64 tensor; vectorized time/padding-mask
        construction keeps shapes static for compile compatibility.
        """
        N, C = self.N, self.C
        use_prefix = self.config.use_action_prefix
        buf_slots = N + 1 if use_prefix else N
        buf_len = buf_slots * C
        bsz = buffer.shape[0]
        device = buffer.device

        prefix_pad_masks, past_key_values = self._encode_and_prefill(
            images, img_masks, lang_tokens, lang_masks, state,
        )

        num_real = step_counter + 1
        slot_idx = torch.arange(buf_slots, device=device)
        slot_idx_buf = torch.arange(buf_len, device=device) // C

        if use_prefix:
            is_prefix = slot_idx == 0
            is_real = slot_idx <= num_real
            time_noisy = (N - num_real + slot_idx).to(torch.float32) / N
            time_slot = torch.where(is_prefix, torch.zeros_like(time_noisy), time_noisy)
            time_slot = torch.where(is_real, time_slot, torch.ones_like(time_slot))
        else:
            is_real = slot_idx < num_real
            time_slot = (N - num_real + 1 + slot_idx).to(torch.float32) / N
            time_slot = torch.where(is_real, time_slot, torch.ones_like(time_slot))

        time_per_slot = time_slot.unsqueeze(0).expand(bsz, -1)
        padding_mask = is_real.unsqueeze(0).unsqueeze(2).expand(bsz, -1, C).reshape(bsz, buf_len)

        v_t = self.denoise_step(
            prefix_pad_masks, past_key_values, buffer, time_per_slot, padding_mask=padding_mask,
        )

        dt = 1.0 / N
        new_noise = self.sample_noise((bsz, buf_len, self.config.max_action_dim), device)
        if use_prefix:
            prefix_part = buffer[:, :C, :]
            noisy_part = buffer[:, C:, :] - dt * v_t[:, C:, :]
            updated_buffer = torch.cat([prefix_part, noisy_part], dim=1)
            keep_mask = slot_idx_buf <= num_real
            noise_mask = slot_idx_buf == (num_real + 1)
        else:
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
        return buffer

    @torch.no_grad()
    def _steady_streaming(
        self, images, img_masks, lang_tokens, lang_masks, state, buffer,
    ):
        """One denoise step during steady streaming (buffer full).

        Static shapes throughout. Compiled into a single CUDA graph when
        ``config.compile_model=True``.
        """
        N, C = self.N, self.C
        use_prefix = self.config.use_action_prefix
        buf_slots = N + 1 if use_prefix else N
        buf_len = buf_slots * C
        bsz = buffer.shape[0]
        device = buffer.device

        prefix_pad_masks, past_key_values = self._encode_and_prefill(
            images, img_masks, lang_tokens, lang_masks, state,
        )

        if use_prefix:
            time_per_slot = torch.zeros(bsz, buf_slots, device=device, dtype=torch.float32)
            time_per_slot[:, 1:] = torch.arange(1, N + 1, device=device, dtype=torch.float32) / N
        else:
            time_slot = torch.arange(1, N + 1, device=device, dtype=torch.float32) / N
            time_per_slot = time_slot.unsqueeze(0).expand(bsz, -1)

        v_t = self.denoise_step(
            prefix_pad_masks, past_key_values, buffer, time_per_slot, padding_mask=None,
        )

        dt = 1.0 / N
        if use_prefix:
            buffer[:, C:, :] = buffer[:, C:, :] - dt * v_t[:, C:, :]
            actions_to_execute = buffer[:, C:2 * C, :self.config.max_action_dim]
            new_buffer = self.sample_noise((bsz, buf_len, self.config.max_action_dim), device)
            new_buffer[:, :C, :] = buffer[:, C:2 * C, :]
            new_buffer[:, C:N * C, :] = buffer[:, 2 * C:, :]
            buffer = new_buffer
        else:
            buffer = buffer - dt * v_t
            actions_to_execute = buffer[:, :C, :self.config.max_action_dim]
            new_buffer = self.sample_noise((bsz, buf_len, self.config.max_action_dim), device)
            new_buffer[:, :-C, :] = buffer[:, C:, :]
            buffer = new_buffer

        return actions_to_execute, buffer

    @torch.no_grad()
    def sample_actions(  # type: ignore[override]
        self, images, img_masks, lang_tokens, lang_masks, state,
        buffer, step_counter, noise=None,
    ):
        """Top-level streaming dispatch.

        Returns (actions_to_execute [B, C, action_dim] or None, updated_buffer, new_step_counter).
        """
        N, C = self.N, self.C
        use_prefix = self.config.use_action_prefix
        buf_slots = N + 1 if use_prefix else N
        buf_len = buf_slots * C
        bsz = state.shape[0]
        device = state.device

        if buffer is None:
            if use_prefix:
                buffer = torch.zeros(bsz, buf_len, self.config.max_action_dim, device=device)
                buffer[:, C:2 * C, :] = self.sample_noise((bsz, C, self.config.max_action_dim), device)
            else:
                buffer = self.sample_noise((bsz, buf_len, self.config.max_action_dim), device)

        if step_counter < N - 1:
            self._cold_step_t.fill_(step_counter)
            buffer = self._cold_start(
                images, img_masks, lang_tokens, lang_masks, state, buffer, self._cold_step_t,
            )
            actions_to_execute = None
        else:
            actions_to_execute, buffer = self._steady_streaming(
                images, img_masks, lang_tokens, lang_masks, state, buffer,
            )

        return actions_to_execute, buffer, step_counter + 1


# ============================================================
# Policy wrapper
# ============================================================


class SmolVLAFlashVLAPolicy(PreTrainedPolicy):
    """SmolVLA FlashVLA policy wrapper."""

    config_class = SmolVLAFlashVLAConfig
    name = "smolvla-flashvla"

    def __init__(self, config: SmolVLAFlashVLAConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = VLAFlowMatchingFlashVLA(config)

        # Pre-allocate streaming buffer (action slots only; state is per-call in prefix)
        buf_len = config.total_buffer_length
        self.register_buffer(
            "action_buffer",
            torch.zeros(1, buf_len, config.max_action_dim),
            persistent=False,
        )
        self._step_counter = 0
        self._cold_start_stats = ColdStartStats()
        self.reset()

    def reset(self):
        """Reset queue, buffer, step counter."""
        self._queues = {ACTION: deque([], maxlen=self.config.n_action_steps)}
        C = self.config.chunk_size
        self.action_buffer.zero_()
        if self.config.use_action_prefix:
            self.action_buffer[:, C:2 * C, :].normal_()
        else:
            self.action_buffer[:, :C, :].normal_()
        self._step_counter = 0
        self._cold_start_action = None

    def get_optim_params(self) -> dict:
        return self.parameters()

    # ------------------------------------------------------------------
    # Input prep — mirror baseline SmolVLAPolicy
    # ------------------------------------------------------------------

    def prepare_images(self, batch):
        images, img_masks = [], []
        present = [k for k in self.config.image_features if k in batch]
        missing = [k for k in self.config.image_features if k not in batch]
        if not present:
            raise ValueError(
                f"All image features missing. Expected at least one of {list(self.config.image_features)}. "
                f"Got: {list(batch.keys())}"
            )
        for key in present:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            img = img * 2.0 - 1.0  # [-1, 1] for siglip
            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)
        for _ in range(min(len(missing), self.config.empty_cameras)):
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def prepare_state(self, batch):
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        return pad_vector(state, self.config.max_state_dim)

    def prepare_action(self, batch):
        return pad_vector(batch[ACTION], self.config.max_action_dim)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise=None) -> Tensor | None:
        """Predict an action chunk via streaming. Returns None during cold start."""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions, buffer, new_step = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            buffer=self.action_buffer,
            step_counter=self._step_counter,
            noise=noise,
        )
        self.action_buffer.copy_(buffer)
        self._step_counter = new_step

        if actions is None:
            return None

        original_action_dim = self.config.action_feature.shape[0]
        return actions[:, :, :original_action_dim].float()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Single action with cold-start fallback during warm-up."""
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        mode = getattr(self.config, "cold_start_mode", "zero_delta")
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            if actions is None:
                cold = compute_cold_start_action(
                    mode=mode,
                    stats=self._cold_start_stats,
                    action_dim=self.config.action_feature.shape[0],
                    device=self.action_buffer.device,
                    state_normalized=batch[OBS_STATE] if mode == "current_state" else None,
                    cache=self._cold_start_action if mode == "zero_delta" else None,
                )
                if mode == "zero_delta":
                    self._cold_start_action = cold
                for _ in range(self.config.n_action_steps):
                    self._queues[ACTION].append(cold)
            else:
                self._queues[ACTION].extend(actions.transpose(0, 1)[:self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor], noise=None) -> tuple[Tensor, dict]:
        """Training forward: shared-observation flow-matching MSE."""
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        action_is_pad = batch.get("action_is_pad")

        losses = self.model.forward_shared_observation(
            images, img_masks, lang_tokens, lang_masks,
            state, actions, action_is_pad, noise,
        )
        losses = losses[..., :self.config.max_action_dim]
        loss = losses.mean()
        return loss, {"loss": loss.item()}

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

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
        """Load weights from a baseline smolvla checkpoint.

        Streaming model uses the same parameter names as the baseline
        ``VLAFlowMatching`` (we subclass it), so standard ``load_state_dict``
        works without any prefix remapping.
        """
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

        if os.path.isdir(pretrained_name_or_path):
            model_file = os.path.join(pretrained_name_or_path, "model.safetensors")
            if not os.path.isfile(model_file):
                raise FileNotFoundError(f"No 'model.safetensors' in {model_file}")
        else:
            model_file = cached_file(
                pretrained_name_or_path, "model.safetensors",
                cache_dir=cache_dir, force_download=force_download,
                resume_download=resume_download, proxies=proxies,
                token=token, revision=revision, local_files_only=local_files_only,
            )
            if model_file is None:
                raise FileNotFoundError(f"Could not resolve model.safetensors for {pretrained_name_or_path}")

        state_dict = load_file(model_file)

        # adaRMS: swap the expert's ``LlamaRMSNorm`` instances with
        # ``LlamaAdaRMSNorm`` BEFORE load_state_dict, so the ckpt's
        # ``...dense.weight`` / ``...dense.bias`` keys land on the new
        # FiLM layers. The original ``weight`` ``nn.Parameter`` is reused
        # (state_dict path stays stable). For warm-start from smolvla_base
        # (no dense.* keys present), the zero-init we set at swap time
        # survives → DiT identity at step 0.
        if getattr(config, "use_adarms_time_cond", False):
            D = instance.model.vlm_with_expert.expert_hidden_size
            expert = instance.model.vlm_with_expert.lm_expert
            norm_sites: list[tuple[nn.Module, str]] = []
            for layer in expert.layers:
                norm_sites.append((layer, "input_layernorm"))
                norm_sites.append((layer, "post_attention_layernorm"))
            norm_sites.append((expert, "norm"))
            n_swapped = 0
            for parent, attr_name in norm_sites:
                old = getattr(parent, attr_name, None)
                if old is None or isinstance(old, LlamaAdaRMSNorm) or not isinstance(old, LlamaRMSNorm):
                    continue
                new_norm = LlamaAdaRMSNorm(
                    weight=old.weight,
                    variance_epsilon=old.variance_epsilon,
                    cond_dim=D,
                )
                # Match device & dtype of the surrounding expert (typically bf16).
                new_norm = new_norm.to(device=old.weight.device, dtype=old.weight.dtype)
                setattr(parent, attr_name, new_norm)
                n_swapped += 1
            logger.info(
                f"SmolVLA adaRMS: swapped {n_swapped} LlamaRMSNorm modules in expert "
                f"(cond_dim={D}); dense layers zero-init for DiT identity."
            )

        incompatible = instance.load_state_dict(state_dict, strict=False)
        # Filter out lm_head (intentionally trimmed) from missing-key warnings.
        missing = [k for k in incompatible.missing_keys if "lm_head" not in k]
        unexpected = list(incompatible.unexpected_keys)
        if unexpected:
            logger.warning(f"Unexpected keys when loading smolvla streaming: {unexpected[:5]}...")
        if missing:
            logger.warning(f"Missing keys: {missing[:5]}...")

        instance.to(config.device)
        instance.eval()

        # Cold-start stats for the N-1 warm-up calls
        instance._cold_start_stats = load_cold_start_stats(pretrained_name_or_path)

        return instance
