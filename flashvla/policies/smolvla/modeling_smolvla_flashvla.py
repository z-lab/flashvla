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
    the action chunk (see ``modeling_smolvla.embed_suffix``) — empirical results on
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
        self.weight = weight
        self.variance_epsilon = variance_epsilon
        self.cond_dim = cond_dim
        dim = weight.numel()
        self.dense = nn.Linear(cond_dim, dim * 2, bias=True)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def _norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
        normed = normed * self.weight.float()

        if cond is None:
            return normed.to(input_dtype)

        if cond.shape[-1] != self.cond_dim:
            raise ValueError(
                f"LlamaAdaRMSNorm: expected cond dim {self.cond_dim}, got {cond.shape[-1]}"
            )

        modulation = self.dense(cond.to(self.dense.weight.dtype))
        if hidden_states.ndim == 3 and modulation.ndim == 2:
            modulation = modulation.unsqueeze(1)
        scale, shift = torch.chunk(modulation, 2, dim=-1)
        normed = normed * (1.0 + scale.to(torch.float32)) + shift.to(torch.float32)
        return normed.to(input_dtype)




class VLAFlowMatchingFlashVLA(VLAFlowMatching):
    """SmolVLA flow-matching backbone with per-slot time + buffer streaming."""

    def __init__(self, config: SmolVLAFlashVLAConfig):
        super().__init__(config)
        self.N = config.num_buffer_slots
        self.C = config.chunk_size

        self.register_buffer("_cold_step_t", torch.zeros((), dtype=torch.int64), persistent=False)

        if config.use_adarms_time_cond:
            D = self.vlm_with_expert.expert_hidden_size
            self.time_mlp_in = nn.Linear(D, D)
            self.time_mlp_out = nn.Linear(D, D)
        
        compile_model = config.compile_model
        
        if compile_model:
            self._cold_start = torch.compile(self._cold_start, mode=config.compile_mode)
            self._steady_streaming = torch.compile(self._steady_streaming, mode=config.compile_mode)

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

        time_per_token = time_per_slot.repeat_interleave(C, dim=1)

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

        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        if padding_mask is not None:
            action_time_emb = action_time_emb * padding_mask.unsqueeze(-1).to(action_time_emb.dtype)
            pad_masks = padding_mask
        else:
            pad_masks = torch.ones(bsize, L_act, dtype=torch.bool, device=device)

        embs = action_time_emb

        slot_att = torch.zeros(num_slots, C, dtype=torch.bool, device=device)
        slot_att[:, 0] = True
        att_masks = slot_att.flatten(0, 1).unsqueeze(0).expand(bsize, -1).contiguous()

        adarms_cond = None
        if self.config.use_adarms_time_cond:
            time_cond = self.time_mlp_in(time_emb)
            time_cond = F.silu(time_cond)
            time_cond = self.time_mlp_out(time_cond)
            time_cond = F.silu(time_cond)
            if padding_mask is not None:
                time_cond = time_cond * padding_mask.unsqueeze(-1).to(time_cond.dtype)
            adarms_cond = time_cond

        return embs, pad_masks, att_masks, adarms_cond


    def _build_per_chunk_time(
        self, batch_size: int, device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Per-chunk: each config independently samples its t."""
        N, C = self.N, self.C
        num_slots_per_config = N

        beta_dist = torch.distributions.Beta(
            concentration1=torch.tensor(1.5, device=device, dtype=torch.float32),
            concentration0=torch.tensor(1.0, device=device, dtype=torch.float32),
        )
        samples = beta_dist.sample(
            (batch_size, N, num_slots_per_config),
        ).to(device=device, dtype=torch.float32)

        seg_start = torch.zeros(N, num_slots_per_config, device=device, dtype=torch.float32)
        is_real = torch.zeros(N, num_slots_per_config, device=device, dtype=torch.bool)
        for k_idx in range(N):
            num_real = k_idx + 1
            for i in range(num_real):
                seg_start[k_idx, i] = (N - num_real + i) / N
                is_real[k_idx, i] = True

        time_per_slot = seg_start[None, :, :] + samples / N
        time_per_slot = time_per_slot * 0.999 + 0.001
        time_per_slot = torch.where(is_real[None, :, :], time_per_slot, torch.ones_like(time_per_slot))
        time_per_slot = time_per_slot.reshape(batch_size, -1)
        time_per_token = time_per_slot.repeat_interleave(C, dim=1)
        return time_per_slot, time_per_token


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
                H_cfg = N*C.
            action_is_pad: ``[B, N * H_cfg]`` boolean.
        """
        bsize = state.shape[0]
        N, C = self.N, self.C
        H_cfg = N * C
        device = state.device

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        time_per_slot_all, time_per_token = self._build_per_chunk_time(bsize, device)

        time_expanded = time_per_token.unsqueeze(-1)
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state,
        )

        num_slots_per_config = N
        x_t_flat = x_t.view(bsize * N, H_cfg, -1)
        time_per_slot_flat = time_per_slot_all.view(bsize * N, num_slots_per_config)
        if action_is_pad is not None:
            real_mask_flat = (~action_is_pad).view(bsize * N, H_cfg)
        else:
            real_mask_flat = None

        embs_flat, pad_flat, att_flat, cond_flat = self.embed_suffix(
            x_t_flat, time_per_slot_flat, padding_mask=real_mask_flat,
        )
        suffix_embs = embs_flat.view(bsize, N, H_cfg, -1).reshape(bsize, N * H_cfg, -1)
        suffix_pad_per_offset = pad_flat.view(bsize, N, H_cfg)
        suffix_att_masks = att_flat[:bsize]
        if cond_flat is not None:
            suffix_adarms_cond = cond_flat.view(bsize, N, H_cfg, -1).reshape(bsize, N * H_cfg, -1)
        else:
            suffix_adarms_cond = None

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
        attention_mask = (additive_mask == 0).squeeze(1)

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
            adarms_cond=suffix_adarms_cond,
        )

        suffix_out = suffix_out.view(bsize, N, H_cfg, -1)
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.reshape(bsize, N * H_cfg, -1)

        losses = F.mse_loss(u_t, v_t, reduction="none")

        if action_is_pad is not None:
            loss_mask = ~action_is_pad
        else:
            loss_mask = torch.ones(bsize, N * H_cfg, dtype=torch.bool, device=device)

        losses = losses[loss_mask]
        return losses


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

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
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
        return self.action_out_proj(suffix_out)

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
        buf_slots = N
        buf_len = buf_slots * C
        bsz = buffer.shape[0]
        device = buffer.device

        prefix_pad_masks, past_key_values = self._encode_and_prefill(
            images, img_masks, lang_tokens, lang_masks, state,
        )

        num_real = step_counter + 1
        slot_idx = torch.arange(buf_slots, device=device)
        slot_idx_buf = torch.arange(buf_len, device=device) // C

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
        buf_slots = N
        buf_len = buf_slots * C
        bsz = buffer.shape[0]
        device = buffer.device

        prefix_pad_masks, past_key_values = self._encode_and_prefill(
            images, img_masks, lang_tokens, lang_masks, state,
        )

        time_slot = torch.arange(1, N + 1, device=device, dtype=torch.float32) / N
        time_per_slot = time_slot.unsqueeze(0).expand(bsz, -1)

        v_t = self.denoise_step(
            prefix_pad_masks, past_key_values, buffer, time_per_slot, padding_mask=None,
        )

        dt = 1.0 / N
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
        buf_slots = N
        buf_len = buf_slots * C
        bsz = state.shape[0]
        device = state.device

        if buffer is None:
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




class SmolVLAFlashVLAPolicy(PreTrainedPolicy):
    """SmolVLA FlashVLA policy wrapper."""

    config_class = SmolVLAFlashVLAConfig
    name = "smolvla-flashvla"

    def __init__(self, config: SmolVLAFlashVLAConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = VLAFlowMatchingFlashVLA(config)

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
        self.action_buffer[:, :C, :].normal_()
        self._step_counter = 0
        self._cold_start_action = None

    def get_optim_params(self) -> dict:
        return self.parameters()


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
            img = img * 2.0 - 1.0
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
                new_norm = new_norm.to(device=old.weight.device, dtype=old.weight.dtype)
                setattr(parent, attr_name, new_norm)
                n_swapped += 1
            logger.info(
                f"SmolVLA adaRMS: swapped {n_swapped} LlamaRMSNorm modules in expert "
                f"(cond_dim={D}); dense layers zero-init for DiT identity."
            )

        incompatible = instance.load_state_dict(state_dict, strict=False)
        missing = [k for k in incompatible.missing_keys if "lm_head" not in k]
        unexpected = list(incompatible.unexpected_keys)
        if unexpected:
            logger.warning(f"Unexpected keys when loading smolvla streaming: {unexpected[:5]}...")
        if missing:
            logger.warning(f"Missing keys: {missing[:5]}...")

        instance.to(config.device)
        instance.eval()

        instance._cold_start_stats = load_cold_start_stats(pretrained_name_or_path)

        return instance
