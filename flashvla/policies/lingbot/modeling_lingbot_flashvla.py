# Portions derived from LingBot-VLA, Copyright Robbyant Team.
# Source: https://github.com/Robbyant/lingbot-vla (commit 4eb34b7).
# Modified by the FlashVLA team. Licensed under Apache-2.0.

import logging
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

from flashvla.policies.lingbot.configuration_lingbot import (
    LingBotFlashVLAConfig,
)
from flashvla.policies.lingbot.modeling_lingbot_vla import (
    FlowMatching,
    LingbotVlaPolicy,
    lingbot_flow_matching_loss,
    lingbot_layout_to_robotwin_raw,
    lingbot_valid_action_mask,
    robotwin_raw_to_lingbot_layout,
)
from flashvla.policies.lingbot.lingbot_utils import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    resize_with_pad,
    sample_beta,
)
from flashvla.policies.pi05.utils import (
    ColdStartStats,
    compute_cold_start_action,
    load_cold_start_stats,
)


# torch.compile / dynamo cannot trace cuda.Event ops, so wrap them in
# dynamo-disabled functions: the compiled cold/steady paths still work under
# profile=True. Mirrors the pi05 FlashVLA policy so the latency benchmark
# reports an identical encode/prefill/action breakdown.
@torch._dynamo.disable
def _ev_record(ev):
    ev.record()


@torch._dynamo.disable
def _ev_elapsed_ms(start, end):
    return start.elapsed_time(end)


@torch._dynamo.disable
def _cuda_sync():
    torch.cuda.synchronize()


def _sinusoidal_pos_embedding_nd(time: Tensor, dimension: int,
                                  min_period: float = 4e-3, max_period: float = 4.0) -> Tensor:
    """Sinusoidal positional embedding for arbitrary-shape time tensors."""
    orig_shape = time.shape
    flat = time.reshape(-1)
    emb = create_sinusoidal_pos_embedding(flat, dimension, min_period, max_period, device=time.device)
    return emb.reshape(*orig_shape, dimension)


class LingbotFlashVLAPolicy(PreTrainedPolicy):
    """LingBot-VLA with FlashVLA action streaming training and inference."""

    config_class = LingBotFlashVLAConfig
    name = "lingbot-flashvla"
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

    def __init__(self, config: LingBotFlashVLAConfig, **kwargs):
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

        self.N = config.num_buffer_slots
        self.C = config.chunk_size

        # Pre-allocated streaming buffer [1, N*C, max_action_dim] (batch=1 eval).
        self.register_buffer(
            "action_buffer",
            torch.zeros(1, config.total_buffer_length, config.max_action_dim),
            persistent=False,
        )
        # Keep a stable-address scalar for the compiled cold-start graph.  Its
        # value changes at each bootstrap observation, while all tensor shapes
        # remain fixed, so one graph covers every one of the N-1 cold steps.
        self.register_buffer(
            "_cold_step_t",
            torch.zeros((), dtype=torch.int64),
            persistent=False,
        )
        # Cold-start normalization stats (filled by from_pretrained); empty default
        # so locally-constructed policies still have a valid attribute.
        self._cold_start_stats = ColdStartStats()

        if config.compile_model:
            # Compile both fixed-shape paths.  The persistent scalar above
            # prevents one cold-start graph specialization per Python step.
            torch.set_float32_matmul_precision("high")
            self._steady_streaming = torch.compile(
                self._steady_streaming,
                mode=config.compile_mode,
            )
            self._cold_start = torch.compile(
                self._cold_start,
                mode=config.compile_mode,
            )

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

    def reset(self):
        """Reset buffer + step counter (call on env reset).

        Buffer layout after reset: [noise, PAD, PAD, ...] — slot 0 is noise,
        the rest zero. Cold start fills one real slot per step.
        """
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        C = self.C
        self.action_buffer.zero_()
        self.action_buffer[:, :C, :].normal_()
        self._step_counter = 0
        self._cold_start_action = None

    def get_optim_params(self):
        return self.parameters()

    # ── image / state / action preparation (same as baseline) ─────────

    @staticmethod
    def _patchify_image(img: Tensor, patch_size: int = 14, temporal_patch_size: int = 2,
                        merge_size: int = 2) -> Tensor:
        return LingbotVlaPolicy._patchify_image(img, patch_size, temporal_patch_size, merge_size)

    def prepare_images(self, batch):
        return LingbotVlaPolicy.prepare_images(self, batch)

    def prepare_state(self, batch):
        state = batch[OBS_STATE]
        if getattr(self.config, "robotwin_feature_layout", False):
            return robotwin_raw_to_lingbot_layout(state, self.config)
        if state.shape[-1] < self.config.max_state_dim:
            state = F.pad(state, (0, self.config.max_state_dim - state.shape[-1]))
        return state

    def prepare_action(self, batch):
        actions = batch[ACTION]
        if getattr(self.config, "robotwin_feature_layout", False):
            return robotwin_raw_to_lingbot_layout(actions, self.config)
        if actions.shape[-1] < self.config.max_action_dim:
            actions = F.pad(actions, (0, self.config.max_action_dim - actions.shape[-1]))
        return actions

    # ── expert vision (optional, suffix tokens) ───────────────────────

    def prepare_expert_images(self, batch) -> Tensor | None:
        """Extract expert-camera images for the DINOv3 expert-vision encoder.

        Returns [B, n_expert, 3, H, W] (resized + CLIP-normalized, matching
        prepare_images) or None when expert vision is disabled / no keys
        present. Preprocessing intentionally mirrors the VLM image pipeline for
        consistency; if the expert encoder is a pretrained DINOv3 expecting
        ImageNet stats, adjust the normalization here.
        """
        if not getattr(self.config, "enable_expert_vision", False):
            return None
        keys = list(getattr(self.config, "expert_image_keys", []) or [])
        present = [k for k in keys if k in batch]
        if not present:
            return None

        _CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        _CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        imgs = []
        for key in present:
            img = batch[key]  # [B, 3, H, W] in [0, 1]
            # Antialiased bilinear interpolation is unavailable for bf16 in
            # the supported PyTorch stack.  Match the VLM path: preprocess in
            # fp32, then cast inside the expert vision encoder.
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
            mean = _CLIP_MEAN.to(device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
            std = _CLIP_STD.to(device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
            img = (img - mean) / std
            imgs.append(img)
        return torch.stack(imgs, dim=1)  # [B, n_expert, 3, H, W]

    def _encode_expert_images(self, expert_imgs: Tensor) -> Tensor:
        """Encode [B, n, 3, H, W] expert images to [B, n, expert_hidden] tokens.

        Mirrors FlowMatching.embed_suffix's expert branch: per-image DINOv3 CLS
        token -> expert_visual_mlp. Computed once on the shared observation.
        """
        qwe = self.model.qwenvl_with_expert
        B, n = expert_imgs.shape[:2]
        flat = expert_imgs.reshape(B * n, *expert_imgs.shape[2:])
        vis_dtype = next(qwe.expert_visual.parameters()).dtype
        feat = qwe.expert_visual.forward_features(flat.to(dtype=vis_dtype))["x_norm_clstoken"]  # [B*n, embed_dim]
        mlp_dtype = qwe.expert_visual_mlp[0].weight.dtype
        feat = qwe.expert_visual_mlp(feat.unsqueeze(1).to(dtype=mlp_dtype))  # [B*n, 1, hidden]
        return feat.reshape(B, n, -1)  # [B, n, hidden]

    # ── time sampling ─────────────────────────────────────────────────

    def _sample_time_for_config(self, bsize: int, num_real_slots: int, device: torch.device) -> Tensor:
        """Sample monotonically increasing per-slot times for a config with k real slots.

        Returns:
            time_per_token: [bsize, N*C] per-token times.
        """
        N, C = self.N, self.C
        k = num_real_slots

        samples = sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            bsize * k,
            device,
        ).reshape(bsize, k)

        slot_indices = torch.arange(k, device=device, dtype=torch.float32)
        seg_starts = (N - k + slot_indices) / N
        seg_widths = 1.0 / N
        time_per_slot = seg_starts[None, :] + samples * seg_widths
        time_per_slot = (
            time_per_slot * self.config.time_sampling_scale
            + self.config.time_sampling_offset
        )

        full_time = torch.ones(bsize, N, device=device, dtype=torch.float32)
        full_time[:, :k] = time_per_slot

        return full_time.unsqueeze(2).expand(-1, -1, C).reshape(bsize, N * C)

    def _sample_shared_training_times(
        self, bsize: int, device: torch.device
    ) -> Tensor:
        """Sample one global N-level ladder shared by all cold-start configs."""
        N, C = self.N, self.C
        samples = sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            bsize * N,
            device,
        ).reshape(bsize, N)
        levels = torch.arange(N, device=device, dtype=torch.float32)
        global_time = levels.unsqueeze(0) / N + samples / N
        global_time = (
            global_time * self.config.time_sampling_scale
            + self.config.time_sampling_offset
        )

        time_all = torch.ones(bsize, N, N * C, device=device, dtype=torch.float32)
        for k_idx in range(N):
            k = k_idx + 1
            slot_times = global_time[:, N - k :]
            time_all[:, k_idx, : k * C] = slot_times.repeat_interleave(C, dim=1)
        return time_all

    # ── embed suffix with per-token time ──────────────────────────────

    def _embed_suffix_per_token_time(self, state: Tensor, noisy_actions: Tensor,
                                      time_per_token: Tensor,
                                      expert_img_emb: Tensor | None = None,
                                      ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Like FlowMatching.embed_suffix but with per-token time [B, seq_len].

        Suffix layout: [expert(n_exp) | state(1) | actions(seq_len)], where the
        expert block is present only when expert_img_emb is given. Non-action
        tokens (expert imgs + state) reuse the first action token's time for
        ada conditioning, matching the baseline's single global time.

        Args:
            expert_img_emb: [B, n_exp, expert_D] precomputed expert-vision tokens
                (encoded once on the shared observation), or None.

        Returns:
            ada_cond: [B, L_suf, D] per-token conditioning for adaRMS.
            embs: [B, L_suf, expert_D] suffix embeddings.
            pad_masks: [B, L_suf].
            att_masks: [B, L_suf].
            (L_suf = n_exp + 1 + seq_len)
        """
        fm = self.model
        bsize = state.shape[0]
        device = state.device
        seq_len = noisy_actions.shape[1]
        n_exp = 0 if expert_img_emb is None else expert_img_emb.shape[1]

        proj_dtype = fm.state_proj.weight.dtype
        state = state.to(dtype=proj_dtype)
        noisy_actions = noisy_actions.to(dtype=proj_dtype)

        state_emb = fm.state_proj(state)  # [B, D]

        time_emb = _sinusoidal_pos_embedding_nd(time_per_token, fm.config.proj_width)
        time_emb = time_emb.to(dtype=proj_dtype)

        action_emb = fm.action_in_proj(noisy_actions)

        if getattr(fm.config, "separate_time_proj", False):
            time_proj = F.silu(fm.time_mlp_in(time_emb))
            ada_tok = F.silu(fm.time_mlp_out(time_proj))  # per-token [B, seq_len, D]
            action_time_emb = action_emb
        else:
            action_time_emb = torch.cat([action_emb, time_emb], dim=-1)
            action_time_emb = F.silu(fm.action_time_mlp_in(action_time_emb))
            action_time_emb = fm.action_time_mlp_out(action_time_emb)
            ada_tok = time_emb  # per-token [B, seq_len, D]

        # ada_cond: non-action tokens (expert imgs + state) use first action's time.
        lead = ada_tok[:, :1, :]
        if n_exp > 0:
            ada_cond = torch.cat([lead.expand(-1, n_exp, -1), lead, ada_tok], dim=1)
            embs = torch.cat([expert_img_emb.to(dtype=proj_dtype), state_emb[:, None], action_time_emb], dim=1)
        else:
            ada_cond = torch.cat([lead, ada_tok], dim=1)
            embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)

        total_len = n_exp + 1 + seq_len
        pad_masks = torch.ones(bsize, total_len, device=device, dtype=torch.bool)
        att_masks = torch.zeros(bsize, total_len, device=device, dtype=torch.bool)
        if n_exp > 0:
            # expert-block start, state, and first action token are causal boundaries
            att_masks[:, [0, n_exp, n_exp + 1]] = True
        else:
            att_masks[:, :2] = True  # state and first action token are causal boundary

        # Every C-token action slot has a different flow time/noise level and
        # must be its own block. Without these boundaries, a cleaner early
        # slot can attend bidirectionally to later, noisier slots, which does
        # not match PI05 streaming training or inference.
        action_offset = n_exp + 1
        att_masks[:, action_offset:: self.C] = True

        return ada_cond, embs, pad_masks, att_masks

    # ── shared observation attention mask ──────────────────────────────

    def _build_shared_obs_mask(
        self,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        suffix_pad_masks_per_config: Tensor,
        suffix_att_masks: Tensor,
        N: int,
    ) -> tuple[Tensor, Tensor]:
        """Build attention mask for shared-observation training.

        Layout: [vlm_prefix | config_0_suffix | config_1_suffix | ...]
        VLM prefix is shared. Cross-config attention is blocked in suffix.
        """
        B = prefix_pad_masks.shape[0]
        L_pre = prefix_pad_masks.shape[1]
        L_suf = suffix_pad_masks_per_config.shape[2]
        total_len = L_pre + N * L_suf
        device = prefix_pad_masks.device

        full_pad = torch.zeros(B, total_len, dtype=torch.bool, device=device)
        full_att = torch.zeros(B, total_len, dtype=torch.bool, device=device)

        full_pad[:, :L_pre] = prefix_pad_masks
        full_att[:, :L_pre] = prefix_att_masks

        full_pad[:, L_pre:] = suffix_pad_masks_per_config.reshape(B, -1)
        full_att[:, L_pre:] = suffix_att_masks.unsqueeze(1).expand(-1, N, -1).reshape(B, -1)

        att_2d = make_att_2d_masks(full_pad, full_att)

        # Block cross-config attention in suffix
        suffix_pos = torch.arange(N * L_suf, device=device)
        config_ids = suffix_pos // L_suf
        same_config = config_ids.unsqueeze(1) == config_ids.unsqueeze(0)
        att_2d[:, L_pre:, L_pre:] = att_2d[:, L_pre:, L_pre:] & same_config

        # Position IDs
        position_ids = torch.zeros(B, total_len, dtype=torch.long, device=device)
        prefix_pos = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
        position_ids[:, :L_pre] = prefix_pos
        last_pre_pos = prefix_pos[:, -1:]

        suffix_pos_per_config = torch.cumsum(suffix_pad_masks_per_config.long(), dim=2)
        position_ids[:, L_pre:] = last_pre_pos + suffix_pos_per_config.reshape(B, -1)

        return att_2d, position_ids

    # ── training forward ──────────────────────────────────────────────

    def forward(self, batch: dict[str, Tensor], noise: Tensor | None = None,
                time: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        """Training forward pass. Delegates to forward_shared_observation."""
        return self.forward_shared_observation(batch, noise=noise, time=time)

    def forward_shared_observation(self, batch: dict[str, Tensor], noise: Tensor | None = None,
                                   time: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        """Training forward with shared observation across N buffer configs.

        Each config k has its own suffix: [state(1), noisy N*C].  H_cfg = N*C.
        """
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        action_is_pad = batch.get("action_is_pad")

        N, C = self.N, self.C
        B = state.shape[0]
        device = state.device
        dtype = state.dtype
        H_cfg = N * C

        # Expert vision (shared across configs): encode once, prepend per config.
        expert_imgs = self.prepare_expert_images(batch)
        expert_img_emb = self._encode_expert_images(expert_imgs) if expert_imgs is not None else None
        n_exp = 0 if expert_img_emb is None else expert_img_emb.shape[1]
        offset = n_exp + 1  # non-action tokens before actions: expert imgs + state

        if noise is None:
            noise = torch.randn(actions.shape, device=device, dtype=dtype)

        # ── Per-config time sampling ──
        actions_per_cfg = actions.reshape(B, N, H_cfg, -1)
        noise_per_cfg = noise.reshape(B, N, H_cfg, -1)

        if time is not None:
            if time.shape == (B, N * H_cfg):
                time_all = time.reshape(B, N, H_cfg).to(device=device, dtype=torch.float32)
            elif time.shape == (B, N, H_cfg):
                time_all = time.to(device=device, dtype=torch.float32)
            else:
                raise ValueError(
                    "Explicit streaming time must have shape "
                    f"{(B, N * H_cfg)} or {(B, N, H_cfg)}, got {tuple(time.shape)}"
                )
        elif self.config.timestep_sample_mode == "per-sample":
            time_all = self._sample_shared_training_times(B, device)
        else:
            time_all = torch.zeros(B, N, H_cfg, device=device, dtype=torch.float32)
            for k_idx in range(N):
                k = k_idx + 1
                time_all[:, k_idx] = self._sample_time_for_config(B, k, device)

        # Compute noisy actions and target velocity
        time_flat = time_all.reshape(B, N * H_cfg)
        time_exp = time_flat.unsqueeze(-1)
        actions_flat = actions_per_cfg.reshape(B, N * H_cfg, -1)
        noise_flat = noise_per_cfg.reshape(B, N * H_cfg, -1)
        x_t = time_exp * noise_flat + (1 - time_exp) * actions_flat
        u_t = noise_flat - actions_flat

        # ── Embed VLM prefix once (shared) ──
        prefix_embs, prefix_pad, prefix_att = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, vlm_causal=False,
        )

        # ── Embed suffix per-config: [B*N, 1+H_cfg, D] ──
        x_t_per_cfg = x_t.reshape(B * N, H_cfg, -1)
        time_per_cfg = time_all.reshape(B * N, H_cfg)
        state_expanded = state.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)

        # Replicate the shared expert tokens across the N configs.
        if expert_img_emb is not None:
            expert_emb_cfg = expert_img_emb.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, n_exp, -1)
        else:
            expert_emb_cfg = None

        ada_cond, suffix_embs, suffix_pad_single, suffix_att_single = \
            self._embed_suffix_per_token_time(state_expanded, x_t_per_cfg, time_per_cfg,
                                              expert_img_emb=expert_emb_cfg)

        L_suf = suffix_embs.shape[1]  # n_exp + 1 + H_cfg
        suffix_embs = suffix_embs.reshape(B, N, L_suf, -1)
        ada_cond = ada_cond.reshape(B, N, L_suf, -1)
        suffix_pad = suffix_pad_single.reshape(B, N, L_suf)

        # Apply action_is_pad to suffix pad masks (skip expert + state tokens)
        if action_is_pad is not None:
            action_pad_per_cfg = action_is_pad.reshape(B, N, H_cfg)
            suffix_pad[:, :, offset:] = suffix_pad[:, :, offset:] & ~action_pad_per_cfg

        # Flatten: [B, N*L_suf, D]
        suffix_embs_flat = suffix_embs.reshape(B, N * L_suf, -1)
        ada_cond_flat = ada_cond.reshape(B, N * L_suf, -1)

        suffix_att_rep = suffix_att_single[:B]  # [B, L_suf]

        # ── Build shared-obs attention mask ──
        att_2d, position_ids = self._build_shared_obs_mask(
            prefix_pad, prefix_att, suffix_pad, suffix_att_rep, N,
        )

        vlm_position_ids = torch.cumsum(prefix_pad.long(), dim=1) - 1

        # ── Joint forward ──
        (_, suffix_out), _ = self.model.qwenvl_with_expert.forward(
            attention_mask=att_2d,
            position_ids=position_ids,
            vlm_position_ids=vlm_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs_flat],
            use_cache=False,
            fill_kv_cache=False,
            ada_cond=(
                ada_cond_flat
                if getattr(self.config, "adanorm_time", False)
                else None
            ),
        )

        # ── Extract action outputs (skip expert + state tokens of each config) ──
        suffix_out = suffix_out.reshape(B, N, L_suf, -1)
        action_out = suffix_out[:, :, offset:, :].reshape(B, N * H_cfg, -1)

        if action_out.dtype != self.model.action_out_proj.weight.dtype:
            action_out = action_out.to(self.model.action_out_proj.weight.dtype)
        v_t = self.model.action_out_proj(action_out)

        # ── Compute loss ──
        losses = lingbot_flow_matching_loss(u_t, v_t, self.config.loss_type)
        valid_dims = lingbot_valid_action_mask(self.config, losses.device)
        losses = losses[:, :, valid_dims]

        # Mask: exclude padded positions
        if action_is_pad is not None:
            loss_mask = ~action_is_pad
        else:
            loss_mask = torch.ones(B, N * H_cfg, dtype=torch.bool, device=device)

        losses = losses[loss_mask]

        loss = losses.mean()
        return loss, {"loss": loss.item()}

    # ── inference ─────────────────────────────────────────────────────

    def _sample_noise(self, bsz: int, device: torch.device, dtype=torch.float32) -> Tensor:
        buf_len = self.N * self.C
        return torch.randn(bsz, buf_len, self.config.max_action_dim, device=device, dtype=dtype)

    @torch.no_grad()
    def _encode_prefix(self, images, img_masks, lang_tokens, lang_masks,
                       ev_after_encode=None, ev_after_prefill=None):
        """Embed the VLM prefix once and fill its KV cache (per observation).

        ``ev_after_encode`` / ``ev_after_prefill`` are optional cuda.Events the
        profiler uses to split this into the encode (image/language embed) and
        prefill (KV-cache fill) segments; both None on the normal path.
        """
        fm = self.model
        prefix_embs, prefix_pad_masks, prefix_att_masks = fm.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, vlm_causal=False,
        )
        if ev_after_encode is not None:
            _ev_record(ev_after_encode)
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = fm.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=fm.config.use_cache,
            fill_kv_cache=True,
        )
        if ev_after_prefill is not None:
            _ev_record(ev_after_prefill)
        return prefix_pad_masks, past_key_values

    @torch.no_grad()
    def _denoise_step(self, prefix_pad_masks, past_key_values, state, x_t,
                      time_per_token, expert_img_emb=None, padding_mask=None) -> Tensor:
        """Predict velocity over the whole buffer using the cached prefix KV.

        x_t: [B, buf_len, action_dim]; time_per_token: [B, buf_len];
        padding_mask: [B, buf_len] bool (True = real action token) or None.
        Returns v_t [B, buf_len, action_dim].
        """
        fm = self.model
        n_exp = 0 if expert_img_emb is None else expert_img_emb.shape[1]
        offset = n_exp + 1

        ada_cond, suffix_embs, suffix_pad_masks, suffix_att_masks = \
            self._embed_suffix_per_token_time(state, x_t, time_per_token, expert_img_emb=expert_img_emb)

        # Mask padded (non-real) action tokens out of attention (cold start).
        if padding_mask is not None:
            suffix_pad_masks = suffix_pad_masks.clone()
            suffix_pad_masks[:, offset:] = suffix_pad_masks[:, offset:] & padding_mask

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = fm.qwenvl_with_expert.forward(
            attention_mask=full_att_2d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=fm.config.use_cache,
            fill_kv_cache=False,
            ada_cond=ada_cond if getattr(fm.config, "adanorm_time", False) else None,
        )
        suffix_out = outputs_embeds[1][:, offset:]  # action tokens (trailing N*C block)
        if suffix_out.dtype != fm.action_out_proj.weight.dtype:
            suffix_out = suffix_out.to(fm.action_out_proj.weight.dtype)
        return fm.action_out_proj(suffix_out)

    @torch.no_grad()
    def _cold_start(self, images, img_masks, lang_tokens, lang_masks, state,
                    expert_img_emb, buffer, step_counter, profile=False) -> Tensor:
        """One denoise step during cold start (buffer filling); emits no chunk.

        Real slot s (0..num_real-1) sits at time (N-num_real+1+s)/N — the upper
        edge of its training band; padded slots are at 1.0 and masked out. After
        the Euler step, real slots are kept, the next slot gets fresh noise, the
        rest stay zero — mirroring the training cold-start ladder.

        Returns ``(buffer, breakdown)``, where ``breakdown`` is the
        encode/prefill/action cuda.Event split under ``profile`` and None
        otherwise. Cold-start steps emit no chunk but still cost a full
        encode + prefill + denoise.
        """
        N, C = self.N, self.C
        buf_len = N * C
        bsz = state.shape[0]
        device = state.device

        _evs = (
            [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            if profile
            else None
        )

        # encode (embed) and prefill (KV fill) are recorded inside _encode_prefix.
        if profile:
            _ev_record(_evs[0])
        prefix_pad_masks, past_key_values = self._encode_prefix(
            images, img_masks, lang_tokens, lang_masks,
            ev_after_encode=_evs[1] if profile else None,
            ev_after_prefill=_evs[2] if profile else None,
        )

        num_real = step_counter + 1
        slot_idx = torch.arange(N, device=device)
        slot_idx_buf = torch.arange(buf_len, device=device) // C

        is_real = slot_idx < num_real
        time_slot = (N - num_real + 1 + slot_idx).to(torch.float32) / N
        time_slot = torch.where(is_real, time_slot, torch.ones_like(time_slot))

        padding_mask = is_real.unsqueeze(0).unsqueeze(2).expand(bsz, -1, C).reshape(bsz, buf_len)
        time = time_slot.unsqueeze(0).unsqueeze(2).expand(bsz, -1, C).reshape(bsz, buf_len)

        v_t = self._denoise_step(
            prefix_pad_masks, past_key_values, state, buffer, time,
            expert_img_emb=expert_img_emb, padding_mask=padding_mask,
        ).to(dtype=buffer.dtype)

        dt = 1.0 / N
        updated_buffer = buffer - dt * v_t
        new_noise = self._sample_noise(bsz, device, dtype=buffer.dtype)
        keep_mask = (slot_idx_buf < num_real).view(1, buf_len, 1)
        noise_mask = (slot_idx_buf == num_real).view(1, buf_len, 1)
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
    def _steady_streaming(self, images, img_masks, lang_tokens, lang_masks, state,
                          expert_img_emb, buffer, profile=False):
        """One denoise step during steady streaming (buffer full); emits first chunk.

        Slot s sits at time (s+1)/N; one Euler step takes slot 0 to t=0 (fully
        denoised -> executed), then the buffer shifts left by one chunk and a
        fresh-noise slot is appended at the tail.

        Returns ``(actions_to_execute, buffer, breakdown)``, where ``breakdown``
        is the encode/prefill/action cuda.Event split under ``profile`` and None
        otherwise. This single-step action segment is the FlashVLA win over the
        baseline's num_steps denoise.
        """
        N, C = self.N, self.C
        buf_len = N * C
        bsz = state.shape[0]
        device = state.device

        _evs = (
            [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            if profile
            else None
        )

        if profile:
            _ev_record(_evs[0])
        prefix_pad_masks, past_key_values = self._encode_prefix(
            images, img_masks, lang_tokens, lang_masks,
            ev_after_encode=_evs[1] if profile else None,
            ev_after_prefill=_evs[2] if profile else None,
        )

        time_per_slot = torch.arange(1, N + 1, device=device, dtype=torch.float32) / N
        time = time_per_slot.unsqueeze(0).unsqueeze(2).expand(bsz, -1, C).reshape(bsz, buf_len)

        v_t = self._denoise_step(
            prefix_pad_masks, past_key_values, state, buffer, time,
            expert_img_emb=expert_img_emb, padding_mask=None,
        ).to(dtype=buffer.dtype)

        dt = 1.0 / N
        buffer = buffer - dt * v_t
        actions_to_execute = buffer[:, :C, :self.config.max_action_dim]
        new_buffer = self._sample_noise(bsz, device, dtype=buffer.dtype)
        new_buffer[:, :-C, :] = buffer[:, C:, :]
        buffer = new_buffer
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
        return actions_to_execute, buffer, breakdown

    @torch.no_grad()
    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state,
                       expert_img_emb, buffer, step_counter, profile=False):
        """Top-level dispatch: cold start (no chunk) or steady streaming (one chunk).

        Returns ``(actions_to_execute_or_None, buffer, step_counter + 1,
        profile_results)``, where ``profile_results`` is the
        encode/prefill/action cuda.Event breakdown under ``profile``.
        """
        N = self.N
        bsz = state.shape[0]
        device = state.device
        if buffer is None:
            buffer = self._sample_noise(bsz, device)

        if step_counter < N - 1:
            self._cold_step_t.fill_(step_counter)
            buffer, breakdown = self._cold_start(
                images, img_masks, lang_tokens, lang_masks, state,
                expert_img_emb, buffer, self._cold_step_t, profile=profile,
            )
            actions_to_execute = None
        else:
            actions_to_execute, buffer, breakdown = self._steady_streaming(
                images, img_masks, lang_tokens, lang_masks, state, expert_img_emb, buffer,
                profile=profile,
            )
        return actions_to_execute, buffer, step_counter + 1, breakdown

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], profile: bool = False) -> Tensor | None:
        """Advance the streaming buffer one step. Returns [B, C, action_dim] or None (cold start).

        With ``profile=True`` this returns ``(chunk_or_None, profile_results)``
        — surfaced on cold-start steps too, which emit no chunk but still run
        encode + prefill + the cold-start denoise. The layout conversion and
        ``.float().clone()`` stay outside the timed region (not model compute),
        matching pi05. The default preserves the plain None/tensor return used
        by select_action, the async manager, and the RoboTwin adapter.
        """
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        if state.shape[0] != 1:
            raise ValueError(
                "LingBot FlashVLA keeps one persistent denoising buffer "
                f"and currently supports batch_size=1 at inference, got {state.shape[0]}"
            )
        # Vision shape metadata must own stable eager storage before the
        # compiled cold/steady paths capture it as an input.  Creating and
        # caching these tensors from inside a CUDAGraph leaves them backed by
        # an output slot that is overwritten by the next replay.
        self.model.qwenvl_with_expert.prepare_vision_metadata(images)
        if self.config.compile_model and images.device.type == "cuda":
            # The policy owns two compiled callables (cold and steady), so the
            # automatic iteration heuristic cannot reliably infer their common
            # inference boundary.
            torch.compiler.cudagraph_mark_step_begin()
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        expert_imgs = self.prepare_expert_images(batch)
        expert_img_emb = self._encode_expert_images(expert_imgs) if expert_imgs is not None else None

        actions, buffer, new_step, profile_results = self.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            expert_img_emb, self.action_buffer, self._step_counter,
            profile=profile,
        )
        self.action_buffer.copy_(buffer)
        self._step_counter = new_step

        if actions is None:
            if profile:
                return None, profile_results
            return None
        actions = lingbot_layout_to_robotwin_raw(actions, self.config)
        # `_steady_streaming` may be backed by a reusable CUDAGraph output
        # pool.  The manager can retain this chunk across environment steps,
        # so give it ordinary storage before the next replay.
        chunk = actions.float().clone()
        if profile:
            return chunk, profile_results
        return chunk

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action with chunking.

        During cold start (buffer not yet full) returns a "hold" action from
        compute_cold_start_action (config.cold_start_mode: "current_state" for
        absolute-qpos envs like RoboTwin, "zero_delta" for delta envs).
        """
        mode = getattr(self.config, "cold_start_mode", "current_state")
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            if actions is None:
                cold_action = compute_cold_start_action(
                    mode=mode,
                    stats=self._cold_start_stats,
                    action_dim=self.config.action_dim,
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
        """Load a FlashVLA checkpoint or initialize from upstream LingBot shards."""
        del resume_download, proxies
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        if not isinstance(config, LingBotFlashVLAConfig):
            raise TypeError(
                "LingbotFlashVLAPolicy requires an explicit "
                "LingBotFlashVLAConfig when initializing from an "
                "upstream baseline checkpoint."
            )

        kwargs.pop("dataset_stats", None)
        instance = cls(config=config, **kwargs)
        from flashvla.policies.lingbot.checkpoint import (
            load_lingbot_weights,
            move_lingbot_for_runtime,
        )

        checkpoint_dir = load_lingbot_weights(
            instance,
            pretrained_name_or_path,
            strict=strict,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            # The default FlashVLA path deliberately replaces the released
            # checkpoint's joint action-time MLP with a fresh time-only MLP.
            # The joint-time ablation instead loads the released projection
            # exactly, so it must not enter the migration path.
            allow_joint_to_separate_time_projection=bool(
                getattr(config, "separate_time_proj", False)
            ),
        )
        move_lingbot_for_runtime(instance)
        instance.eval()
        try:
            # load_lingbot_weights resolves Hugging Face repo ids to their
            # local snapshot.  Processor statistics live beside the resolved
            # weights, not at the literal repo-id string.
            instance._cold_start_stats = load_cold_start_stats(checkpoint_dir)
        except Exception as e:  # noqa: BLE001 — stats are optional; degrade gracefully
            logging.warning(f"LingBot streaming: cold-start stats load failed ({e}); using empty stats.")
            instance._cold_start_stats = ColdStartStats()
        return instance
