# Copyright 2025 FlashVLA team. All rights reserved.
"""Server-side pi05_flashvla model wrapper for RoboTwin eval.

This module is loaded ONLY in the flashvla conda env (where flashvla is
pip-installed). The RoboTwin client env never imports it — it just holds a
``ModelClient`` that talks to an instance of ``PI05FlashVLAModel`` running here
over a TCP socket.

Async overlap (server-side):
    Wraps the streaming policy in an ``AsyncStreamingActionManager`` so that
    when ``inference_overlap_steps > 0`` and the policy is compiled, the next
    chunk inference is dispatched ``overlap_steps`` calls before the current
    chunk runs out. The TCP client still sees one ``get_action`` call per env
    step — the async hiding all happens inside this process while SAPIEN is
    stepping the simulator.

Protocol exposed to ``RoboTwin/script/policy_model_server.py``:

    PI05FlashVLAModel.call(func_name, obs) -> result
        Dispatcher — the server invokes ``getattr(model, cmd)(obs)`` directly,
        but we also expose ``call`` so local-mode callers get the same API.

    PI05FlashVLAModel.get_action(obs_dict) -> np.ndarray (14,) float32
        obs_dict keys: instruction, head_rgb, left_rgb, right_rgb, state.

    PI05FlashVLAModel.reset_model() -> None
        Called between episodes by the client via
        ``model.call(func_name='reset_model')``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from flashvla.policies.factory import make_pre_post_processors
from flashvla.policies.pi05.configuration_pi05 import PI05FlashVLAConfig  # noqa: F401 — registers the "pi05-flashvla" type with PreTrainedConfig
from flashvla.policies.pi05.modeling_pi05_flashvla import PI05FlashVLAPolicy

from flashvla.async_manager import AsyncStreamingActionManager


class PI05FlashVLAModel:
    def __init__(
        self,
        policy_path: str,
        cold_start_mode: str = "current_state",
        device: str = "cuda",
        inference_overlap_steps: int = 0,
        n_action_steps: int | None = None,
        compile_model: bool | None = None,
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[pi05_flashvla] CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = torch.device(device)

        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.cold_start_mode = cold_start_mode
        cfg.device = str(self.device)
        if n_action_steps is not None:
            cfg.n_action_steps = int(n_action_steps)
        if compile_model is not None:
            cfg.compile_model = bool(compile_model)

        self.policy = PI05FlashVLAPolicy.from_pretrained(policy_path, config=cfg)
        self.policy.to(self.device)
        self.policy.eval()

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        self.overlap_steps = int(inference_overlap_steps)
        # RTC realignment: skip the first overlap_steps stale actions of each
        # replanned chunk (env-gated, mirrors the LIBERO eval).
        self.skip_stale_actions = os.environ.get("SKIP_STALE_ACTIONS") == "1"
        self.manager = AsyncStreamingActionManager(
            policy=self.policy,
            overlap_steps=self.overlap_steps,
            skip_stale_actions=self.skip_stale_actions,
        )

        self._current_instruction: str | None = None

        print(f"[pi05_flashvla] loaded {policy_path}")
        print(
            f"[pi05_flashvla] cold_start_mode={cold_start_mode}, device={self.device}, "
            f"n_action_steps={self.policy.config.n_action_steps}, "
            f"inference_overlap_steps={self.overlap_steps}, "
            f"skip_stale_actions={self.skip_stale_actions}, "
            f"compile_model={getattr(self.policy.config, 'compile_model', False)}"
        )

        self._eager_warmup()

    def _eager_warmup(self) -> None:
        """Pre-run manager.warmup with a fabricated obs.

        The compiled graph is shape-parametrized: image resolution is fixed
        by config.image_resolution; state is padded to max_state_dim; text
        tokens are padded to the tokenizer's max_length by the preprocessor.
        So the dummy warmup graph matches every real episode's graph as
        long as the camera count + image size + instruction tokenization
        behaviour line up.
        """
        try:
            state_feat = self.policy.config.input_features.get("observation.state")
            state_dim = state_feat.shape[0] if state_feat is not None else 14
            image_res = getattr(self.policy.config, "image_resolution", None) or (224, 224)
            H, W = (image_res, image_res) if isinstance(image_res, int) else image_res

            dummy_img = np.zeros((H, W, 3), dtype=np.uint8)
            dummy_obs = {
                "instruction": "warmup placeholder instruction for compile",
                "head_rgb": dummy_img,
                "left_rgb": dummy_img,
                "right_rgb": dummy_img,
                "state": np.zeros(state_dim, dtype=np.float32),
            }
            self._current_instruction = dummy_obs["instruction"]
            batch = self._encode_batch(dummy_obs)
            batch = self.preprocessor(batch)

            import time as _time
            t0 = _time.perf_counter()
            print(f"[pi05_flashvla] eager warmup: state_dim={state_dim}, image=({H},{W}) — "
                  "compile + CUDA-graph capture, may take 30s-10min "
                  f"(compile_mode={getattr(self.policy.config, 'compile_mode', 'default')})")
            self.manager.warmup(batch)
            print(f"[pi05_flashvla] eager warmup done in {_time.perf_counter()-t0:.1f}s — "
                  "first real get_action will be fast")
        finally:
            self._current_instruction = None
            self.manager.reset()


    def call(self, func_name: str, obs: Any = None):
        method = getattr(self, func_name, None)
        if method is None or not callable(method):
            raise AttributeError(f"PI05FlashVLAModel has no callable '{func_name}'")
        if obs is None:
            return method()
        return method(obs)


    def reset_model(self, *_args, **_kwargs) -> None:
        """Clear episode state. Called at the start of every rollout.

        Resets the async manager (which in turn resets the underlying
        policy's streaming buffer + action queue). The warmed-up CUDA
        graph persists across episodes — only the per-episode buffer
        state is wiped.
        """
        self.manager.reset()
        self._current_instruction = None
        print("[pi05_flashvla] model reset")

    def reset(self, *args, **kwargs) -> None:
        self.reset_model(*args, **kwargs)

    def get_action(self, obs_dict: dict) -> np.ndarray:
        """Run one flashvla step and return a 14-dim absolute qpos.

        obs_dict (from deploy_policy._pack_request):
            instruction: str
            head_rgb:   np.uint8 (H, W, 3)
            left_rgb:   np.uint8 (H, W, 3)
            right_rgb:  np.uint8 (H, W, 3)
            state:      np.float32 (14,)
        """
        instruction = obs_dict["instruction"]
        if instruction != self._current_instruction:
            self._current_instruction = instruction
            print(f"[pi05_flashvla] instruction: {instruction[:80]}")

        batch = self._encode_batch(obs_dict)
        batch = self.preprocessor(batch)

        with torch.inference_mode():
            action = self.manager.act(batch)
        action = self.postprocessor(action)

        return action.detach().to("cpu").numpy().reshape(-1).astype(np.float32)

    def needs_image_obs(self) -> bool:
        """Whether the next action call will launch policy inference."""
        m = self.manager
        if not m.is_running():
            return True
        if m.chunk_index == 0 and m.next_chunk is None:
            return True
        return bool(m._should_launch_next())

    def get_cached_action(self) -> np.ndarray:
        """Replay one already-computed action without a rendered observation."""
        if self.needs_image_obs():
            raise RuntimeError("get_cached_action called when the next step needs image observation")

        m = self.manager
        if m.current_chunk is None:
            if m.next_chunk is None:
                raise RuntimeError("no current or next action chunk is available")
            m.current_chunk = m.next_chunk.detach().cpu().numpy()
            m.next_chunk = None

        action_np = m.current_chunk[:, m.chunk_index, :]
        # Keep cached actions on CPU. An H2D copy here would wait behind the
        # in-flight next-chunk inference on the default CUDA stream, defeating
        # the overlap that lets RoboTwin execute cached actions asynchronously.
        action = torch.from_numpy(action_np)

        m.chunk_index = (m.chunk_index + 1) % m.n_action_steps
        if m.chunk_index == 0:
            m.current_chunk = None

        action = self.postprocessor(action)
        return action.detach().to("cpu").numpy().reshape(-1).astype(np.float32)


    def _encode_batch(self, obs_dict: dict) -> dict[str, Any]:
        def _to_chw_float(hwc_uint8: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(np.ascontiguousarray(hwc_uint8)).to(dtype=torch.float32) / 255.0
            return t.permute(2, 0, 1).unsqueeze(0).to(self.device)

        state_np = np.asarray(obs_dict["state"], dtype=np.float32)
        state = torch.from_numpy(state_np).unsqueeze(0).to(self.device)

        return {
            "observation.images.cam_high":        _to_chw_float(obs_dict["head_rgb"]),
            "observation.images.cam_left_wrist":  _to_chw_float(obs_dict["left_rgb"]),
            "observation.images.cam_right_wrist": _to_chw_float(obs_dict["right_rgb"]),
            "observation.state":                  state,
            "task":                               [self._current_instruction or ""],
        }
