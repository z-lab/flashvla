# Copyright 2025 FlashVLA team. All rights reserved.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
import flashvla.configs  # noqa: F401 — registers the "lingbot-flashvla" discriminator
from flashvla.async_manager import AsyncStreamingActionManager
from flashvla.policies.factory import make_pre_post_processors
from flashvla.policies.lingbot.configuration_lingbot import LingBotFlashVLAConfig
from flashvla.policies.lingbot.modeling_lingbot_flashvla import LingbotFlashVLAPolicy


class LingBotFlashVLAModel:
    """Expose a LingBot-VLA FlashVLA checkpoint through the RoboTwin RPC."""

    def __init__(
        self,
        policy_path: str,
        cold_start_mode: str = "current_state",
        device: str = "cuda",
        inference_overlap_steps: int = 0,
        n_action_steps: int | None = None,
        compile_model: bool | None = None,
        compile_mode: str | None = None,
        skip_stale_actions: bool = False,
        tokenizer_path: str | None = None,
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[lingbot_flashvla] CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = torch.device(device)

        cfg = PreTrainedConfig.from_pretrained(policy_path)
        if not isinstance(cfg, LingBotFlashVLAConfig):
            raise TypeError(
                "Expected a lingbot-flashvla checkpoint, got "
                f"{getattr(cfg, 'type', type(cfg).__name__)!r}"
            )
        cfg.device = str(self.device)
        cfg.cold_start_mode = cold_start_mode
        if tokenizer_path is not None:
            cfg.tokenizer_path = tokenizer_path
        if n_action_steps is not None:
            requested = int(n_action_steps)
            if not 1 <= requested <= int(cfg.chunk_size):
                raise ValueError(
                    "LingBot n_action_steps must be in "
                    f"[1, chunk_size={cfg.chunk_size}], got {requested}"
                )
            cfg.n_action_steps = requested
        if compile_model is not None:
            cfg.compile_model = bool(compile_model)
        if compile_mode is not None:
            cfg.compile_mode = str(compile_mode)

        self.policy = LingbotFlashVLAPolicy.from_pretrained(
            policy_path,
            config=cfg,
        )
        self.policy.to(self.device)
        self.policy.eval()
        self._runtime_dtype = next(self.policy.parameters()).dtype

        # FlashVLA checkpoints must carry the processors saved at training time:
        # those files own the raw-14 quantile stats and the exact
        # ``<bos>{instruction}\n`` tokenization contract.
        local_checkpoint = Path(policy_path).expanduser()
        if local_checkpoint.is_dir():
            missing = [
                name
                for name in ("policy_preprocessor.json", "policy_postprocessor.json")
                if not (local_checkpoint / name).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    "LingBot FlashVLA checkpoint is missing saved processor metadata: "
                    + ", ".join(missing)
                )
        else:
            # Unlike a base model used for fine-tuning, a deployable policy
            # must carry its exact normalization/tokenization pipeline. Check
            # this explicitly so a Hub/auth error cannot silently fall back to
            # freshly constructed processors with missing dataset statistics.
            from huggingface_hub import hf_hub_download

            for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
                hf_hub_download(
                    repo_id=policy_path,
                    filename=name,
                )

        preprocessor_overrides = {
            "device_processor": {
                "device": str(self.device),
                # The saved pipeline moves every floating tensor, RGB
                # included, through this step. Keep pixels fp32 until
                # resize/normalization; _cast_floating below converts only
                # the non-image model inputs to the runtime dtype.
                "float_dtype": "float32",
            },
        }
        if tokenizer_path is not None:
            preprocessor_overrides.update(
                self._tokenizer_processor_override(
                    policy_path=policy_path,
                    tokenizer_path=cfg.tokenizer_path,
                )
            )

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=policy_path,
            preprocessor_overrides=preprocessor_overrides,
        )

        self.overlap_steps = int(inference_overlap_steps)
        # RTC realignment: skip the stale prefix of each replanned chunk.
        # Enabled via the skip_stale_actions config/CLI override, or forced
        # server-wide with SKIP_STALE_ACTIONS=1 (mirrors the LIBERO eval).
        self.skip_stale_actions = (
            bool(skip_stale_actions) or os.environ.get("SKIP_STALE_ACTIONS") == "1"
        )
        self.manager = AsyncStreamingActionManager(
            policy=self.policy,
            overlap_steps=self.overlap_steps,
            skip_stale_actions=self.skip_stale_actions,
        )
        self._current_instruction: str | None = None

        print(f"[lingbot_flashvla] loaded {policy_path}")
        print(
            f"[lingbot_flashvla] cold_start_mode={cold_start_mode}, device={self.device}, "
            f"n_action_steps={self.policy.config.n_action_steps}, "
            f"chunk_size={self.policy.config.chunk_size}, "
            f"num_buffer_slots={self.policy.config.num_buffer_slots}, "
            f"inference_overlap_steps={self.overlap_steps}, "
            f"skip_stale_actions={self.skip_stale_actions}, "
            f"compile_model={getattr(self.policy.config, 'compile_model', False)}, "
            f"dtype={self._runtime_dtype}"
        )

        # Compilation is expensive; capture both the cold-start and steady
        # graphs at server boot, before an episode begins.
        if getattr(self.policy.config, "compile_model", False):
            self._eager_warmup()

    # -- dispatcher -----------------------------------------------------

    def call(self, func_name: str, obs: Any = None):
        method = getattr(self, func_name, None)
        if method is None or not callable(method):
            raise AttributeError(f"LingBotFlashVLAModel has no callable {func_name!r}")
        return method() if obs is None else method(obs)

    # -- RoboTwin RPC surface -------------------------------------------

    def reset_model(self, *_args, **_kwargs) -> None:
        """Clear per-episode state; any warmed-up CUDA graph is preserved."""
        self.manager.reset()
        self._current_instruction = None
        print("[lingbot_flashvla] model reset")

    def reset(self, *args, **kwargs) -> None:
        self.reset_model(*args, **kwargs)

    def get_action(self, obs_dict: dict) -> np.ndarray:
        """Run one FlashVLA step and return a 14-dim absolute qpos."""
        instruction = str(obs_dict["instruction"])
        if instruction != self._current_instruction:
            self._current_instruction = instruction
            print(f"[lingbot_flashvla] instruction: {instruction[:80]}")

        batch = self._preprocess(obs_dict)
        with torch.inference_mode():
            action = self.manager.act(batch)
        return self._postprocess(action)

    def needs_image_obs(self) -> bool:
        """Whether the next action call will launch policy inference."""
        return self.manager.needs_observation()

    def get_cached_action(self) -> np.ndarray:
        """Replay one already-computed action without a rendered observation."""
        return self._postprocess(self.manager.pop_cached_action())

    # -- preprocessing / postprocessing ---------------------------------

    @staticmethod
    def _tokenizer_processor_override(
        *,
        policy_path: str,
        tokenizer_path: str,
    ) -> dict[str, dict[str, Any]]:
        """Override the tokenizer repository in a saved LeRobot processor."""
        checkpoint = Path(policy_path).expanduser()
        if checkpoint.is_dir():
            config_path = checkpoint / "policy_preprocessor.json"
        else:
            from huggingface_hub import hf_hub_download

            config_path = Path(
                hf_hub_download(
                    repo_id=policy_path,
                    filename="policy_preprocessor.json",
                )
            )

        with config_path.open() as stream:
            registry_names = {
                step.get("registry_name")
                for step in json.load(stream).get("steps", [])
            }

        if "tokenizer_processor" in registry_names:
            return {"tokenizer_processor": {"tokenizer_name": tokenizer_path}}
        raise KeyError(
            "Saved LingBot preprocessor has no supported tokenizer step; "
            f"registry_names={sorted(name for name in registry_names if name)}"
        )

    def _preprocess(self, obs_dict: dict) -> dict[str, Any]:
        return self._cast_floating(self.preprocessor(self._encode_batch(obs_dict)))

    def _encode_batch(self, obs_dict: dict) -> dict[str, Any]:
        def to_chw_float(image: np.ndarray) -> torch.Tensor:
            tensor = torch.from_numpy(np.ascontiguousarray(image)).float().div_(255.0)
            return tensor.permute(2, 0, 1).unsqueeze(0)

        state = np.asarray(obs_dict["state"], dtype=np.float32).reshape(-1)
        if state.size != 14:
            raise ValueError(f"RoboTwin LingBot state must be raw-14, got {state.shape}")

        return {
            "observation.images.cam_high": to_chw_float(obs_dict["head_rgb"]),
            "observation.images.cam_left_wrist": to_chw_float(obs_dict["left_rgb"]),
            "observation.images.cam_right_wrist": to_chw_float(obs_dict["right_rgb"]),
            "observation.state": torch.from_numpy(state).unsqueeze(0),
            "task": [self._current_instruction or ""],
        }

    def _cast_floating(self, value: Any, feature_key: str | None = None) -> Any:
        """Move to device, keeping RGB in fp32 and other floats at runtime dtype."""
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                dtype = (
                    torch.float32
                    if feature_key is not None and feature_key.startswith("observation.images.")
                    else self._runtime_dtype
                )
                return value.to(device=self.device, dtype=dtype)
            return value.to(device=self.device)
        if isinstance(value, dict):
            return {key: self._cast_floating(item, feature_key=key) for key, item in value.items()}
        if isinstance(value, list):
            return [self._cast_floating(item, feature_key=feature_key) for item in value]
        if isinstance(value, tuple):
            return tuple(self._cast_floating(item, feature_key=feature_key) for item in value)
        return value

    def _postprocess(self, action: torch.Tensor) -> np.ndarray:
        action = self.postprocessor(action)
        raw = action.detach().to("cpu").numpy().reshape(-1).astype(np.float32)
        if raw.size != 14:
            raise ValueError(f"LingBot FlashVLA must return raw-14 actions, got {raw.shape}")
        return raw

    def _eager_warmup(self) -> None:
        """Compile / capture the cold-start and steady graphs before episode 1."""
        import time

        image_resolution = getattr(self.policy.config, "image_resolution", (224, 224))
        height, width = (
            (image_resolution, image_resolution)
            if isinstance(image_resolution, int)
            else image_resolution
        )
        dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
        dummy = {
            "instruction": "warmup placeholder instruction for compile",
            "head_rgb": dummy_image,
            "left_rgb": dummy_image,
            "right_rgb": dummy_image,
            "state": np.zeros(14, dtype=np.float32),
        }

        try:
            self._current_instruction = dummy["instruction"]
            batch = self._preprocess(dummy)
            started = time.perf_counter()
            print(
                "[lingbot_flashvla] eager warmup: compiling/capturing the cold-start "
                f"and steady streaming paths (compile_mode="
                f"{getattr(self.policy.config, 'compile_mode', 'default')})"
            )
            self.manager.warmup(batch)
            print(f"[lingbot_flashvla] eager warmup done in {time.perf_counter() - started:.1f}s")
        finally:
            self._current_instruction = None
            self.manager.reset()
