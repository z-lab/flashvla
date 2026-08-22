# Copyright 2025 FlashVLA team. All rights reserved.
"""RoboTwin deploy_policy entrypoint for lingbot_flashvla (server/client mode).

This module is imported by BOTH sides of the RoboTwin eval pipeline:

  1. RoboTwin/script/policy_model_server.py (runs in the flashvla env):
       calls ``get_model(usr_args)`` → ``LingBotFlashVLAModel`` instance.

  2. RoboTwin/script/eval_policy_client.py (runs in the RoboTwin conda env):
       loads ``eval(TASK_ENV, model, observation)`` from here. ``model`` is a
       ``ModelClient`` that forwards everything to the server.

The flashvla env and the RoboTwin SAPIEN env have hard conflicts on
torch / numpy / gymnasium versions, so we NEVER import flashvla at module top
level: ``get_model`` does a lazy import, keeping this file importable in the
client env where flashvla is absent.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _coerce_optional_int(value: Any) -> int | None:
    """policy_model_server.py stores non-int strings verbatim, so 'null'/''
    can leak through as strings. Normalize them to None."""
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("", "null", "none"):
            return None
        return int(normalized)
    return int(value)


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("", "null", "none"):
            return None
        if normalized in ("true", "1", "yes"):
            return True
        if normalized in ("false", "0", "no"):
            return False
        raise ValueError(f"can't coerce {value!r} to bool")
    return bool(value)


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if normalized.lower() in ("", "null", "none"):
        return None
    return normalized


def get_model(usr_args: dict) -> Any:
    """Server-side entrypoint: instantiate a LingBotFlashVLAModel."""
    from lingbot_flashvla_model import LingBotFlashVLAModel  # lazy on purpose

    return LingBotFlashVLAModel(
        policy_path=usr_args["policy_path"],
        cold_start_mode=usr_args.get("cold_start_mode", "current_state"),
        device=usr_args.get("device", "cuda"),
        inference_overlap_steps=_coerce_optional_int(
            usr_args.get("inference_overlap_steps", 0)
        ) or 0,
        n_action_steps=_coerce_optional_int(usr_args.get("n_action_steps")),
        compile_model=_coerce_optional_bool(usr_args.get("compile_model")),
        compile_mode=_coerce_optional_str(usr_args.get("compile_mode")),
        skip_stale_actions=_coerce_optional_bool(usr_args.get("skip_stale_actions")) or False,
        tokenizer_path=_coerce_optional_str(usr_args.get("tokenizer_path")),
    )


def _pack_request(TASK_ENV, observation: dict) -> dict:
    """Flatten a RoboTwin observation dict into a JSON-friendly payload.

    Nested RoboTwin obs (with pickled sapien objects) doesn't round-trip
    cleanly through the numpy-aware JSON codec used by the transport, so we
    pull out exactly the fields the policy needs.
    """
    cam = observation["observation"]
    return {
        "instruction": TASK_ENV.get_instruction(),
        "head_rgb": np.ascontiguousarray(cam["head_camera"]["rgb"]),
        "left_rgb": np.ascontiguousarray(cam["left_camera"]["rgb"]),
        "right_rgb": np.ascontiguousarray(cam["right_camera"]["rgb"]),
        "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
    }


def encode_obs(observation):
    return observation


def eval(TASK_ENV, model, observation):
    """Single outer eval step — encode obs, ship to server, execute action.

    ``observation is None`` means the client skipped rendering because the
    server reported it does not need a fresh observation this step.
    """
    if observation is None:
        action = model.call(func_name="get_cached_action")
    else:
        action = model.call(func_name="get_action", obs=_pack_request(TASK_ENV, observation))
    TASK_ENV.take_action(np.asarray(action, dtype=np.float32).reshape(-1))


def reset_model(model):
    """Local-mode reset. eval_policy_client.py calls the RPC directly instead."""
    if hasattr(model, "call"):
        model.call(func_name="reset_model")
    elif hasattr(model, "reset_model"):
        model.reset_model()
