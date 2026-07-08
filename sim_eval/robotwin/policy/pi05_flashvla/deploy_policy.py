# Copyright 2025 FlashVLA team. All rights reserved.
"""RoboTwin deploy_policy entrypoint for pi05_flashvla (server/client mode).

This module is imported by BOTH sides of the RoboTwin eval pipeline:

  1. RoboTwin/script/policy_model_server.py (runs in the flashvla env):
       calls ``get_model(usr_args)`` → ``PI05FlashVLAModel`` instance.
       The server exposes every callable method on that instance over a
       TCP socket as ``{cmd: <method_name>, obs: <payload>}`` requests.

  2. RoboTwin/script/eval_policy_client.py (runs in the RoboTwin conda env):
       loads ``eval(TASK_ENV, model, observation)`` from here. ``model`` is a
       ``ModelClient`` that forwards everything to the server via
       ``model.call(func_name=..., obs=...)``.

The flashvla training env and the RoboTwin SAPIEN env have hard conflicts on
torch / numpy / gymnasium versions — see README.md for the full matrix.
Because of this we NEVER import flashvla at module top level: ``get_model``
does a lazy import so the client env (which doesn't have flashvla) can still
import this file without errors.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _coerce_optional_int(v):
    """policy_model_server.py's CLI parser stores non-int strings verbatim,
    so 'null'/'none'/'' all leak through as strings. Normalize them to None."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("", "null", "none"):
            return None
        return int(s)
    return int(v)


def _coerce_optional_bool(v):
    """Same story for bool: yml gives true/false/None, CLI gives 'true'/'false'/'null'."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("", "null", "none"):
            return None
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        raise ValueError(f"can't coerce compile_model={v!r} to bool")
    return bool(v)


def get_model(usr_args: dict) -> Any:
    """Server-side entrypoint: instantiate a PI05FlashVLAModel.

    Lazy-imports ``pi05_flashvla_model`` (and therefore ``flashvla``) so that
    this module is cheap to import on the client side, where flashvla is not
    installed.
    """
    from pi05_flashvla_model import PI05FlashVLAModel  # noqa: WPS433 — lazy on purpose
    return PI05FlashVLAModel(
        policy_path=usr_args["policy_path"],
        cold_start_mode=usr_args.get("cold_start_mode", "current_state"),
        device=usr_args.get("device", "cuda"),
        inference_overlap_steps=_coerce_optional_int(usr_args.get("inference_overlap_steps", 0)) or 0,
        n_action_steps=_coerce_optional_int(usr_args.get("n_action_steps")),
        compile_model=_coerce_optional_bool(usr_args.get("compile_model")),
    )


def _pack_request(TASK_ENV, observation: dict) -> dict:
    """Flatten a RoboTwin observation dict into a JSON-friendly payload.

    Nested RoboTwin obs (with pickled sapien objects) doesn't round-trip
    cleanly through the numpy-aware JSON codec used by the server/client
    transport. We pull out exactly the fields the policy needs.
    """
    cam = observation["observation"]
    return {
        "instruction":    TASK_ENV.get_instruction(),
        "head_rgb":       np.ascontiguousarray(cam["head_camera"]["rgb"]),
        "left_rgb":       np.ascontiguousarray(cam["left_camera"]["rgb"]),
        "right_rgb":      np.ascontiguousarray(cam["right_camera"]["rgb"]),
        "state":          np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
    }


def encode_obs(observation):
    return observation


def eval(TASK_ENV, model, observation):
    """Single outer eval step — encode obs, ship to server, execute action.

    ``model`` is either a ``ModelClient`` (remote mode, common) or a local
    ``PI05FlashVLAModel`` (rare: only works if flashvla + sapien are installable
    in the same env, which is currently blocked by torch/numpy conflicts).
    Both expose ``call(func_name, obs)``.
    """
    if observation is None:
        action = model.call(func_name="get_cached_action")
    else:
        request = _pack_request(TASK_ENV, observation)
        action = model.call(func_name="get_action", obs=request)
    TASK_ENV.take_action(np.asarray(action).reshape(-1))


def reset_model(model):
    """Local-mode reset. eval_policy_client.py bypasses this and calls
    ``model.call(func_name='reset_model')`` directly, so this function is
    only hit from RoboTwin/script/eval_policy.py (local mode)."""
    if hasattr(model, "call"):
        model.call(func_name="reset_model")
    elif hasattr(model, "reset_model"):
        model.reset_model()
