#!/usr/bin/env python
# Copyright 2025 FlashVLA team. All rights reserved.
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

from __future__ import annotations

import logging

from lerobot.envs.libero import LiberoEnv

logger = logging.getLogger(__name__)

# Set by the rollout each step: True -> the upcoming env.step's observation is
# not consumed -> wrapped camera sensors reuse the last frame instead of
# calling sim.render. Physics, controller, and success checks are untouched, so
# with the same seed the trajectory is identical to rendering every step.
_STATE = {"skip": False}
_ENABLED = False
_orig_step = LiberoEnv.step  # captured at import, before any patching


def set_skip_render(skip: bool) -> None:
    _STATE["skip"] = bool(skip)


def _wrap_camera_sensors(env: LiberoEnv) -> None:
    """Wrap each camera observable's sensor to reuse the last frame on skip steps."""
    robo = getattr(env._env, "env", None)
    obs_map = getattr(robo, "_observables", None)
    if not isinstance(obs_map, dict):
        return
    wrapped_any = False
    for name in env.camera_name:
        ob = obs_map.get(name)
        if ob is None or getattr(ob, "_flashvla_sensor_wrapped", False):
            continue
        orig_sensor = getattr(ob, "_sensor", None)
        if not callable(orig_sensor):
            continue
        cache = {"img": None}

        def wrapped(obs_cache, _orig=orig_sensor, _cache=cache):
            if _STATE["skip"] and _cache["img"] is not None:
                return _cache["img"]
            img = _orig(obs_cache)
            _cache["img"] = img
            return img

        # robosuite's Observable reads sensor.__modality__ (set by the @sensor
        # decorator); preserve it so the observable machinery keeps working.
        if hasattr(orig_sensor, "__modality__"):
            wrapped.__modality__ = orig_sensor.__modality__
        wrapped.__name__ = getattr(orig_sensor, "__name__", "wrapped_sensor")
        ob._sensor = wrapped
        ob._flashvla_sensor_wrapped = True
        wrapped_any = True
    if wrapped_any:
        logger.info(
            "[libero_render_skip] wrapped camera sensors for task=%s "
            "(render skipped on open-loop steps)",
            getattr(env, "task", "?"),
        )


def _patched_step(self, action):
    if not getattr(self, "_flashvla_sensors_ready", False):
        _wrap_camera_sensors(self)
        self._flashvla_sensors_ready = True
    return _orig_step(self, action)


def enable_patch() -> None:
    """Idempotently install the lazy sensor-wrapping hook on LiberoEnv.step."""
    global _ENABLED
    if _ENABLED:
        return
    LiberoEnv.step = _patched_step
    _ENABLED = True
    logger.info("[libero_render_skip] enabled (sensors wrapped lazily on first step)")
