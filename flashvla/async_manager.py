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
"""Async chunk-overlap action manager for FlashVLA streaming policies.

This is the single integration point for running FlashVLA policies in a
control loop — simulation (LIBERO, RoboTwin) and real robots alike. It wraps
a streaming (or plain chunked) policy with asynchronous chunk-overlap
execution:

  * Launch the NEXT chunk inference ``overlap_steps`` steps before the
    current chunk ends. Under ``torch.compile`` + CUDA graphs the launch is
    dispatch-only (sub-millisecond CPU cost); the GPU work runs in the
    background of subsequent control steps.
  * At the chunk transition, the ``.cpu().numpy()`` sync waits only on
    whatever GPU work the overlap window has not already hidden.
  * During the streaming cold start (the first ``num_buffer_slots - 1``
    calls return ``None``), a hold-pose action is synthesized via
    :func:`flashvla.policies.pi05.utils.compute_cold_start_action`.

Async-overlap timeline (chunk_size = n_action_steps = 10, overlap_steps = 2)::

  chunk_index:   0   1   2   3   4   5   6   7   8   9   →   0   1
  step:          execute current_chunk[0..7] ──┐
                                               │ launch_next_inference(obs)
                                               │   (non-blocking under compile)
                                               │   GPU runs in background
                              execute [8]  [9]┘                    │
                                                                    │
                                                  chunk_index wraps │
                                                  to 0; transition: │
                                                  next_chunk.cpu()  ┘
                                                  (sync; GPU is done)

Set ``overlap_steps = 0`` to disable async — the manager then launches the
next chunk synchronously at the transition.

Minimal real-robot integration::

    from flashvla.async_manager import AsyncStreamingActionManager

    mgr = AsyncStreamingActionManager(policy, overlap_steps=1)
    mgr.reset()                                  # at episode start
    while running:
        processed = preprocessor(robot_obs)      # tokenize/normalize/device
        action = mgr.act(processed)              # [B, action_dim] on CPU
        action = postprocessor(action)           # unnormalize
        robot.send_action(action.cpu().numpy())
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import torch

from lerobot.policies.pretrained import PreTrainedPolicy

from flashvla.policies.pi05.utils import compute_cold_start_action


class AsyncStreamingActionManager:
    """Wraps a streaming or chunked policy with async chunk-overlap execution.

    Usage:
        mgr = AsyncStreamingActionManager(policy, overlap_steps=2)
        mgr.reset()
        for obs in env_loop:
            processed = preprocessor(obs)
            action = mgr.act(processed)             # [B, action_dim] on CPU
            action = postprocessor({ACTION: action})[ACTION]
            obs, r, done, info = env.step(action.cpu().numpy())
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        overlap_steps: int = 0,
        skip_stale_actions: bool = False,
    ):
        """
        Args:
            policy: Loaded policy. Must expose ``predict_action_chunk`` and
                ``config.n_action_steps`` / ``config.action_feature``.
            overlap_steps: Launch the next inference this many steps before
                the current chunk ends. ``0`` = sync (transition triggers a
                blocking launch). Must satisfy
                ``0 <= overlap_steps <= n_action_steps``.
        """
        self.policy = policy
        self.n_action_steps = policy.config.n_action_steps
        self.action_dim = policy.config.action_feature.shape[0]
        self.overlap_steps = overlap_steps
        self.skip_stale_actions = skip_stale_actions

        if not (0 <= self.overlap_steps <= self.n_action_steps):
            raise ValueError(
                f"overlap_steps must be in [0, {self.n_action_steps}], got {self.overlap_steps}"
            )
        if self.skip_stale_actions:
            # Realignment consumes [overlap_steps, overlap_steps + n_action_steps)
            # of each promoted chunk, so the policy's full chunk must cover it.
            chunk_size = getattr(policy.config, "chunk_size", None)
            if chunk_size is not None and self.overlap_steps + self.n_action_steps > chunk_size:
                raise ValueError(
                    "skip_stale_actions requires overlap_steps + n_action_steps <= chunk_size; "
                    f"got {self.overlap_steps} + {self.n_action_steps} > {chunk_size}"
                )

        self.device = next(self.policy.parameters()).device

        self._is_streaming = policy.config.type in (
            "lingbot-flashvla",
            "pi05-flashvla",
            "pi0-flashvla",
            "smolvla-flashvla",
        )
        self.cold_start_stats = getattr(policy, "_cold_start_stats", None)

        self.current_chunk: Optional[np.ndarray] = None
        self.next_chunk: Optional[torch.Tensor] = None
        self.chunk_index = 0
        # First consumed index of current_chunk: 0 normally, overlap_steps for a
        # chunk promoted under skip_stale_actions (RTC realignment).
        self._skip_offset = 0

    def reset(self) -> None:
        """Clear manager state AND reset the underlying policy's streaming
        buffer. Call at episode boundaries."""
        self.current_chunk = None
        self.next_chunk = None
        self.chunk_index = 0
        self._skip_offset = 0
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def is_running(self) -> bool:
        return self.current_chunk is not None or self.next_chunk is not None

    def _should_launch_next(self) -> bool:
        """True when this call should start the next inference asynchronously."""
        return (
            self.overlap_steps > 0
            and self.chunk_index
            == self._skip_offset + self.n_action_steps - self.overlap_steps
        )

    def needs_observation(self) -> bool:
        """Whether the next control step must call the policy.

        Simulation adapters can use this to skip rendering and transferring
        images while replaying actions that are already cached on CPU.
        """
        if not self.is_running():
            return True
        if self.chunk_index == 0 and self.next_chunk is None:
            return True
        return self._should_launch_next()

    def _pop_current_action(self) -> torch.Tensor:
        if self.current_chunk is None:
            raise RuntimeError("No current action chunk is available")
        action_np = self.current_chunk[:, self.chunk_index, :]
        action = torch.from_numpy(action_np)
        self.chunk_index += 1
        if self.chunk_index - self._skip_offset >= self.n_action_steps:
            self.chunk_index = 0
            self._skip_offset = 0
            self.current_chunk = None
        return action

    def _promote_next_chunk(self) -> None:
        """Make ``next_chunk`` current. Under skip_stale_actions, realign by
        resuming at index ``overlap_steps``: the chunk was launched that many
        steps early, so its first ``overlap_steps`` actions target control
        steps the outgoing chunk already executed. The consumption window
        [overlap_steps, overlap_steps + n_action_steps) draws on the policy's
        full chunk, which predict_action_chunk returns unsliced."""
        self.current_chunk = self.next_chunk.detach().cpu().numpy()
        self.next_chunk = None
        if self.skip_stale_actions and self.overlap_steps > 0:
            if self.current_chunk.shape[1] < self.overlap_steps + self.n_action_steps:
                raise ValueError(
                    "skip_stale_actions needs overlap_steps + n_action_steps <= "
                    f"chunk length; got chunk length {self.current_chunk.shape[1]} "
                    f"< {self.overlap_steps} + {self.n_action_steps}"
                )
            self.chunk_index = self.overlap_steps
            self._skip_offset = self.overlap_steps

    def pop_cached_action(self) -> torch.Tensor:
        """Return one cached CPU action without running policy inference."""
        if self.needs_observation():
            raise RuntimeError(
                "pop_cached_action called when the next step requires an observation"
            )
        if self.current_chunk is None:
            if self.next_chunk is None:
                raise RuntimeError("No current or next action chunk is available")
            self._promote_next_chunk()
        return self._pop_current_action()

    def _launch_inference(self, processed_obs: dict) -> torch.Tensor:
        """Run policy + handle cold-start fallback.

        Returns a GPU tensor ``[B, n_action_steps, action_dim]``. NOT synced
        to CPU here — caller decides when to ``.cpu().numpy()``.

        Under torch.compile + CUDA graphs, this call is non-blocking: the
        compiled ``_steady_streaming`` is dispatched to the GPU and the
        Python returns immediately with a tensor reference. The GPU work
        completes in the background; subsequent ``.cpu().numpy()`` syncs.
        """
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(processed_obs)

        if chunk is not None:
            return chunk

        if not self._is_streaming:
            raise RuntimeError(
                "predict_action_chunk returned None for non-streaming policy "
                f"(type={self.policy.config.type!r}); did the policy load correctly?"
            )

        bsz = processed_obs["observation.state"].shape[0]
        cold = compute_cold_start_action(
            mode=self.policy.config.cold_start_mode,
            stats=self.cold_start_stats,
            action_dim=self.action_dim,
            device=self.device,
            state_normalized=processed_obs.get("observation.state"),
        )
        if cold.shape[0] == 1 and bsz > 1:
            cold = cold.expand(bsz, -1)
        if cold.shape[0] != bsz:
            raise ValueError(
                f"cold-start chunk batch {cold.shape[0]} != env batch {bsz}"
            )
        # Pad the hold-pose chunk so a skip_stale_actions promotion can still
        # consume n_action_steps actions after skipping overlap_steps.
        cold_len = self.n_action_steps + (self.overlap_steps if self.skip_stale_actions else 0)
        chunk = cold.unsqueeze(1).expand(bsz, cold_len, self.action_dim).contiguous()
        return chunk

    def act(self, processed_obs: dict) -> torch.Tensor:
        """Return one action per env per call.

        Args:
            processed_obs: Already-preprocessed observation dict, on the
                policy's device. Same shape/keys policy.predict_action_chunk
                expects.

        Returns:
            CPU action tensor ``[B, action_dim]``. Caller is responsible for
            postprocessing (unnormalize).
        """
        if not self.is_running():
            self.current_chunk = self._launch_inference(processed_obs).detach().cpu().numpy()
        elif self.chunk_index == 0:
            if self.next_chunk is not None:
                self._promote_next_chunk()
            elif self.overlap_steps == 0:
                self.current_chunk = self._launch_inference(processed_obs).detach().cpu().numpy()
            else:
                raise RuntimeError(
                    "Async overlap is enabled but next_chunk is None at "
                    "transition. Did _launch_inference fail silently?"
                )

        if self._should_launch_next():
            self.next_chunk = self._launch_inference(processed_obs)

        # Keep execution on CPU: moving the tiny action back to the policy GPU
        # would synchronize it behind an in-flight inference on the default
        # stream and defeat overlap.
        return self._pop_current_action()

    def warmup(self, processed_obs: dict, num_steps: Optional[int] = None) -> None:
        """Pre-capture the CUDA graph for ``_steady_streaming``.

        Without warmup, the FIRST chunk transition into compiled
        ``_steady_streaming`` (which happens around step
        ``(num_buffer_slots - 1) * n_action_steps`` of the rollout)
        triggers torch.compile + CUDA graph capture, blocking the rollout
        for 10-30+ seconds and skewing episode latency stats.

        This method runs ``num_steps`` inferences on the given obs,
        forcing the compile+capture to complete, then **resets** all
        streaming state (the manager's own + the policy's) so the buffer
        is back to its pristine initial condition. The actual rollout
        then walks through cold-start cleanly with the cached graph
        replaying in ~10-30 ms instead of seconds.

        Args:
            processed_obs: Any valid preprocessed observation (often
                ``preprocessor(env.reset()[0])``). The contents don't
                affect the captured graph beyond tensor shape and dtype.
            num_steps: Number of inferences. Default
                ``num_buffer_slots + 1`` for streaming (covers cold-start
                + 2 steady) or 3 for non-streaming.
        """
        if num_steps is None:
            n_buf = getattr(self.policy.config, "num_buffer_slots", None)
            num_steps = (n_buf + 1) if n_buf is not None else 3

        logging.info(
            f"[AsyncStreamingActionManager] warmup: {num_steps} inferences "
            "(captures CUDA graph; state is reset afterwards)..."
        )
        t0 = time.perf_counter()
        for _ in range(num_steps):
            with torch.inference_mode():
                _ = self.policy.predict_action_chunk(processed_obs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        self.reset()

        logging.info(
            f"[AsyncStreamingActionManager] warmup done in {elapsed:.2f}s "
            f"(avg {1000 * elapsed / num_steps:.0f}ms/step); state reset."
        )
