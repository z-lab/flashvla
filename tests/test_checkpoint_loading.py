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
"""Regression tests for tied-weight checkpoint loading.

The openpi-derived base checkpoints (``lerobot/pi05_base`` and friends) were
serialized while ``lm_head.weight`` was tied to the input embedding, so
safetensors dropped the embedding as a duplicate and recorded the tie in
``__metadata__`` -- which ``load_file`` does not read back. FlashVLA builds its
backbone from ``lerobot.policies.pi_gemma``, whose classes *untie* the two, so a
plain ``load_state_dict(strict=False)`` leaves the token embedding at random
initialization while the forward pass reads it every step.

These tests reproduce that shape on the meta device (no weights are allocated and
no checkpoint is downloaded) and pin the two guards in
``flashvla.policies.loading``.

Set ``FLASHVLA_PI05_BASE=/path/to/pi05_base/model.safetensors`` to additionally
run the check against the real checkpoint header.
"""

from __future__ import annotations

import json
import os
import struct

import pytest
import torch

from flashvla.policies.loading import (
    assert_checkpoint_covers_parameters,
    find_unfilled_parameters,
    restore_untied_lm_head_embeddings,
)


def _policy_classes():
    from flashvla.policies.pi0.configuration_pi0 import PI0Config, PI0FlashVLAConfig
    from flashvla.policies.pi0.modeling_pi0 import PI0Policy
    from flashvla.policies.pi0.modeling_pi0_flashvla import PI0FlashVLAPolicy
    from flashvla.policies.pi05.configuration_pi05 import PI05Config, PI05FlashVLAConfig
    from flashvla.policies.pi05.modeling_pi05 import PI05Policy
    from flashvla.policies.pi05.modeling_pi05_flashvla import PI05FlashVLAPolicy

    return [
        ("pi05-flashvla", PI05FlashVLAPolicy, PI05FlashVLAConfig),
        ("pi05", PI05Policy, PI05Config),
        ("pi0-flashvla", PI0FlashVLAPolicy, PI0FlashVLAConfig),
        ("pi0", PI0Policy, PI0Config),
    ]


def _build_on_meta(policy_cls, config_cls):
    config = config_cls()
    config.device = "cpu"
    config.dtype = "float32"
    with torch.device("meta"):
        return policy_cls(config)


def _input_embedding_keys(policy) -> list[str]:
    """State-dict keys of every input embedding whose lm_head is untied."""
    names = {}
    for name, module in policy.named_modules():
        names.setdefault(id(module), name)

    keys = []
    for module in policy.modules():
        head = getattr(module, "lm_head", None)
        get_input_embeddings = getattr(module, "get_input_embeddings", None)
        if head is None or not callable(get_input_embeddings):
            continue
        embedding = get_input_embeddings()
        if embedding is None or embedding.weight is head.weight:
            continue
        keys.append(f"{names[id(embedding)]}.weight")
    return keys


@pytest.mark.parametrize("name,policy_cls,config_cls", _policy_classes())
def test_tied_checkpoint_leaves_no_parameter_unfilled(name, policy_cls, config_cls):
    """A tie-deduplicated checkpoint must still fill every parameter."""
    policy = _build_on_meta(policy_cls, config_cls)
    target_sd = policy.state_dict()

    embedding_keys = _input_embedding_keys(policy)
    assert embedding_keys, (
        f"{name}: expected at least one untied lm_head/embedding pair. If lerobot "
        "started tying them again, this test needs updating (and the restore "
        "helper becomes a no-op, which is safe)."
    )

    # A checkpoint saved with the tie in place: safetensors keeps one name per
    # shared storage and drops the input embedding.
    mapped_sd = {
        key: torch.empty(tuple(value.shape), device="meta")
        for key, value in target_sd.items()
        if key not in embedding_keys
    }

    before = find_unfilled_parameters(policy, mapped_sd)
    assert set(before) == set(embedding_keys), (
        f"{name}: expected exactly the input embeddings to be unfilled, got {before}"
    )

    restored = restore_untied_lm_head_embeddings(policy, mapped_sd, target_sd)
    assert set(restored) == set(embedding_keys)

    assert find_unfilled_parameters(policy, mapped_sd) == []
    assert_checkpoint_covers_parameters(policy, mapped_sd, source="synthetic-tied")


@pytest.mark.parametrize("name,policy_cls,config_cls", _policy_classes())
def test_restore_is_a_no_op_for_untied_checkpoints(name, policy_cls, config_cls):
    """FlashVLA's own exports carry the embedding; resume must not overwrite it."""
    policy = _build_on_meta(policy_cls, config_cls)
    target_sd = policy.state_dict()
    mapped_sd = {
        key: torch.empty(tuple(value.shape), device="meta")
        for key, value in target_sd.items()
    }
    sentinels = {key: mapped_sd[key] for key in _input_embedding_keys(policy)}

    assert restore_untied_lm_head_embeddings(policy, mapped_sd, target_sd) == []
    for key, tensor in sentinels.items():
        assert mapped_sd[key] is tensor, f"{name}: {key} was overwritten on resume"


@pytest.mark.parametrize("name,policy_cls,config_cls", _policy_classes())
def test_coverage_check_rejects_a_truncated_checkpoint(name, policy_cls, config_cls):
    """The guard must fire for an ordinary missing tensor, not just embeddings."""
    policy = _build_on_meta(policy_cls, config_cls)
    target_sd = policy.state_dict()
    dropped = "model.action_out_proj.weight"
    assert dropped in target_sd, f"{name}: {dropped} not in state_dict"

    mapped_sd = {
        key: torch.empty(tuple(value.shape), device="meta")
        for key, value in target_sd.items()
        if key != dropped
    }
    restore_untied_lm_head_embeddings(policy, mapped_sd, target_sd)

    with pytest.raises(RuntimeError, match="random initialization"):
        assert_checkpoint_covers_parameters(policy, mapped_sd, source="truncated")

    # ...unless it is declared as intentionally freshly initialized.
    assert_checkpoint_covers_parameters(
        policy, mapped_sd, source="truncated", allow={dropped}
    )


@pytest.mark.skipif(
    not os.environ.get("FLASHVLA_PI05_BASE"),
    reason="set FLASHVLA_PI05_BASE to the pi05_base model.safetensors to run",
)
def test_real_pi05_base_header_covers_every_parameter():
    """End-to-end against the real checkpoint's key set (header read only)."""
    from flashvla.policies.pi05.configuration_pi05 import PI05FlashVLAConfig
    from flashvla.policies.pi05.modeling_pi05_flashvla import PI05FlashVLAPolicy

    path = os.environ["FLASHVLA_PI05_BASE"]
    with open(path, "rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    header.pop("__metadata__", None)

    policy = _build_on_meta(PI05FlashVLAPolicy, PI05FlashVLAConfig)
    target_sd = policy.state_dict()

    prefix_rules = [
        ("model.action_in_proj.", "model.suffix_embedder.action_in_proj."),
        ("model.action_out_proj.", "model.action_out_proj."),
        ("model.time_mlp_in.", "model.suffix_embedder.time_mlp_in."),
        ("model.time_mlp_out.", "model.suffix_embedder.time_mlp_out."),
        ("model.state_proj.", "model.suffix_embedder.state_proj."),
        ("model.state_mlp_in.", "model.suffix_embedder.state_mlp_in."),
        ("model.state_mlp_out.", "model.suffix_embedder.state_mlp_out."),
        ("model.paligemma_with_expert.gemma_expert.", "model.action_expert."),
        ("model.paligemma_with_expert.paligemma.", "model.vlm."),
        ("action_in_proj.", "model.suffix_embedder.action_in_proj."),
        ("action_out_proj.", "model.action_out_proj."),
        ("time_mlp_in.", "model.suffix_embedder.time_mlp_in."),
        ("time_mlp_out.", "model.suffix_embedder.time_mlp_out."),
        ("state_proj.", "model.suffix_embedder.state_proj."),
        ("state_mlp_in.", "model.suffix_embedder.state_mlp_in."),
        ("state_mlp_out.", "model.suffix_embedder.state_mlp_out."),
        ("paligemma_with_expert.gemma_expert.", "model.action_expert."),
        ("paligemma_with_expert.paligemma.", "model.vlm."),
    ]

    def map_key(key: str) -> str:
        for src, dst in prefix_rules:
            if key.startswith(src):
                return dst + key[len(src) :]
        return key

    mapped_sd = {}
    for old_key, spec in header.items():
        new_key = map_key(old_key)
        shape = tuple(spec["shape"])
        if new_key in target_sd and tuple(target_sd[new_key].shape) != shape:
            continue
        mapped_sd[new_key] = torch.empty(shape, device="meta")

    assert find_unfilled_parameters(policy, mapped_sd), (
        "pi05_base is expected to omit the tied embeddings; if it no longer does, "
        "this test's premise is stale."
    )
    restore_untied_lm_head_embeddings(policy, mapped_sd, target_sd)
    assert_checkpoint_covers_parameters(policy, mapped_sd, source=path)
