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
from collections.abc import Collection, Iterable

from torch import Tensor, nn


def _module_name_by_id(root: nn.Module) -> dict[int, str]:
    """Map module identity to its first registered path under ``root``."""
    names: dict[int, str] = {}
    for name, module in root.named_modules():
        names.setdefault(id(module), name)
    return names


def _parameter_aliases(model: nn.Module) -> dict[int, list[str]]:
    """Map ``id(parameter)`` to every state_dict name it is registered under, in
    registration order."""
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    return aliases


def restore_untied_lm_head_embeddings(
    instance: nn.Module,
    mapped_sd: dict[str, Tensor],
    target_sd: dict[str, Tensor],
) -> list[str]:
    """Re-materialize input embeddings that a tied-weight checkpoint omitted.

    Mutates ``mapped_sd`` in place. For every submodule that owns both an
    ``lm_head`` and an input embedding, and whose two weights are *not* the same
    tensor object, copies the head weight into the embedding entry when the
    checkpoint supplied the head but not the embedding.

    The guard checks every name the embedding is registered under (e.g. the
    prefix embedder's ``lang_embedder`` view, which safetensors' storage dedup
    may have kept instead of ``embed_tokens``), so a checkpoint that already
    carries the embedding under any alias is left alone.

    Args:
        instance: The freshly constructed policy, before ``load_state_dict``.
        mapped_sd: Checkpoint tensors already renamed to this policy's key space.
        target_sd: ``instance.state_dict()``, used for shape validation.

    Returns:
        The embedding keys that were filled in, for logging.
    """
    module_names = _module_name_by_id(instance)
    aliases = _parameter_aliases(instance)
    restored: list[str] = []

    for name, module in instance.named_modules():
        head = getattr(module, "lm_head", None)
        if not isinstance(head, nn.Module):
            continue
        get_input_embeddings = getattr(module, "get_input_embeddings", None)
        if not callable(get_input_embeddings):
            continue
        try:
            embedding = get_input_embeddings()
        except (AttributeError, NotImplementedError):
            continue
        if not isinstance(embedding, nn.Module):
            continue

        head_weight = getattr(head, "weight", None)
        embedding_weight = getattr(embedding, "weight", None)
        if head_weight is None or embedding_weight is None:
            continue
        if head_weight is embedding_weight:
            # Still tied: load_state_dict reaches both through the one tensor.
            continue

        head_name = module_names.get(id(head))
        embedding_name = module_names.get(id(embedding))
        if head_name is None or embedding_name is None:
            continue
        head_key = f"{head_name}.weight"
        embedding_key = f"{embedding_name}.weight"

        embedding_present = any(
            alias in mapped_sd for alias in aliases.get(id(embedding_weight), [embedding_key])
        )
        if head_key not in mapped_sd or embedding_present:
            continue
        if embedding_key not in target_sd:
            continue
        if tuple(target_sd[embedding_key].shape) != tuple(mapped_sd[head_key].shape):
            continue

        mapped_sd[embedding_key] = mapped_sd[head_key].clone()
        restored.append(embedding_key)

    return restored


def find_unfilled_parameters(
    instance: nn.Module,
    mapped_sd: Collection[str],
    *,
    allow: Iterable[str] = (),
) -> list[str]:
    """Return one name per parameter tensor the checkpoint never wrote.

    Deduplicates by tensor identity. These policies register the same
    ``nn.Parameter`` under several names -- the FlashVLA joint layers alias the
    backbone decoder layers, and PaliGemma exposes the language model twice --
    so a name-based ``missing_keys`` check reports hundreds of false positives
    for tensors that were in fact filled under a different name.

    Args:
        instance: The policy after ``load_state_dict``.
        mapped_sd: The key set that was passed to ``load_state_dict``.
        allow: Parameter names that are deliberately left at their fresh
            initialization (for example adaRMS projections that a non-adaRMS
            base checkpoint cannot supply).

    Returns:
        Sorted parameter names, one per distinct unfilled tensor.
    """
    allow = set(allow)
    # remove_duplicate=False so every alias name of an aliased parameter is
    # visible: a tensor is "filled" if ANY of its names is in the checkpoint.
    # The FlashVLA joint layers alias the backbone decoder layers, and a
    # checkpoint may store a given tensor under EITHER alias (e.g. VLM layers
    # saved under the fused `model.layers.*` name while the deduped first name
    # is the un-fused `model.vlm.*` one). With the default remove_duplicate=True
    # the deduped name can miss the checkpoint's chosen alias and this check
    # falsely reports a loaded tensor as unfilled.
    named = list(instance.named_parameters(remove_duplicate=False))
    filled_ids = {id(parameter) for name, parameter in named if name in mapped_sd}
    allowed_ids = {id(parameter) for name, parameter in named if name in allow}

    unfilled: dict[int, str] = {}
    for name, parameter in named:
        identity = id(parameter)
        if identity in filled_ids or identity in allowed_ids:
            continue
        unfilled.setdefault(identity, name)
    return sorted(unfilled.values())


def assert_checkpoint_covers_parameters(
    instance: nn.Module,
    mapped_sd: Collection[str],
    *,
    source: str,
    allow: Iterable[str] = (),
) -> None:
    """Fail loudly when a checkpoint leaves parameters at random initialization.

    ``load_state_dict(strict=False)`` reports such tensors in ``missing_keys``,
    which these policies cannot check directly because of weight aliasing. This
    identity-based equivalent is safe to enforce.

    Raises:
        RuntimeError: If any parameter tensor was never written.
    """
    unfilled = find_unfilled_parameters(instance, mapped_sd, allow=allow)
    if not unfilled:
        return

    parameters = dict(instance.named_parameters())
    total = sum(parameters[name].numel() for name in unfilled)
    detail = ", ".join(
        f"{name} {tuple(parameters[name].shape)}" for name in unfilled[:10]
    )
    if len(unfilled) > 10:
        detail += f", ... (+{len(unfilled) - 10} more)"
    raise RuntimeError(
        f"Checkpoint {source!r} left {len(unfilled)} parameter tensors "
        f"({total:,} parameters) at random initialization: {detail}. "
        "Every trainable tensor must come from the checkpoint; pass its name via "
        "`allow=` only if it is intentionally freshly initialized."
    )


def collapse_shared_parameter_aliases(
    model: nn.Module, state: dict[str, Tensor]
) -> list[str]:
    """Keep one checkpoint entry per shared parameter and return the dropped names.

    A parameter registered under several names (tied lm_head/embed_tokens, the
    prefix embedder's view of the VLM embedding, joint layers aliasing backbone
    layers) is one tensor in the model. A non-FSDP save writes it once, but an
    FSDP2 full-state export gathers every name separately, so all aliases reach
    the file, and a dormant alias may hold a stale value. Loading several names
    onto one parameter would let module order pick the winner; keeping a single
    name makes exactly one ``copy_`` decide. Preference: the first-registered
    non-``lm_head`` name (the embedding the FSDP plan wraps and trains), with a
    dormant ``lm_head`` used only when it is the sole name the file carries.
    """
    dropped: list[str] = []
    for names in _parameter_aliases(model).values():
        present = [name for name in names if name in state]
        present.sort(key=lambda name: name.endswith("lm_head.weight"))
        for name in present[1:]:
            del state[name]
            dropped.append(name)
    return dropped


# Prefix rewrites turning a raw lerobot/openpi pi0.5 checkpoint (paligemma_with_expert.*
# namespace, bare openpi suffix-net keys) into this repo's model.* namespace. Shared by
# the eager (PI05Policy) and streaming (PI05FlashVLAPolicy) loaders.
PI05_RENAME_RULES: list[tuple[str, str]] = [
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


def load_remapped_checkpoint(
    model: nn.Module,
    model_file: str,
    rename_rules: Iterable[tuple[str, str]] = (),
    *,
    allow_fresh: Iterable[str] = (),
    expected_unexpected: Iterable[str] = (),
    device: str = "cpu",
    source: str | None = None,
) -> None:
    """Fill ``model`` from a single-file safetensors checkpoint.

    One path serves every layout. Keys are prefix-remapped with ``rename_rules``
    (first match wins, identity fall-through), so a raw openpi base lands in
    this policy's ``model.*`` namespace and a native export passes through
    unchanged. Shape-mismatched targets are dropped, tied embeddings the file
    omitted are restored from their lm_head, and alias names of one shared
    parameter are collapsed to a single entry. The load is then non-strict,
    unexpected keys (minus ``expected_unexpected``) are fatal, and every
    parameter must be covered by identity (``allow_fresh`` names may stay at
    init). Callers must invoke this while the model's alias tree and tied
    embeddings are still LIVE, i.e. before any weight fusion / alias detach.
    """
    from safetensors.torch import load_file

    rules = list(rename_rules)

    def map_key(key: str) -> str:
        for src, dst in rules:
            if key.startswith(src):
                return dst + key[len(src):]
        return key

    original_sd = load_file(model_file, device=device)
    target_sd = model.state_dict()
    mapped_sd: dict[str, Tensor] = {}
    shape_mismatched: list[str] = []
    for old_key, value in original_sd.items():
        new_key = map_key(old_key)
        if new_key in target_sd and target_sd[new_key].shape != value.shape:
            shape_mismatched.append(
                f"{new_key} file{tuple(value.shape)} vs model{tuple(target_sd[new_key].shape)}"
            )
            continue
        mapped_sd[new_key] = value
    if shape_mismatched:
        logging.warning(
            "Dropped %d shape-mismatched tensor(s); the coverage check below reports "
            "them if nothing else fills those parameters: %s",
            len(shape_mismatched),
            ", ".join(shape_mismatched),
        )

    restored = restore_untied_lm_head_embeddings(model, mapped_sd, target_sd)
    if restored:
        logging.info(
            "Restored %d tied input embedding(s) from their lm_head: %s",
            len(restored),
            ", ".join(restored),
        )

    dropped = collapse_shared_parameter_aliases(model, mapped_sd)
    if dropped:
        logging.info(
            "Dropped %d alias tensor(s) of shared parameters: %s",
            len(dropped),
            ", ".join(dropped),
        )

    incompatible = model.load_state_dict(mapped_sd, strict=False)
    tolerated = set(expected_unexpected)
    fatal = [k for k in incompatible.unexpected_keys if k not in tolerated]
    if fatal:
        raise RuntimeError("Checkpoint loading failed.\n" f"Unexpected keys: {fatal}")

    assert_checkpoint_covers_parameters(
        model, mapped_sd, source=source or str(model_file), allow=allow_fresh
    )


__all__ = [
    "PI05_RENAME_RULES",
    "load_remapped_checkpoint",
    "assert_checkpoint_covers_parameters",
    "find_unfilled_parameters",
    "restore_untied_lm_head_embeddings",
    "collapse_shared_parameter_aliases",
]
