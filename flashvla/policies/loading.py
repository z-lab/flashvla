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
"""Checkpoint-loading helpers shared by the PI0 / PI0.5 policies.

These policies build their VLM and action expert from ``lerobot.policies.pi_gemma``.
``PiGemmaForCausalLM`` and ``PaliGemmaForConditionalGenerationWithPiGemma`` call
``super().__init__(config)`` -- which ties ``lm_head.weight`` to the input
embedding -- and only then replace ``self.model``. The replacement orphans
``lm_head``, so the head and the embedding are two *independent* tensors here
even though upstream transformers keeps them tied.

That matters when loading the openpi-derived base checkpoints. ``lerobot/pi05_base``
was serialized with the tie in place, so ``safetensors`` dropped the duplicate and
recorded the relationship in ``__metadata__`` only::

    embed_tokens keys: []
    lm_head keys: ['paligemma_with_expert.paligemma.lm_head.weight', ...]
    __metadata__: {"...language_model.embed_tokens.weight": "...paligemma.lm_head.weight"}

``safetensors.torch.load_file`` does not read ``__metadata__`` back, so a plain
``load_state_dict(..., strict=False)`` fills ``lm_head`` and silently leaves the
526M-parameter token embedding at its random initialization -- while the embedding
*is* on the forward hot path via the prefix embedder. Upstream lerobot patches this
in ``lerobot/policies/pi05/modeling_pi05.py`` by cloning the head into the embedding;
:func:`restore_untied_lm_head_embeddings` does the same thing generically.

:func:`assert_checkpoint_covers_parameters` is the backstop: it turns any remaining
never-written parameter into a hard error instead of a silent accuracy loss.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable

from torch import Tensor, nn


def _module_name_by_id(root: nn.Module) -> dict[int, str]:
    """Map module identity to its first registered path under ``root``."""
    names: dict[int, str] = {}
    for name, module in root.named_modules():
        names.setdefault(id(module), name)
    return names


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

    The ``embedding_key not in mapped_sd`` guard makes this a no-op for
    checkpoints that already carry the embedding explicitly -- in particular
    FlashVLA's own exports, so resume is unaffected.

    Args:
        instance: The freshly constructed policy, before ``load_state_dict``.
        mapped_sd: Checkpoint tensors already renamed to this policy's key space.
        target_sd: ``instance.state_dict()``, used for shape validation.

    Returns:
        The embedding keys that were filled in, for logging.
    """
    module_names = _module_name_by_id(instance)
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

        if head_key not in mapped_sd or embedding_key in mapped_sd:
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
    named = list(instance.named_parameters())
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


__all__ = [
    "assert_checkpoint_covers_parameters",
    "find_unfilled_parameters",
    "restore_untied_lm_head_embeddings",
]
