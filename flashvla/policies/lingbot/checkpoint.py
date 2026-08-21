# Portions derived from LingBot-VLA, Copyright Robbyant Team.
# Source: https://github.com/Robbyant/lingbot-vla (commit 4eb34b7).
# Modified by the FlashVLA team. Licensed under Apache-2.0.
"""Checkpoint loading helpers shared by LingBot baseline and FlashVLA policies."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from safetensors.torch import load_file

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy


_JOINT_TIME_PROJECTION_KEYS = frozenset(
    {
        "model.action_time_mlp_in.weight",
        "model.action_time_mlp_in.bias",
        "model.action_time_mlp_out.weight",
        "model.action_time_mlp_out.bias",
    }
)
_SEPARATE_TIME_PROJECTION_KEYS = frozenset(
    {
        "model.time_mlp_in.weight",
        "model.time_mlp_in.bias",
        "model.time_mlp_out.weight",
        "model.time_mlp_out.bias",
    }
)


def _resolve_checkpoint_dir(
    pretrained_name_or_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> Path:
    path = Path(pretrained_name_or_path).expanduser()
    if path.is_dir():
        return path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(pretrained_name_or_path),
            allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.yml"],
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
        )
    )


def _model_shards(checkpoint_dir: Path) -> list[Path]:
    single = checkpoint_dir / "model.safetensors"
    if single.is_file():
        return [single]

    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open() as handle:
            index = json.load(handle)
        names = sorted(set(index.get("weight_map", {}).values()))
        shards = [checkpoint_dir / name for name in names]
    else:
        # LingBot's exporter may emit several model-*.safetensors files without
        # a Transformers index. Avoid processor state files from FlashVLA dirs.
        shards = sorted(checkpoint_dir.glob("model*.safetensors"))

    missing_files = [str(path) for path in shards if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"LingBot checkpoint index references missing shards: {missing_files}")
    if not shards:
        raise FileNotFoundError(
            f"No model.safetensors or model*.safetensors shards found in {checkpoint_dir}"
        )
    return shards


def _tensor_shapes(shards: list[Path]) -> dict[str, tuple[int, ...]]:
    """Read safetensor headers without materializing checkpoint tensors."""
    shapes: dict[str, tuple[int, ...]] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in shapes:
                    raise RuntimeError(
                        f"LingBot checkpoint tensor {key!r} appears in multiple shards"
                    )
                shapes[key] = tuple(handle.get_slice(key).get_shape())
    return shapes


def _is_joint_to_separate_time_migration(
    policy: "PreTrainedPolicy",
    target_shapes: dict[str, tuple[int, ...]],
    source_shapes: dict[str, tuple[int, ...]],
    missing: set[str],
    unexpected: set[str],
    shape_mismatches: list[str],
    *,
    allow: bool,
) -> tuple[bool, list[str]]:
    """Recognize only the released-joint -> fresh-FlashVLA time-MLP change."""
    if not allow:
        return False, []

    config = getattr(policy, "config", None)
    if (
        getattr(config, "type", None) != "lingbot-flashvla"
        or not getattr(config, "separate_time_proj", False)
    ):
        raise ValueError(
            "joint-to-separate time projection migration is only valid for a "
            "LingBot FlashVLA policy with separate_time_proj=True"
        )

    if (
        missing != _SEPARATE_TIME_PROJECTION_KEYS
        or unexpected != _JOINT_TIME_PROJECTION_KEYS
        or shape_mismatches
    ):
        return False, []

    width = target_shapes["model.time_mlp_in.weight"][0]
    expected_target = {
        "model.time_mlp_in.weight": (width, width),
        "model.time_mlp_in.bias": (width,),
        "model.time_mlp_out.weight": (width, width),
        "model.time_mlp_out.bias": (width,),
    }
    expected_source = {
        "model.action_time_mlp_in.weight": (width, width * 2),
        "model.action_time_mlp_in.bias": (width,),
        "model.action_time_mlp_out.weight": (width, width),
        "model.action_time_mlp_out.bias": (width,),
    }
    fingerprint_errors = [
        f"target {key}: expected={expected}, actual={target_shapes.get(key)}"
        for key, expected in expected_target.items()
        if target_shapes.get(key) != expected
    ]
    fingerprint_errors.extend(
        f"checkpoint {key}: expected={expected}, actual={source_shapes.get(key)}"
        for key, expected in expected_source.items()
        if source_shapes.get(key) != expected
    )
    return not fingerprint_errors, fingerprint_errors


def load_lingbot_weights(
    policy: "PreTrainedPolicy",
    pretrained_name_or_path: str | Path,
    *,
    strict: bool = True,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: str | bool | None = None,
    allow_joint_to_separate_time_projection: bool = False,
) -> Path:
    """Strictly load one or many shards, with one explicit FlashVLA migration."""
    checkpoint_dir = _resolve_checkpoint_dir(
        pretrained_name_or_path,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
    )
    shards = _model_shards(checkpoint_dir)
    target_state = policy.state_dict()
    target_shapes = {key: tuple(value.shape) for key, value in target_state.items()}
    source_shapes = _tensor_shapes(shards)
    target_keys = set(target_shapes)
    source_keys = set(source_shapes)
    shape_mismatches = sorted(
        f"{key}: checkpoint={source_shapes[key]} model={target_shapes[key]}"
        for key in target_keys & source_keys
        if source_shapes[key] != target_shapes[key]
    )
    compatible_keys = {
        key
        for key in target_keys & source_keys
        if source_shapes[key] == target_shapes[key]
    }
    missing = target_keys - compatible_keys
    unexpected = source_keys - target_keys

    migrated, migration_shape_errors = _is_joint_to_separate_time_migration(
        policy,
        target_shapes,
        source_shapes,
        missing,
        unexpected,
        shape_mismatches,
        allow=allow_joint_to_separate_time_projection,
    )
    policy._lingbot_checkpoint_migration = None
    if migrated:
        policy._lingbot_checkpoint_migration = {
            "type": "joint-to-separate-time-projection",
            "dropped_checkpoint_keys": tuple(sorted(_JOINT_TIME_PROJECTION_KEYS)),
            "fresh_model_keys": tuple(sorted(_SEPARATE_TIME_PROJECTION_KEYS)),
        }
        logging.warning(
            "LingBot FlashVLA architecture migration: dropping the checkpoint's "
            "4 joint action_time_mlp tensors and leaving 4 separate time_mlp "
            "tensors freshly initialized. This path is intended for fresh training, "
            "not direct pretrained inference."
        )
        missing = set()
        unexpected = set()

    all_shape_errors = shape_mismatches + migration_shape_errors
    if strict and (missing or unexpected or all_shape_errors):
        raise RuntimeError(
            "LingBot checkpoint is incompatible with the requested policy.\n"
            f"Missing keys ({len(missing)}): {sorted(missing)[:20]}\n"
            f"Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:20]}\n"
            f"Shape mismatches ({len(all_shape_errors)}): {all_shape_errors[:20]}"
        )

    loaded_keys: set[str] = set()
    for shard in shards:
        state = load_file(str(shard), device="cpu")
        compatible = {
            key: value for key, value in state.items() if key in compatible_keys
        }
        policy.load_state_dict(compatible, strict=False)
        loaded_keys.update(compatible)
    if loaded_keys != compatible_keys:
        raise RuntimeError(
            "LingBot checkpoint changed between header scan and tensor load: "
            f"missing compatible keys={sorted(compatible_keys - loaded_keys)[:20]}"
        )
    return checkpoint_dir


def move_lingbot_for_runtime(policy: "PreTrainedPolicy") -> None:
    """Apply the config's explicit inference/training initialization dtype."""
    dtype_name = getattr(policy.config, "dtype", "float32")
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        dtype = dtype_by_name[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported LingBot dtype: {dtype_name!r}") from exc
    policy.to(device=policy.config.device, dtype=dtype)
