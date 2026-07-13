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
"""Policy Factory Module.

This module provides factory functions for creating policy instances.
It supports both creating fresh policies and loading pretrained ones.

Usage:
    from flashvla.policies.factory import make_policy, get_policy_class
    
    # Get policy class by name
    policy_cls = get_policy_class("pi05-flashvla")
    
    # Create policy instance
    policy = make_policy(cfg.policy, dataset.meta)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from torch import nn
from lerobot.envs.utils import env_to_policy_features
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.envs.configs import EnvConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import validate_visual_features_consistency
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)

import importlib as _importlib
for _mod in (
    "flashvla.policies.pi0.processor",
    "flashvla.policies.pi05.processor",
    "flashvla.policies.smolvla.processor_smolvla",
):
    _importlib.import_module(_mod)
del _importlib, _mod
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """Get policy class by name.
    
    Args:
        name: Policy type name (e.g. "pi05", "pi05-flashvla").
        
    Returns:
        Policy class (not instance).
        
    Raises:
        NotImplementedError: If policy name is not recognized.
    """
    if name == "pi0":
        from flashvla.policies.pi0.modeling_pi0 import PI0Policy
        return PI0Policy

    if name == "pi0-flashvla":
        from flashvla.policies.pi0.modeling_pi0_flashvla import PI0FlashVLAPolicy
        return PI0FlashVLAPolicy

    if name == "pi05":
        from flashvla.policies.pi05.modeling_pi05 import PI05Policy
        return PI05Policy
    
    if name == "pi05-flashvla":
        from flashvla.policies.pi05.modeling_pi05_flashvla import PI05FlashVLAPolicy
        return PI05FlashVLAPolicy

    if name == "smolvla":
        from flashvla.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        return SmolVLAPolicy

    if name == "smolvla-flashvla":
        from flashvla.policies.smolvla.modeling_smolvla_flashvla import SmolVLAFlashVLAPolicy
        return SmolVLAFlashVLAPolicy

    raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
    rename_map: dict[str, str] | None = None,
) -> PreTrainedPolicy:
    """Create a policy instance from configuration and dataset metadata.
    
    Args:
        cfg: Policy configuration with type, device, pretrained_path, etc.
        ds_meta: Dataset metadata containing feature definitions and stats.
        env_cfg: Environment config used to derive features when ds_meta is None.
        rename_map: Optional feature-rename map for visual-feature validation.

    Returns:
        Initialized policy ready for training or inference.
    """
    policy_cls = get_policy_class(cfg.type)

    kwargs: dict[str, Any] = {}
    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
    else:
        if not cfg.pretrained_path:
            logging.warning(
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                "rather than a dataset."
            )
        if env_cfg is None:
            raise ValueError("env_cfg cannot be None when ds_meta is not provided")
        features = env_to_policy_features(env_cfg)


    cfg.output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    if not cfg.input_features:
        cfg.input_features = {
            key: ft for key, ft in features.items() if key not in cfg.output_features
        }
    kwargs["config"] = cfg

    if cfg.pretrained_path:
        policy = policy_cls.from_pretrained(
            pretrained_name_or_path=cfg.pretrained_path,
            **kwargs,
        )
    else:
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    policy.eval()

    assert isinstance(policy, nn.Module)

    if not rename_map:
        validate_visual_features_consistency(cfg, features)

    return policy


def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
    **kwargs,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create or load pre- and post-processor pipelines for a policy.

    If the pretrained_path contains processor JSON files, loads them directly.
    Otherwise, creates full processor pipelines with normalization, tokenization,
    and device transfer based on the policy type.

    Args:
        policy_cfg: Policy configuration.
        pretrained_path: Path to pretrained model directory.
        dataset_stats: Dataset statistics for normalization.
        **kwargs: Keyword arguments including preprocessor_overrides and
            postprocessor_overrides.

    Returns:
        A tuple of (preprocessor, postprocessor) pipelines.
    """
    if pretrained_path:
        preprocessor_json = Path(pretrained_path) / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
        postprocessor_json = Path(pretrained_path) / f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"

        has_pretrained_processors = preprocessor_json.exists() and postprocessor_json.exists()
        if (
            not has_pretrained_processors
            and dataset_stats is None
            and not Path(pretrained_path).is_dir()
        ):
            try:
                from huggingface_hub import hf_hub_download

                hf_hub_download(str(pretrained_path), f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json")
                hf_hub_download(str(pretrained_path), f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")
                has_pretrained_processors = True
            except Exception:
                has_pretrained_processors = False

        if has_pretrained_processors:
            return (
                PolicyProcessorPipeline.from_pretrained(
                    pretrained_model_name_or_path=pretrained_path,
                    config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
                    overrides=kwargs.get("preprocessor_overrides", {}),
                    to_transition=batch_to_transition,
                    to_output=transition_to_batch,
                ),
                PolicyProcessorPipeline.from_pretrained(
                    pretrained_model_name_or_path=pretrained_path,
                    config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
                    overrides=kwargs.get("postprocessor_overrides", {}),
                    to_transition=policy_action_to_transition,
                    to_output=transition_to_policy_action,
                ),
            )

    policy_type = policy_cfg.type
    if policy_type.startswith("pi05"):
        from flashvla.policies.pi05.processor import make_flashvla_pi05_pre_post_processors
        return make_flashvla_pi05_pre_post_processors(policy_cfg, dataset_stats)
    elif policy_type.startswith("pi0"):
        from flashvla.policies.pi0.processor import make_flashvla_pi0_pre_post_processors
        return make_flashvla_pi0_pre_post_processors(policy_cfg, dataset_stats)
    elif policy_type.startswith("smolvla"):
        from flashvla.policies.smolvla.processor_smolvla import make_flashvla_smolvla_pre_post_processors
        return make_flashvla_smolvla_pre_post_processors(policy_cfg, dataset_stats)
    else:
        raise NotImplementedError(
            f"No processor pipeline defined for policy type {policy_type!r}. "
            "Either add a processor factory or provide pretrained processor JSON files."
        )
