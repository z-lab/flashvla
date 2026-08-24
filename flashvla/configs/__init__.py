#!/usr/bin/env python
# Copyright 2025 FlashVLA team. All rights reserved.
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

from flashvla.configs.train_config import FlashVLATrainConfig
from flashvla.policies.lingbot import LingBotConfig, LingBotFlashVLAConfig
from flashvla.policies.pi05 import PI05Config, PI05FlashVLAConfig
from flashvla.policies.pi0 import PI0Config, PI0FlashVLAConfig
from flashvla.policies.smolvla import SmolVLAConfig, SmolVLAFlashVLAConfig

from lerobot.configs.policies import PreTrainedConfig as _LRPreTrainedConfig

_LRPreTrainedConfig._choice_registry["pi05"] = PI05Config
_LRPreTrainedConfig._choice_registry["pi05-flashvla"] = PI05FlashVLAConfig
_LRPreTrainedConfig._choice_registry["pi0"] = PI0Config
_LRPreTrainedConfig._choice_registry["pi0-flashvla"] = PI0FlashVLAConfig
_LRPreTrainedConfig._choice_registry["lingbot"] = LingBotConfig
_LRPreTrainedConfig._choice_registry["lingbot-flashvla"] = LingBotFlashVLAConfig
# Read configs saved by the pre-open-source VLASH integration. New saves use
# the canonical ``lingbot-flashvla`` discriminator above.
_LRPreTrainedConfig._choice_registry["lingbot-action-streaming"] = LingBotFlashVLAConfig
_LRPreTrainedConfig._choice_registry["smolvla"] = SmolVLAConfig
_LRPreTrainedConfig._choice_registry["smolvla-flashvla"] = SmolVLAFlashVLAConfig

__all__ = [
    "FlashVLATrainConfig",
    "PI05Config",
    "PI05FlashVLAConfig",
    "PI0Config",
    "PI0FlashVLAConfig",
    "LingBotConfig",
    "LingBotFlashVLAConfig",
    "SmolVLAConfig",
    "SmolVLAFlashVLAConfig",
]
