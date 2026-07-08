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
"""FlashVLA Datasets Module.

Provides the FlashVLA action-streaming dataset plus compatibility patches for
lerobot v2.1/v3.0 formats. The compat module is imported first to patch lerobot
before other imports.
"""

import flashvla.datasets.compat  # noqa: F401

from flashvla.datasets.flashvla_dataset import FlashVLADataset, flashvla_collate_fn

__all__ = [
    "FlashVLADataset",
    "flashvla_collate_fn",
]
