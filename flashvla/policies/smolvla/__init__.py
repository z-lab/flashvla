"""FlashVLA SmolVLA policy package.

Ported from lerobot/smolvla with RTC support stripped. Architecture:
SmolVLM2-500M + smaller action expert (width × 0.75), joint attention
(cross_attn default, self_attn every 2 layers), flow matching with
per-token AR causal mask in the action suffix.
"""

from __future__ import annotations

from .configuration_smolvla import SmolVLAConfig, SmolVLAFlashVLAConfig
from .modeling_smolvla import SmolVLAPolicy
from .modeling_smolvla_flashvla import SmolVLAFlashVLAPolicy

__all__ = [
    "SmolVLAConfig",
    "SmolVLAPolicy",
    "SmolVLAFlashVLAConfig",
    "SmolVLAFlashVLAPolicy",
]
