"""FlashVLA LingBot-VLA policy package.

Ported from LingBot-VLA: a Qwen2.5-VL 3B backbone joined to a Qwen2 action
expert through per-layer joint attention, with flow-matching action decoding
and AdaRMS time conditioning.

Only the configs are re-exported at package import time; the modeling modules
pull in the vendored Qwen2.5-VL stack, so ``flashvla.configs`` stays cheap to
import. Import the policy classes from their modules (or via
``flashvla.policies.factory.get_policy_class``).
"""

from __future__ import annotations

from flashvla.policies.lingbot.configuration_lingbot import (
    LingBotConfig,
    LingBotFlashVLAConfig,
)

__all__ = [
    "LingBotConfig",
    "LingBotFlashVLAConfig",
]
