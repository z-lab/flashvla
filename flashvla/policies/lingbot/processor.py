# Portions derived from LingBot-VLA, Copyright Robbyant Team.
# Source: https://github.com/Robbyant/lingbot-vla (commit 4eb34b7).
# Modified by the FlashVLA team. Licensed under Apache-2.0.
"""Processor pipelines for LingBot-VLA policies.

Creates external pre/post-processor pipelines that handle normalization,
language prompt formatting, tokenization, and device transfer.

LingBot-VLA uses the Qwen2.5-VL tokenizer and formats prompts as:
    <bos>{task}\n
(following lingbot-vla's prepare_language convention for weight compatibility)
"""

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from flashvla.policies.lingbot.configuration_lingbot import LingBotConfig


@ProcessorStepRegistry.register(name="flashvla_lingbot_prepare_language_processor_step")
@dataclass
class FlashVLALingBotPrepareLanguageProcessorStep(ProcessorStep):
    """Prepare language prompt for LingBot-VLA policies.

    Formats the task string as ``<bos>{task}\\n`` to match lingbot-vla's
    ``prepare_language`` convention, which is what keeps the token sequence
    weight-compatible with pretrained LingBot checkpoints.
    """

    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        full_prompts = []
        for task in tasks:
            # Preserve the upstream text byte-for-byte apart from the two
            # required wrappers. In particular, do not strip whitespace or
            # rewrite underscores: both change the released checkpoint's
            # token sequence.
            prompt = task if task.startswith("<bos>") else f"<bos>{task}"
            prompt = prompt if prompt.endswith("\n") else f"{prompt}\n"
            full_prompts.append(prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        return transition

    def get_config(self) -> dict[str, Any]:
        return {"task_key": self.task_key}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="vlash_lingbot_prepare_language_processor_step")
@dataclass
class LegacyVLASHLingBotPrepareLanguageProcessorStep(
    FlashVLALingBotPrepareLanguageProcessorStep
):
    """Deserialize processors saved before the VLASH-to-FlashVLA rename.

    New processor pipelines always use the FlashVLA registry name above. This
    subclass exists only so released/development VLASH checkpoints can still
    restore their saved processor JSON without editing it in place.
    """


def make_lingbot_pre_post_processors(
    config: LingBotConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Construct pre/post-processor pipelines for LingBot-VLA policies.

    Preprocessor pipeline:
    1. Rename observations (identity by default)
    2. Add batch dimension
    3. Normalize inputs and outputs using dataset statistics
    4. Prepare language prompt (<bos>{task}\\n)
    5. Tokenize prompt with Qwen2.5-VL tokenizer
    6. Move to device

    Postprocessor pipeline:
    1. Unnormalize outputs
    2. Move to CPU
    """
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        FlashVLALingBotPrepareLanguageProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_path,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
