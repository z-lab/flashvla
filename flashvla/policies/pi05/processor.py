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
"""Processor pipelines for FlashVLA PI05 policies.

Creates external pre/post-processor pipelines that handle normalization,
state discretization, tokenization, and device transfer.
"""

from dataclasses import dataclass
from typing import Any
from copy import deepcopy
import torch
import numpy as np

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
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from flashvla.policies.pi05.configuration_pi05 import PI05Config


@ProcessorStepRegistry.register(name="flashvla_pi05_prepare_language_processor_step")
@dataclass
class FlashVLAPi05PrepareLanguageProcessorStep(ProcessorStep):
    """Prepare language prompt for PI05 policies.

    Handles both state_cond modes:
    - state_cond=False: Discretize normalized state into 256 bins and include
      in the prompt as "Task: {task}, State: {state_str};\\nAction: ".
      If a multi-frame state tensor [B, K, state_dim] is provided (i.e. the
      dataloader is feeding history via observation_delta_indices), all K
      frames are flattened in temporal order into a single state string of
      length K * state_dim.
    - state_cond=True: Simple prompt "Task: {task};\\nAction: "
    """

    state_cond: bool = False
    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        if not self.state_cond:
            state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
            if state is None:
                raise ValueError("State is required for PI05 when state_cond=False")

            state = deepcopy(state)

            state_np = state.cpu().numpy()
            discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

            full_prompts = []
            for i, task in enumerate(tasks):
                cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
                per_sample = discretized_states[i]
                if per_sample.ndim == 2:
                    per_sample = per_sample.reshape(-1)
                state_str = " ".join(map(str, per_sample))
                full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
                full_prompts.append(full_prompt)
        else:
            full_prompts = []
            for task in tasks:
                cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
                full_prompt = f"Task: {cleaned_text};\nAction: "
                full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_flashvla_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Construct pre/post-processor pipelines for FlashVLA PI05 policies.

    Preprocessor pipeline:
    1. Rename observations (identity by default)
    2. Add batch dimension
    3. Normalize inputs and outputs using dataset statistics
    4. Prepare language prompt (with optional state discretization)
    5. Tokenize prompt with PaliGemma tokenizer
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
        FlashVLAPi05PrepareLanguageProcessorStep(
            state_cond=config.state_cond,
            max_state_dim=config.max_state_dim,
        ),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
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
