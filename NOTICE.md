# Third-party notices

## LingBot-VLA

The implementation under `flashvla/policies/lingbot/` contains code derived
from [Robbyant/lingbot-vla](https://github.com/Robbyant/lingbot-vla), based on
upstream commit `4eb34b7`, and subsequently modified for FlashVLA,
Transformers 5 compatibility, LeRobot processors, strict checkpoint loading,
and policy-owned FSDP2. LingBot-VLA code is distributed under Apache-2.0; see
its upstream repository for the original license and legal notices.

## Hugging Face Transformers

Parts of the vendored Qwen2/Qwen2.5-VL model implementation and flex-attention
helpers are derived from
[Hugging Face Transformers](https://github.com/huggingface/transformers) and
retain their Apache-2.0 copyright headers.

## Model weights and tokenizers

The repository's Apache-2.0 license covers its source code, not third-party
model weights, tokenizers, datasets, or simulator assets. In particular,
LingBot configs refer to `Qwen/Qwen2.5-VL-3B-Instruct`; users must review and
accept that model repository's Qwen Research License before downloading or
using it; it is not the same as this repository's Apache-2.0 source license.
LingBot and FlashVLA checkpoints are likewise governed by the license stated
on their respective model cards.

Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c)
Alibaba Cloud. All Rights Reserved. The referenced Qwen weights are licensed
for non-commercial research or evaluation; commercial use requires a separate
license from Alibaba Cloud. Redistributions of Qwen materials or derivatives
must include the Qwen agreement and its required notices. Models trained or
improved using those materials must prominently state “Built with Qwen” or
“Improved using Qwen” in their product documentation.
