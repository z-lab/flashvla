"""Alias handling in load_remapped_checkpoint across the checkpoint layouts PI0.5 sees.

The toy mirrors the real trees: ``vlm``/``action_expert`` each own ``model.embed_tokens``
and ``lm_head`` (tied in FlashVLA, untied in the baseline), ``prefix_embedder.lang_embedder``
is the VLM embedding module under a second name, and the joint layer aliases the backbone
layers. Three file layouts are exercised: an FSDP2 export that wrote every alias (with a
stale ``lm_head``), a safetensors ``save_model`` export that kept one alphabetical name per
tensor, and a raw openpi base that carries only ``lm_head`` keys.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file, save_model
from torch import nn

from flashvla.policies.loading import load_remapped_checkpoint

RAW_RULES = [
    ("paligemma_with_expert.gemma_expert.", "model.action_expert."),
    ("paligemma_with_expert.paligemma.", "model.vlm."),
]


class _Backbone(nn.Module):
    def __init__(self, vocab: int, dim: int, tie: bool) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, dim)
        self.model.layers = nn.ModuleList([nn.Linear(dim, dim, bias=False)])
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        if tie:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens


class _Policy(nn.Module):
    def __init__(self, tie: bool) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.vlm = _Backbone(8, 4, tie)
        self.model.action_expert = _Backbone(8, 2, tie)
        self.model.prefix_embedder = nn.Module()
        self.model.prefix_embedder.lang_embedder = self.model.vlm.model.embed_tokens
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].vlm_layer = self.model.vlm.model.layers[0]
        self.model.layers[0].expert_layer = self.model.action_expert.model.layers[0]
        self.model.out = nn.Linear(2, 3)

    def embed(self) -> torch.Tensor:
        return self.model.vlm.model.embed_tokens.weight


def _trained(tie: bool) -> _Policy:
    torch.manual_seed(0)
    policy = _Policy(tie)
    with torch.no_grad():
        policy.model.vlm.model.embed_tokens.weight.fill_(7.0)
        policy.model.action_expert.model.embed_tokens.weight.fill_(5.0)
        if not tie:  # the baseline's dead heads keep their pretrained value
            policy.model.vlm.lm_head.weight.fill_(-1.0)
            policy.model.action_expert.lm_head.weight.fill_(-1.0)
    return policy


class AliasLoadingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _load(self, name: str, tie: bool, rules=()) -> _Policy:
        policy = _Policy(tie)
        load_remapped_checkpoint(policy, str(self.dir / name), rules, source=name)
        return policy

    def _fsdp_export(self) -> str:
        # Every alias as its own tensor, backbone layer aliases already detached,
        # and the dormant lm_head holding a stale value.
        sd = {k: v.detach().clone() for k, v in _trained(tie=True).state_dict().items()}
        for k in [k for k in sd if ".model.layers." in k]:
            del sd[k]
        sd["model.vlm.lm_head.weight"].fill_(-100.0)
        save_file(sd, str(self.dir / "fsdp.safetensors"))
        return "fsdp.safetensors"

    def _save_model(self, tie: bool) -> str:
        name = f"save_model_{'tied' if tie else 'untied'}.safetensors"
        save_model(_trained(tie), str(self.dir / name))
        return name

    def _raw(self) -> str:
        src = _trained(tie=True).state_dict()
        sd = {
            "paligemma_with_expert.paligemma.lm_head.weight": src["model.vlm.lm_head.weight"],
            "paligemma_with_expert.paligemma.model.layers.0.weight": src["model.vlm.model.layers.0.weight"],
            "paligemma_with_expert.gemma_expert.lm_head.weight": src["model.action_expert.lm_head.weight"],
            "paligemma_with_expert.gemma_expert.model.layers.0.weight": src["model.action_expert.model.layers.0.weight"],
            "model.out.weight": src["model.out.weight"],
            "model.out.bias": src["model.out.bias"],
        }
        save_file({k: v.detach().clone() for k, v in sd.items()}, str(self.dir / "raw.safetensors"))
        return "raw.safetensors"

    def test_fsdp_export_prefers_trained_embedding_over_stale_head(self) -> None:
        policy = self._load(self._fsdp_export(), tie=True)
        self.assertTrue(torch.all(policy.embed() == 7.0))
        self.assertIs(policy.model.vlm.lm_head.weight, policy.embed())
        self.assertTrue(torch.all(policy.model.action_expert.model.embed_tokens.weight == 5.0))

    def test_fsdp_export_into_untied_baseline(self) -> None:
        policy = self._load(self._fsdp_export(), tie=False)
        self.assertTrue(torch.all(policy.embed() == 7.0))
        self.assertTrue(torch.all(policy.model.vlm.lm_head.weight == -100.0))

    def test_save_model_exports_round_trip(self) -> None:
        # (file tied, model untied) is excluded: a tied export carries no separate
        # head tensor for the baseline's untied lm_head, so that cross-load fails
        # the coverage check by design.
        for file_tie, model_tie in ((True, True), (False, True), (False, False)):
            with self.subTest(file_tie=file_tie, model_tie=model_tie):
                policy = self._load(self._save_model(file_tie), tie=model_tie)
                self.assertTrue(torch.all(policy.embed() == 7.0))
                if not model_tie:
                    self.assertTrue(torch.all(policy.model.vlm.lm_head.weight == -1.0))

    def test_raw_base_fills_embeddings_from_heads(self) -> None:
        for tie in (True, False):
            with self.subTest(tie=tie):
                policy = self._load(self._raw(), tie=tie, rules=RAW_RULES)
                self.assertTrue(torch.all(policy.embed() == 7.0))
                self.assertTrue(torch.all(policy.model.vlm.lm_head.weight == 7.0))

    def test_missing_unexpected_and_shape_mismatch_are_fatal(self) -> None:
        sd = {k: v.detach().clone() for k, v in _trained(tie=False).state_dict().items()}
        cases = {
            "missing": ({k: v for k, v in sd.items() if k != "model.out.bias"}, "random initialization"),
            "unexpected": ({**sd, "model.bogus.weight": torch.zeros(1)}, "Unexpected keys"),
            "shape": ({**sd, "model.out.bias": torch.zeros(5)}, "model.out.bias"),
        }
        for name, (state, message) in cases.items():
            with self.subTest(name):
                save_file(state, str(self.dir / f"{name}.safetensors"))
                with self.assertRaisesRegex(RuntimeError, message):
                    self._load(f"{name}.safetensors", tie=False)


if __name__ == "__main__":
    unittest.main()
