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
"""FlashVLA Training Module.

Trains flashvla policies (pi05 / pi0 / smolvla) with shared
observation and padded cold start.

Usage:
    python train/train.py --config_path=train/configs/pi05/libero/flashvla_action.yaml
"""

import logging
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
import accelerate as accelerate_lib
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer
import shutil

from lerobot.configs import parser
from lerobot.datasets.factory import (
    IMAGENET_STATS,
    resolve_delta_timestamps,
)
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.constants import PRETRAINED_MODEL_DIR, TRAINING_STATE_DIR
from lerobot.utils.train_utils import (get_step_checkpoint_dir, load_training_state, load_training_step, save_checkpoint, save_training_step, update_last_checkpoint)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)
from tqdm import tqdm

from flashvla.configs.train_config import FlashVLATrainConfig
from flashvla.datasets.flashvla_dataset import (
    FlashVLADataset,
    flashvla_collate_fn,
    make_robotwin_multitask_flashvla_dataset,
)
from flashvla.policies.factory import make_policy, make_pre_post_processors


ACCELERATOR_STATE_DIR = "accelerator_state"
REPLICATED_OPTIMIZER_STATE = "replicated_optimizer_state.pt"
FSDP_POLICY_CLASS_NAMES = {
    "pi0-flashvla": ("PI0ModelLayer", "PI0SuffixEmbedder"),
    "pi05-flashvla": ("PI05ModelLayer", "PI05SuffixEmbedder"),
}


def count_parameters(model: torch.nn.Module, only_trainable: bool = False) -> int:
    """Count model parameters."""
    return sum(
        p.numel()
        for p in model.parameters()
        if not only_trainable or p.requires_grad
    )


def resolve_fsdp_module_classes(
    policy_type: str,
    mixed_precision: str,
    wrap_layers: list[str],
    fp32_module_classes: list[str],
) -> tuple[list[str], list[str]]:
    """Add the policy-specific FSDP classes without mutating the config."""
    resolved_wrap_layers = list(dict.fromkeys(wrap_layers))
    resolved_fp32_module_classes = list(dict.fromkeys(fp32_module_classes))
    policy_classes = FSDP_POLICY_CLASS_NAMES.get(policy_type)
    if policy_classes is None:
        return resolved_wrap_layers, resolved_fp32_module_classes

    model_layer_class, suffix_class = policy_classes
    if model_layer_class not in resolved_wrap_layers:
        resolved_wrap_layers.insert(0, model_layer_class)

    if mixed_precision == "bf16":
        if suffix_class not in resolved_fp32_module_classes:
            resolved_fp32_module_classes.append(suffix_class)
    elif mixed_precision == "no":
        if suffix_class not in resolved_wrap_layers:
            resolved_wrap_layers.append(suffix_class)
    else:
        raise ValueError(f"Unsupported FSDP mixed precision mode: {mixed_precision!r}")

    return resolved_wrap_layers, resolved_fp32_module_classes


def build_fsdp_mixed_precision_policy(mixed_precision: str, reduce_dtype: str):
    """Build the FSDP2 policy without enabling global autocast.

    Returns None for full-fp32 mode. In bf16 mode, parameters are gathered and
    computed in bf16 while gradient reductions stay in ``reduce_dtype`` (fp32 by
    default) and each module's natural output dtype is preserved so the fp32
    flow-matching loss from action_out_proj is not downcast.
    """
    if mixed_precision == "no":
        return None
    if mixed_precision != "bf16":
        raise ValueError(f"Unsupported FSDP mixed precision mode: {mixed_precision!r}")

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        resolved_reduce_dtype = dtype_by_name[reduce_dtype]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported fsdp.reduce_dtype={reduce_dtype!r}; expected 'bf16' or 'float32'."
        ) from exc

    from torch.distributed.fsdp import MixedPrecisionPolicy

    return MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=resolved_reduce_dtype,
        output_dtype=None,
        cast_forward_inputs=True,
    )


class Fp32LayerNorm(torch.nn.LayerNorm):
    """Run an ignored LayerNorm in fp32 and preserve its input dtype.

    FSDP2's module-level policy does not cast the inputs of ignored modules, so
    a bf16 activation would otherwise meet an fp32 LayerNorm weight. This casts
    to the weight dtype for the computation and casts the result back.
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output_dtype = inputs.dtype
        compute_dtype = self.weight.dtype if self.weight is not None else torch.float32
        outputs = torch.nn.functional.layer_norm(
            inputs.to(dtype=compute_dtype),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )
        return outputs.to(dtype=output_dtype)


def install_fsdp_bf16_checkpoint_input_casts(policy: PreTrainedPolicy) -> list[str]:
    """Cast inputs before checkpoint wrappers capture their recompute tensors.

    Transformers' ``GradientCheckpointingLayer.__call__`` sits outside FSDP2's
    forward pre-hook. The first SigLIP layer would otherwise checkpoint the
    fp32 output of the ignored vision embeddings, even though FSDP casts the
    initial computation to bf16. During backward FSDP is already in
    PRE_BACKWARD and deliberately skips that input cast, causing fp32 inputs
    to meet bf16 unsharded parameters on recompute.
    """
    path = "model.vlm.model.vision_tower.vision_model.encoder"
    try:
        encoder = policy.get_submodule(path)
    except AttributeError:
        return []

    def cast_inputs(_module, args, kwargs):
        args = list(args)
        kwargs = dict(kwargs)
        if args and isinstance(args[0], torch.Tensor) and args[0].is_floating_point():
            args[0] = args[0].to(torch.bfloat16)
        inputs_embeds = kwargs.get("inputs_embeds")
        if isinstance(inputs_embeds, torch.Tensor) and inputs_embeds.is_floating_point():
            kwargs["inputs_embeds"] = inputs_embeds.to(torch.bfloat16)
        return tuple(args), kwargs

    encoder.register_forward_pre_hook(cast_inputs, prepend=True, with_kwargs=True)
    return [path]


def _base_optimizer(optimizer: Optimizer) -> Optimizer:
    """Unwrap Accelerate's AcceleratedOptimizer to the underlying torch optimizer."""
    while hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer
    return optimizer


@contextmanager
def exclude_params_from_optimizer(
    optimizer: Optimizer,
    excluded_params: list[torch.nn.Parameter],
):
    """Temporarily hide replicated params from FSDP2 optimizer state APIs.

    Accelerate's FSDP2 optimizer save/load only round-trips sharded DTensor
    state. The fp32/ignored replicated params carry regular Tensor optimizer
    state that is saved and loaded separately, so hide them here while the
    sharded state is written or read.
    """
    base_optimizer = _base_optimizer(optimizer)
    excluded_ids = {id(param) for param in excluded_params}
    original_group_params = [list(group["params"]) for group in base_optimizer.param_groups]
    excluded_state = {
        param: base_optimizer.state.pop(param)
        for param in excluded_params
        if param in base_optimizer.state
    }
    for group in base_optimizer.param_groups:
        group["params"] = [param for param in group["params"] if id(param) not in excluded_ids]
    try:
        yield
    finally:
        for group, params in zip(base_optimizer.param_groups, original_group_params, strict=True):
            group["params"] = params
        base_optimizer.state.update(excluded_state)


def _replicated_optimizer_state_by_name(
    policy: torch.nn.Module,
    optimizer: Optimizer,
    replicated_params: list[torch.nn.Parameter],
) -> dict[str, dict[str, Any]]:
    base_optimizer = _base_optimizer(optimizer)
    name_by_id = {id(param): name for name, param in policy.named_parameters()}
    state_by_name = {}
    for param in replicated_params:
        state = base_optimizer.state.get(param)
        if not state:
            continue
        name = name_by_id.get(id(param))
        if name is None:
            raise KeyError("Replicated optimizer parameter has no stable model name")
        serialized = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                serialized[key] = {
                    "tensor": value.detach().cpu(),
                    "device_type": value.device.type,
                }
            else:
                serialized[key] = value
        state_by_name[name] = serialized
    return state_by_name


def save_replicated_optimizer_state(
    path: Path,
    policy: torch.nn.Module,
    optimizer: Optimizer,
    replicated_params: list[torch.nn.Parameter],
) -> None:
    payload = {
        "version": 1,
        "state": _replicated_optimizer_state_by_name(
            policy,
            optimizer,
            replicated_params,
        ),
    }
    torch.save(payload, path)


def load_replicated_optimizer_state(
    path: Path,
    policy: torch.nn.Module,
    optimizer: Optimizer,
    replicated_params: list[torch.nn.Parameter],
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"FSDP2 checkpoint is missing replicated optimizer state: {path}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported replicated optimizer state version in {path}")

    base_optimizer = _base_optimizer(optimizer)
    param_by_name = {name: param for name, param in policy.named_parameters()}
    replicated_ids = {id(param) for param in replicated_params}
    unexpected = []
    for name, state in payload["state"].items():
        param = param_by_name.get(name)
        if param is None or id(param) not in replicated_ids:
            unexpected.append(name)
            continue
        restored = {}
        for key, value in state.items():
            if isinstance(value, dict) and "tensor" in value:
                device = torch.device("cpu") if value["device_type"] == "cpu" else param.device
                restored[key] = value["tensor"].to(device=device)
            else:
                restored[key] = value
        base_optimizer.state[param] = restored
    if unexpected:
        raise KeyError(
            "Replicated optimizer checkpoint contains parameters absent from the model: "
            f"{unexpected[:20]}"
        )


def load_fsdp2_sharded_optimizer_allow_partial(
    fsdp_plugin,
    accelerator: Accelerator,
    optimizer: Optimizer,
    model: torch.nn.Module,
    input_dir: str,
    optimizer_index: int = 0,
    adapter_only: bool = False,
) -> None:
    """Load FSDP2 optimizer state while allowing legitimately absent lazy state.

    The sharded optimizer state omits entries for params whose optimizer state
    has not been materialized yet (e.g. lazily on first step). Allowing a
    partial load prevents a hard failure on resume.
    """
    from accelerate.utils.fsdp_utils import (
        _prepare_sd_options,
        load_fsdp_optimizer as default_load_fsdp_optimizer,
    )
    from torch.distributed.checkpoint import (
        DefaultLoadPlanner,
        FileSystemReader,
        load,
    )
    from torch.distributed.checkpoint.state_dict import (
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    if (
        fsdp_plugin.fsdp_version != 2
        or fsdp_plugin.state_dict_type != StateDictType.SHARDED_STATE_DICT
    ):
        default_load_fsdp_optimizer(
            fsdp_plugin,
            accelerator,
            optimizer,
            model,
            input_dir,
            optimizer_index,
            adapter_only,
        )
        return

    accelerator.wait_for_everyone()
    checkpoint_dir = Path(input_dir) / f"optimizer_{optimizer_index}"
    options = _prepare_sd_options(fsdp_plugin)
    optimizer_state = get_optimizer_state_dict(model, optimizer, options=options)
    state_dict = {"optimizer": optimizer_state}
    load(
        state_dict,
        checkpoint_id=checkpoint_dir,
        storage_reader=FileSystemReader(checkpoint_dir),
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )
    set_optimizer_state_dict(
        model,
        optimizer,
        state_dict["optimizer"],
        options=options,
    )


@contextmanager
def patch_accelerate_fsdp_optimizer_loader():
    """Use the partial optimizer loader only during one Accelerator load."""
    import importlib

    accelerator_module = importlib.import_module("accelerate.accelerator")
    original_loader = accelerator_module.load_fsdp_optimizer
    accelerator_module.load_fsdp_optimizer = load_fsdp2_sharded_optimizer_allow_partial
    try:
        yield
    finally:
        accelerator_module.load_fsdp_optimizer = original_loader


def make_flashvla_dataset(cfg: FlashVLATrainConfig):
    """Create a FlashVLADataset for FlashVLA training.

    When `cfg.robotwin_multitask.enable` is True, builds a concatenation of
    per-task FlashVLADataset subsets from the RoboTwin-LeRobot-v3.0
    directory layout. Otherwise returns a single FlashVLADataset built
    from `cfg.dataset.repo_id`/`cfg.dataset.root`.

    Args:
        cfg: Training configuration.

    Returns:
        (Multi)FlashVLADataset instance.
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    mt_cfg = getattr(cfg, "robotwin_multitask", None)
    if mt_cfg is not None and mt_cfg.enable:
        dataset = make_robotwin_multitask_flashvla_dataset(cfg, image_transforms=image_transforms)
        if cfg.dataset.use_imagenet_stats:
            for key in dataset.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
        return dataset

    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )

    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

    dataset = FlashVLADataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        num_buffer_slots=cfg.policy.num_buffer_slots,
        chunk_size=cfg.policy.chunk_size,
    )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset



def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    *,
    loss_scale: float = 1.0,
    do_step: bool = True,
    autocast_ctx=None,
    replicated_params_to_sync: list[torch.nn.Parameter] | None = None,
) -> tuple[MetricsTracker, dict]:
    """Performs a single training step."""
    policy.train()

    with autocast_ctx if autocast_ctx is not None else accelerator.autocast():
        # Invoke nn.Module.__call__ so distributed wrappers can run their
        # pre/post-forward hooks (in particular FSDP2 root unshard/reshard).
        loss, output_dict = policy(batch)
        raw_loss = loss.detach()
        loss = loss * loss_scale

    accelerator.backward(loss)

    grad_norm_value: float | None = None

    if do_step:
        if replicated_params_to_sync:
            sync_replicated_gradients(replicated_params_to_sync, accelerator)

        if grad_clip_norm > 0:
            if replicated_params_to_sync:
                grad_norm = clip_fsdp2_mixed_grad_norm_(
                    list(policy.parameters()),
                    grad_clip_norm,
                    accelerator,
                )
            else:
                grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        else:
            if replicated_params_to_sync:
                grad_norm = clip_fsdp2_mixed_grad_norm_(
                    list(policy.parameters()), float("inf"), accelerator
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), float("inf"), error_if_nonfinite=False
                )
        grad_norm_value = grad_norm.item()

        with lock if lock is not None else nullcontext():
            optimizer.step()

        optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
            accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = raw_loss.item()
    if grad_norm_value is not None:
        train_metrics.grad_norm = grad_norm_value
    train_metrics.lr = optimizer.param_groups[0]["lr"]

    return train_metrics, output_dict


def sync_replicated_gradients(
    params: list[torch.nn.Parameter],
    accelerator: Accelerator,
) -> None:
    """Average gradients for trainable params excluded from FSDP2 sharding."""
    if accelerator.num_processes <= 1:
        return
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return

    for param in params:
        if param.grad is None:
            continue
        torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
        param.grad.div_(accelerator.num_processes)


def broadcast_replicated_parameters(
    params: list[torch.nn.Parameter],
    accelerator: Accelerator,
) -> None:
    """Make the initial state of ignored trainable parameters identical.

    Ignored (replicated) params are not managed by FSDP, so broadcast rank 0's
    values to keep every rank's replica bit-identical before training.
    """
    if accelerator.num_processes <= 1:
        return
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return

    for param in params:
        torch.distributed.broadcast(param.data, src=0)


def clip_fsdp2_mixed_grad_norm_(
    params: list[torch.nn.Parameter],
    max_norm: float,
    accelerator: Accelerator,
) -> torch.Tensor:
    """Clip a mix of FSDP2 DTensor grads and replicated Tensor grads."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover - older torch without DTensor
        return accelerator.clip_grad_norm_(params, max_norm)

    device = accelerator.device
    replicated_sq = torch.zeros((), device=device, dtype=torch.float32)
    sharded_sq = torch.zeros((), device=device, dtype=torch.float32)
    grads = []

    for param in params:
        grad = param.grad
        if grad is None:
            continue
        grads.append(grad)
        if isinstance(grad, DTensor):
            local_grad = grad.to_local()
            grad_sq = local_grad.detach().float().pow(2).sum()
            is_sharded = any(type(placement).__name__ == "Shard" for placement in grad.placements)
            if is_sharded:
                sharded_sq = sharded_sq + grad_sq
            else:
                replicated_sq = replicated_sq + grad_sq
        else:
            replicated_sq = replicated_sq + grad.detach().float().pow(2).sum()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(sharded_sq, op=torch.distributed.ReduceOp.SUM)

    total_norm = (replicated_sq + sharded_sq).sqrt()
    clip_coef = float(max_norm) / (total_norm.item() + 1e-6)
    if clip_coef < 1.0:
        for grad in grads:
            grad.mul_(clip_coef)
    return total_norm


def disable_optimizer_foreach(optimizer: Optimizer) -> None:
    """Avoid foreach kernels mixing regular Tensor params with DTensor params."""
    optimizer.defaults["foreach"] = False
    if "fused" in optimizer.defaults:
        optimizer.defaults["fused"] = False
    for group in optimizer.param_groups:
        group["foreach"] = False
        if "fused" in group:
            group["fused"] = False


def save_fsdp2_checkpoint(
    *,
    accelerator: Accelerator,
    checkpoint_dir: Path,
    step: int,
    cfg: FlashVLATrainConfig,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    replicated_params: list[torch.nn.Parameter],
    preprocessor,
    postprocessor,
) -> None:
    """Save FSDP2 checkpoints in two forms.

    `accelerator.save_state()` owns the sharded model/optimizer/scheduler state
    used for exact resume. `accelerator.save_model()` exports a normal
    pretrained_model/model.safetensors for inference/evaluation code that does
    not know about FSDP2.
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR

    accelerator.wait_for_everyone()
    accelerator.save_model(
        policy,
        pretrained_dir,
        max_shard_size=cfg.fsdp.save_pretrained_max_shard_size,
        safe_serialization=True,
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped_policy = accelerator.unwrap_model(policy)
        unwrapped_policy.config.save_pretrained(pretrained_dir)
        cfg.save_pretrained(pretrained_dir)
        if preprocessor is not None:
            preprocessor.save_pretrained(pretrained_dir)
        if postprocessor is not None:
            postprocessor.save_pretrained(pretrained_dir)

        training_state_dir.mkdir(parents=True, exist_ok=True)
        save_training_step(step, training_state_dir)

    accelerator.wait_for_everyone()
    accelerator_state_dir = checkpoint_dir / ACCELERATOR_STATE_DIR
    with exclude_params_from_optimizer(optimizer, replicated_params):
        accelerator.save_state(str(accelerator_state_dir), safe_serialization=True)
    if accelerator.is_main_process:
        save_replicated_optimizer_state(
            accelerator_state_dir / REPLICATED_OPTIMIZER_STATE,
            policy,
            optimizer,
            replicated_params,
        )
    accelerator.wait_for_everyone()


def load_fsdp2_checkpoint(
    *,
    accelerator: Accelerator,
    checkpoint_dir: Path,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    replicated_params: list[torch.nn.Parameter],
) -> int:
    """Load FSDP2 training state after `accelerator.prepare()`."""
    accelerator_state_dir = checkpoint_dir / ACCELERATOR_STATE_DIR
    if not accelerator_state_dir.is_dir():
        raise NotADirectoryError(
            f"FSDP2 resume requires {accelerator_state_dir}. "
            "Older LeRobot-style checkpoints do not contain sharded FSDP2 optimizer state."
        )

    step = load_training_step(checkpoint_dir / TRAINING_STATE_DIR)
    with (
        exclude_params_from_optimizer(optimizer, replicated_params),
        patch_accelerate_fsdp_optimizer_loader(),
    ):
        accelerator.load_state(str(accelerator_state_dir))
    load_replicated_optimizer_state(
        accelerator_state_dir / REPLICATED_OPTIMIZER_STATE,
        policy,
        optimizer,
        replicated_params,
    )
    return step


def auto_resume(cfg: FlashVLATrainConfig) -> None:
    """Automatically resume training from the last checkpoint if available."""
    if cfg.resume or cfg.output_dir is None:
        return

    output_dir_path = Path(cfg.output_dir)
    checkpoints_dir = output_dir_path / "checkpoints"

    if checkpoints_dir.is_dir():
        last_checkpoint = checkpoints_dir / "last"
        candidate_checkpoint_dir = None

        if last_checkpoint.exists():
            candidate_checkpoint_dir = last_checkpoint

        if candidate_checkpoint_dir is not None:
            cfg.resume = True

            train_config_json = candidate_checkpoint_dir / "pretrained_model" / "train_config.json"
            if train_config_json.is_file():
                new_argv: list[str] = [sys.argv[0]]
                for arg in sys.argv[1:]:
                    if arg.startswith("--config_path="):
                        continue
                    new_argv.append(arg)
                new_argv.append(f"--config_path={train_config_json}")
                sys.argv = new_argv

                print(f"Auto-resume enabled. Using config_path={train_config_json} and resume=True.")
        else:
            shutil.rmtree(output_dir_path, ignore_errors=True)
            print(f"Existing output directory {output_dir_path} had checkpoints dir but no valid 'last' checkpoint; "
                  "it has been removed to start a fresh run.")

    elif output_dir_path.is_dir():
        shutil.rmtree(output_dir_path, ignore_errors=True)
        print(f"Existing output directory {output_dir_path} had no checkpoints; "
              "it has been removed to start a fresh run.")


@parser.wrap()
def train(cfg: FlashVLATrainConfig, accelerator: Accelerator | None = None):
    """Main training function for FlashVLA."""
    auto_resume(cfg)
    cfg.validate()

    fsdp_mixed_precision = getattr(cfg.fsdp, "mixed_precision", "auto")
    if fsdp_mixed_precision == "auto":
        fsdp_mixed_precision = "no"
    if fsdp_mixed_precision not in {"no", "bf16"}:
        raise ValueError(
            f"Unsupported fsdp.mixed_precision={fsdp_mixed_precision!r}. "
            "Supported values are 'auto', 'no', and 'bf16'."
        )
    fsdp_reduce_dtype = getattr(cfg.fsdp, "reduce_dtype", "float32")
    fsdp_precision_policy = (
        build_fsdp_mixed_precision_policy(fsdp_mixed_precision, fsdp_reduce_dtype)
        if cfg.fsdp.enable
        else None
    )
    fsdp_wrap_layers = list(cfg.fsdp.wrap_layers)
    fsdp_fp32_module_classes = list(cfg.fsdp.fp32_module_classes)
    if cfg.fsdp.enable:
        fsdp_wrap_layers, fsdp_fp32_module_classes = resolve_fsdp_module_classes(
            cfg.policy.type,
            fsdp_mixed_precision,
            fsdp_wrap_layers,
            fsdp_fp32_module_classes,
        )
    fsdp_policy_dtype_override = None
    if cfg.fsdp.enable and fsdp_mixed_precision == "no" and getattr(cfg.policy, "dtype", "float32") != "float32":
        old_policy_dtype = cfg.policy.dtype
        cfg.policy.dtype = "float32"
        fsdp_policy_dtype_override = (old_policy_dtype, cfg.policy.dtype)

    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        if cfg.deepspeed.enable and cfg.fsdp.enable:
            raise ValueError("deepspeed and fsdp are mutually exclusive; enable only one.")

        kwargs_handlers = []
        ds_plugin = None
        fsdp_plugin = None

        if cfg.deepspeed.enable:
            from accelerate import DeepSpeedPlugin

            ds_config = {
                "train_micro_batch_size_per_gpu": cfg.batch_size,
                "gradient_accumulation_steps": cfg.grad_accum_steps,
                "gradient_clipping": cfg.optimizer.grad_clip_norm,
                "zero_optimization": {
                    "stage": cfg.deepspeed.stage,
                    "allgather_partitions": True,
                    "allgather_bucket_size": cfg.deepspeed.allgather_bucket_size,
                    "reduce_scatter": True,
                    "reduce_bucket_size": cfg.deepspeed.reduce_bucket_size,
                    "overlap_comm": cfg.deepspeed.overlap_comm,
                },
                "bf16": {"enabled": False},
                "fp16": {"enabled": False},
            }
            if cfg.deepspeed.offload_optimizer:
                ds_config["zero_optimization"]["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }

            ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)
        elif cfg.fsdp.enable:
            from accelerate import FullyShardedDataParallelPlugin

            fsdp_plugin = FullyShardedDataParallelPlugin(
                fsdp_version=2,
                reshard_after_forward=cfg.fsdp.reshard_after_forward,
                cpu_offload=cfg.fsdp.cpu_offload,
                mixed_precision_policy=fsdp_precision_policy,
                state_dict_type=cfg.fsdp.state_dict_type,
                auto_wrap_policy="transformer_based_wrap" if fsdp_wrap_layers else "no_wrap",
                transformer_cls_names_to_wrap=fsdp_wrap_layers or None,
            )
        else:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            kwargs_handlers.append(ddp_kwargs)

        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=kwargs_handlers,
            deepspeed_plugin=ds_plugin,
            fsdp_plugin=fsdp_plugin,
            mixed_precision="no" if cfg.fsdp.enable else None,
        )

    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process

    if is_main_process:
        logging.info(pformat(cfg.to_dict()))
        if cfg.deepspeed.enable:
            logging.info(
                f"DeepSpeed enabled with ZeRO stage={cfg.deepspeed.stage}, "
                f"offload_optimizer={cfg.deepspeed.offload_optimizer}, "
                f"overlap_comm={cfg.deepspeed.overlap_comm}"
            )
        if cfg.fsdp.enable:
            logging.info(
                f"Runtime: torch={torch.__version__}, "
                f"accelerate={accelerate_lib.__version__}, cuda={torch.version.cuda}"
            )
            logging.info(
                f"FSDP2 enabled with reshard_after_forward={cfg.fsdp.reshard_after_forward}, "
                f"mixed_precision={fsdp_mixed_precision}, "
                f"reduce_dtype={fsdp_reduce_dtype}, global_autocast={accelerator.mixed_precision}, "
                f"state_dict_type={cfg.fsdp.state_dict_type}, "
                f"wrap_layers={fsdp_wrap_layers}, "
                f"ignored_module_classes={cfg.fsdp.ignored_module_classes}, "
                f"fp32_module_classes={fsdp_fp32_module_classes}, "
                f"ignored_module_name_suffixes={cfg.fsdp.ignored_module_name_suffixes}"
            )
            if fsdp_policy_dtype_override is not None:
                old_dtype, new_dtype = fsdp_policy_dtype_override
                logging.info(
                    f"FSDP2 full-fp32 mode: overriding policy.dtype={old_dtype!r} "
                    f"to {new_dtype!r}"
                )

    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    if getattr(cfg, "cudnn_deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    make_dataset_fn = make_flashvla_dataset

    if is_main_process:
        logging.info(f"Creating dataset for {cfg.policy.type}")
        dataset = make_dataset_fn(cfg)

    accelerator.wait_for_everyone()

    if not is_main_process:
        dataset = make_dataset_fn(cfg)

    if is_main_process:
        logging.info("Creating policy")

    policy_config_dtype = getattr(cfg.policy, "dtype", None)
    if cfg.fsdp.enable and fsdp_mixed_precision == "bf16" and policy_config_dtype == "bfloat16":
        cfg.policy.dtype = "float32"
        if is_main_process:
            logging.info("FSDP2 bf16: initializing policy parameters as fp32 master weights")

    try:
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
        )
    finally:
        if policy_config_dtype is not None:
            cfg.policy.dtype = policy_config_dtype
    if policy_config_dtype is not None:
        policy.config.dtype = policy_config_dtype

    if (
        cfg.fsdp.enable
        and hasattr(policy, "model")
        and hasattr(policy.model, "detach_backbone_layer_aliases")
    ):
        parameter_ids_before = {id(param) for param in policy.parameters()}
        policy.model.detach_backbone_layer_aliases()
        parameter_ids_after = {id(param) for param in policy.parameters()}
        if parameter_ids_before != parameter_ids_after:
            raise RuntimeError(
                "Detaching duplicate backbone paths changed the FSDP parameter set: "
                f"before={len(parameter_ids_before)}, after={len(parameter_ids_after)}, "
                f"dropped={len(parameter_ids_before - parameter_ids_after)}, "
                f"added={len(parameter_ids_after - parameter_ids_before)}"
            )
        if is_main_process:
            logging.info(
                "FSDP2: detached duplicate backbone layer paths while preserving "
                f"{len(parameter_ids_after)} unique parameter objects"
            )

    if cfg.fsdp.enable and fsdp_mixed_precision == "bf16":
        checkpoint_input_casts = install_fsdp_bf16_checkpoint_input_casts(policy)
        if is_main_process and checkpoint_input_casts:
            logging.info(
                "FSDP2 bf16: installed checkpoint-safe input casts on "
                f"{checkpoint_input_casts}"
            )

    if getattr(cfg.policy, "freeze_vision_encoder", False):
        if hasattr(policy.model, "vlm") and hasattr(policy.model.vlm, "model") and hasattr(policy.model.vlm.model, "vision_tower"):
            vision_tower = policy.model.vlm.model.vision_tower
        elif hasattr(policy.model, "vlm_with_expert"):
            vision_tower = policy.model.vlm_with_expert.vlm.model.vision_model
        else:
            raise AttributeError("Cannot find vision tower to freeze")
        for param in vision_tower.parameters():
            param.requires_grad = False
        if is_main_process:
            n_frozen = sum(p.numel() for p in vision_tower.parameters())
            logging.info(f"Vision encoder frozen ({n_frozen:,} params)")

    accelerator.wait_for_everyone()

    if is_main_process:
        logging.info("Creating preprocessor and postprocessor")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=dataset.meta.stats,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0

    if cfg.resume and not cfg.fsdp.enable:
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler
        )

    num_learnable_params = count_parameters(policy, only_trainable=True)
    num_total_params = count_parameters(policy, only_trainable=False)

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")

        num_processes = accelerator.num_processes
        micro_bs = cfg.batch_size * num_processes
        effective_bs = micro_bs * cfg.grad_accum_steps
        logging.info(
            f"Effective batch size (per optimizer step): "
            f"{cfg.batch_size} x {num_processes} x {cfg.grad_accum_steps} = {effective_bs}"
        )
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        collate_fn=flashvla_collate_fn,
    )

    fsdp_replicated_params_to_sync: list[torch.nn.Parameter] = []
    fp32_module_classes = (
        set(fsdp_fp32_module_classes)
        if fsdp_mixed_precision == "bf16"
        else set()
    )
    if cfg.fsdp.enable and (
        cfg.fsdp.ignored_module_classes
        or cfg.fsdp.ignored_module_name_suffixes
        or fp32_module_classes
    ):
        ignored_class_names = set(cfg.fsdp.ignored_module_classes)
        all_ignored_class_names = ignored_class_names | fp32_module_classes
        ignored_name_suffixes = tuple(cfg.fsdp.ignored_module_name_suffixes)
        ignored = []
        seen_module_ids = set()
        matched_names = []
        patched_fp32_layernorm_names = []

        for name, module in policy.named_modules():
            class_match = type(module).__name__ in all_ignored_class_names
            name_match = ignored_name_suffixes and name.endswith(ignored_name_suffixes)
            if not (class_match or name_match):
                continue
            module_id = id(module)
            if module_id in seen_module_ids:
                continue
            seen_module_ids.add(module_id)
            if fsdp_mixed_precision == "bf16" and isinstance(module, torch.nn.LayerNorm):
                module.__class__ = Fp32LayerNorm
                patched_fp32_layernorm_names.append(name or "<root>")
            ignored.append(module)
            matched_names.append(name or "<root>")

        accelerator.state.fsdp_plugin.ignored_modules = ignored
        seen_param_ids = set()
        for module in ignored:
            for param in module.parameters():
                if not param.requires_grad or id(param) in seen_param_ids:
                    continue
                seen_param_ids.add(id(param))
                fsdp_replicated_params_to_sync.append(param)

        if is_main_process:
            preview = matched_names[:20]
            suffix = "" if len(matched_names) <= len(preview) else f", ... (+{len(matched_names) - len(preview)} more)"
            num_replicated_params = sum(param.numel() for param in fsdp_replicated_params_to_sync)
            logging.info(
                f"FSDP2: ignoring {len(ignored)} module(s); "
                f"structural_classes={sorted(ignored_class_names)}, "
                f"fp32_classes={sorted(fp32_module_classes)}, "
                f"name_suffixes={list(ignored_name_suffixes)}, "
                f"matches={preview}{suffix}, "
                f"replicated_trainable_params_to_sync={num_replicated_params:,}"
            )
            if patched_fp32_layernorm_names:
                logging.info(
                    "FSDP2: patched ignored LayerNorm modules for explicit fp32 compute: "
                    f"{patched_fp32_layernorm_names}"
                )

        if fsdp_replicated_params_to_sync:
            disable_optimizer_foreach(optimizer)
            if is_main_process:
                logging.info("FSDP2: disabled optimizer foreach/fused kernels for mixed Tensor+DTensor params")

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )

    if cfg.resume and cfg.fsdp.enable:
        step = load_fsdp2_checkpoint(
            accelerator=accelerator,
            checkpoint_dir=Path(cfg.checkpoint_path),
            policy=policy,
            optimizer=optimizer,
            replicated_params=fsdp_replicated_params_to_sync,
        )
        if is_main_process:
            logging.info(f"Loaded FSDP2 accelerator state from {cfg.checkpoint_path} at step {step}")

    if fsdp_replicated_params_to_sync:
        broadcast_replicated_parameters(fsdp_replicated_params_to_sync, accelerator)
        if is_main_process:
            logging.info("FSDP2: synchronized ignored trainable parameters from rank 0")

    dl_iter = cycle(dataloader)

    policy.train()

    if cfg.deepspeed.enable and getattr(cfg.policy, "dtype", None) == "bfloat16":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif cfg.fsdp.enable:
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = None

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size * cfg.grad_accum_steps,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info("Start FlashVLA training")
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )

    for _ in range(step, cfg.steps):
        step_compute_time = 0.0

        for micro_step in range(cfg.grad_accum_steps):
            start_time = time.perf_counter()
            batch = next(dl_iter)
            batch = preprocessor(batch)
            train_tracker.dataloading_s = time.perf_counter() - start_time

            do_step = micro_step == cfg.grad_accum_steps - 1

            compute_start = time.perf_counter()
            train_tracker, output_dict = update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                accelerator=accelerator,
                lr_scheduler=lr_scheduler if do_step else None,
                loss_scale=1.0 / cfg.grad_accum_steps,
                do_step=do_step,
                autocast_ctx=autocast_ctx,
                replicated_params_to_sync=fsdp_replicated_params_to_sync,
            )
            step_compute_time += time.perf_counter() - compute_start

        train_tracker.update_s = step_compute_time

        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()

                if output_dict:
                    for k, v in output_dict.items():
                        if k in wandb_log_dict:
                            wandb_log_dict[f"raw_{k}"] = v
                        else:
                            wandb_log_dict[k] = v

                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)

            if cfg.fsdp.enable:
                save_fsdp2_checkpoint(
                    accelerator=accelerator,
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=policy,
                    optimizer=optimizer,
                    replicated_params=fsdp_replicated_params_to_sync,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )

                if is_main_process:
                    update_last_checkpoint(checkpoint_dir)
                    if wandb_logger:
                        wandb_logger.log_policy(checkpoint_dir)
                    logging.info(f"FSDP2 policy checkpointed at step {step}")
                accelerator.wait_for_everyone()
            else:
                unwrapped_policy = accelerator.unwrap_model(policy)

                if is_main_process:
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        step=step,
                        cfg=cfg,
                        policy=unwrapped_policy,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                    )

                    update_last_checkpoint(checkpoint_dir)

                    if wandb_logger:
                        wandb_logger.log_policy(checkpoint_dir)
                    logging.info(f"Policy checkpointed at step {step}")

                accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()
        logging.info("End of FlashVLA training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    """CLI entry point."""
    train()


if __name__ == "__main__":
    main()
