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
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
import accelerate as accelerate_lib
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

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
from flashvla.distributed.fsdp import (
    build_fsdp_mixed_precision_policy,
    fully_shard_policy,
    patch_accelerate_fsdp_optimizer_loader,
)
from flashvla.policies.factory import make_policy, make_pre_post_processors


ACCELERATOR_STATE_DIR = "accelerator_state"


def count_parameters(model: torch.nn.Module, only_trainable: bool = False) -> int:
    """Count model parameters."""
    return sum(
        p.numel()
        for p in model.parameters()
        if not only_trainable or p.requires_grad
    )


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
        if grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
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


def save_fsdp2_checkpoint(
    *,
    accelerator: Accelerator,
    checkpoint_dir: Path,
    step: int,
    cfg: FlashVLATrainConfig,
    policy: PreTrainedPolicy,
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

    if accelerator.is_main_process:
        training_state_dir.mkdir(parents=True, exist_ok=True)
        save_training_step(step, training_state_dir)

    accelerator.wait_for_everyone()
    accelerator_state_dir = checkpoint_dir / ACCELERATOR_STATE_DIR
    accelerator.save_state(str(accelerator_state_dir), safe_serialization=True)
    accelerator.wait_for_everyone()

    # Export the gathered inference checkpoint only after the exact sharded
    # resume state is durable. Gathering a multi-billion-parameter fp32 model
    # has a much higher peak-memory requirement than the sharded save.
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
    accelerator.wait_for_everyone()


def load_fsdp2_checkpoint(
    *,
    accelerator: Accelerator,
    checkpoint_dir: Path,
) -> int:
    """Load FSDP2 training state after `accelerator.prepare()`."""
    accelerator_state_dir = checkpoint_dir / ACCELERATOR_STATE_DIR
    if not accelerator_state_dir.is_dir():
        raise NotADirectoryError(
            f"FSDP2 resume requires {accelerator_state_dir}. "
            "Older LeRobot-style checkpoints do not contain sharded FSDP2 optimizer state."
        )
    step = load_training_step(checkpoint_dir / TRAINING_STATE_DIR)
    with patch_accelerate_fsdp_optimizer_loader():
        accelerator.load_state(str(accelerator_state_dir))
    return step


def auto_resume(cfg: FlashVLATrainConfig) -> FlashVLATrainConfig:
    """Automatically resume training from the last checkpoint if available."""
    if cfg.output_dir is None:
        return cfg

    if cfg.resume:
        config_path_arg = parser.parse_arg("config_path")
        config_path = Path(config_path_arg) if config_path_arg else None
        if (
            config_path is not None
            and config_path.is_file()
            and config_path.parent.name == PRETRAINED_MODEL_DIR
            and config_path.parent.parent.is_dir()
        ):
            return cfg
        raise ValueError(
            "Explicit --resume=true requires config_path to point to a checkpoint's "
            "pretrained_model/train_config.json. Omit --resume to auto-resume from "
            "output_dir/checkpoints/last."
        )

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
            if not train_config_json.is_file():
                raise FileNotFoundError(
                    f"Auto-resume checkpoint is missing {train_config_json}"
                )

            cli_overrides = []
            cli_args = iter(sys.argv[1:])
            for arg in cli_args:
                if arg in {"--config_path", "--policy.path"}:
                    next(cli_args, None)
                elif not arg.startswith(("--config_path=", "--policy.path=")):
                    cli_overrides.append(arg)
            cfg = FlashVLATrainConfig.from_pretrained(
                train_config_json,
                cli_args=cli_overrides,
            )
            cfg.resume = True
            new_argv = [
                sys.argv[0],
                *cli_overrides,
                f"--config_path={train_config_json}",
            ]
            sys.argv = new_argv

            print(f"Auto-resume enabled. Using config_path={train_config_json} and resume=True.")
        else:
            checkpoint_entries = sorted(path.name for path in checkpoints_dir.iterdir())
            raise RuntimeError(
                f"Existing output directory {output_dir_path} has checkpoints but no valid "
                f"'last' link. Preserving it for recovery; entries={checkpoint_entries}."
            )

    elif output_dir_path.is_dir():
        raise FileExistsError(
            f"Existing output directory {output_dir_path} has no resumable checkpoint; "
            "refusing to delete it. Choose another output_dir or clean it explicitly."
        )
    return cfg


@parser.wrap()
def train(cfg: FlashVLATrainConfig, accelerator: Accelerator | None = None):
    """Main training function for FlashVLA."""
    cfg = auto_resume(cfg)
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
                auto_wrap_policy="no_wrap",
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
                "sharding_plan=policy-owned, trainable_ignored_params=0"
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

    if cfg.fsdp.enable:
        fsdp_report = fully_shard_policy(
            policy,
            accelerator,
            mixed_precision=fsdp_mixed_precision,
            mixed_precision_policy=fsdp_precision_policy,
            reshard_after_forward=cfg.fsdp.reshard_after_forward,
            cpu_offload=cfg.fsdp.cpu_offload,
        )
        if is_main_process:
            logging.info(
                "FSDP2 policy plan applied: "
                f"compute_units={len(fsdp_report.compute_modules)}, "
                f"fp32_units={len(fsdp_report.fp32_modules)}, "
                f"fp32_output_units={len(fsdp_report.fp32_output_modules)}, "
                f"managed_trainable_params={fsdp_report.trainable_parameters:,}"
            )
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

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )

    if cfg.resume and cfg.fsdp.enable:
        step = load_fsdp2_checkpoint(
            accelerator=accelerator,
            checkpoint_dir=Path(cfg.checkpoint_path),
        )
        if is_main_process:
            logging.info(f"Loaded FSDP2 accelerator state from {cfg.checkpoint_path} at step {step}")

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
