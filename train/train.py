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
from lerobot.utils.train_utils import get_step_checkpoint_dir, load_training_state, save_checkpoint, update_last_checkpoint
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from flashvla.configs.train_config import FlashVLATrainConfig
from flashvla.datasets.task_filter import get_episode_indices_for_suite
from flashvla.datasets.flashvla_dataset import (
    FlashVLADataset,
    MultiFlashVLADataset,
    flashvla_collate_fn,
    make_robotwin_multitask_dataset,
    make_multi_root_flashvla_dataset,
)
from flashvla.policies.factory import make_policy, make_pre_post_processors


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
        if not mt_cfg.root:
            raise ValueError("cfg.robotwin_multitask.enable=True requires robotwin_multitask.root to be set")

        # Resolve delta_timestamps against the first discovered subset's metadata.
        from pathlib import Path as _Path
        root_path = _Path(mt_cfg.root).expanduser()
        if mt_cfg.tasks:
            probe_task = mt_cfg.tasks[0]
        else:
            probe_task = next(
                p.name for p in sorted(root_path.iterdir())
                if p.is_dir() and (p / mt_cfg.config_subdir / "meta" / "info.json").is_file()
            )
        probe_root = root_path / probe_task / mt_cfg.config_subdir
        ds_meta = LeRobotDatasetMetadata(
            repo_id=f"robotwin/{probe_task}_{mt_cfg.config_subdir}",
            root=probe_root,
            revision=None,
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)

        dataset = make_robotwin_multitask_dataset(
            root=mt_cfg.root,
            config_subdir=mt_cfg.config_subdir,
            tasks=mt_cfg.tasks,
            num_buffer_slots=cfg.policy.num_buffer_slots,
            chunk_size=cfg.policy.chunk_size,
            use_action_prefix=getattr(cfg.policy, "use_action_prefix", False),
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )

        if cfg.dataset.use_imagenet_stats:
            for key in dataset.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

        return dataset

    # Multi-root path: mix different settings of the same task (e.g. clean + random).
    # Triggered when cfg.extra_dataset_roots is non-empty.
    extra_roots = getattr(cfg, "extra_dataset_roots", None) or []
    if extra_roots:
        all_roots = [cfg.dataset.root] + list(extra_roots)
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        dataset = make_multi_root_flashvla_dataset(
            repo_id=cfg.dataset.repo_id,
            roots=all_roots,
            num_buffer_slots=cfg.policy.num_buffer_slots,
            chunk_size=cfg.policy.chunk_size,
            use_action_prefix=getattr(cfg.policy, "use_action_prefix", False),
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        if cfg.dataset.use_imagenet_stats:
            for key in dataset.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
        return dataset

    # Single-dataset path (unchanged)
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
        use_action_prefix=getattr(cfg.policy, "use_action_prefix", False),
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
        loss, output_dict = policy.forward(batch)
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

    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        kwargs_handlers = []
        ds_plugin = None

        if cfg.deepspeed.enable:
            # DeepSpeed ZeRO: bf16/fp16 disabled so DeepSpeed does NOT touch
            # parameter dtypes. The model keeps its mixed precision (most params
            # bf16, sensitive params fp32). bf16 compute is handled by torch.autocast.
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
        else:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            kwargs_handlers.append(ddp_kwargs)

        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=kwargs_handlers,
            deepspeed_plugin=ds_plugin,
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

    # === Create Dataset ===
    make_dataset_fn = make_flashvla_dataset

    if is_main_process:
        logging.info(f"Creating dataset for {cfg.policy.type}")
        dataset = make_dataset_fn(cfg)

    accelerator.wait_for_everyone()

    if not is_main_process:
        dataset = make_dataset_fn(cfg)

    # === Create Policy ===
    if is_main_process:
        logging.info("Creating policy")

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    # Freeze VLM if configured.
    # Policy modules use different attribute paths:
    #   pi0/pi05:        policy.model.vlm.model.vision_tower (SigLIP)
    #   smolvla:         policy.model.vlm_with_expert.vlm.model.vision_model
    #                    (smolvla also self-handles via train_expert_only/
    #                     freeze_vision_encoder inside SmolVLMWithExpertModel,
    #                     so the call below is idempotent for that policy.)
    if getattr(cfg.policy, "freeze_vlm", False):
        if hasattr(policy.model, "vlm"):
            vlm_module = policy.model.vlm
        elif hasattr(policy.model, "vlm_with_expert"):
            vlm_module = policy.model.vlm_with_expert.vlm
        else:
            raise AttributeError("Cannot find VLM module to freeze")
        for param in vlm_module.parameters():
            param.requires_grad = False
        if is_main_process:
            logging.info("VLM backbone frozen")
    elif getattr(cfg.policy, "freeze_vision_encoder", False):
        # Freeze only the vision tower, keep language backbone trainable
        if hasattr(policy.model, "vlm") and hasattr(policy.model.vlm, "model") and hasattr(policy.model.vlm.model, "vision_tower"):
            vision_tower = policy.model.vlm.model.vision_tower
        elif hasattr(policy.model, "vlm_with_expert"):
            # smolvla: SmolVLM2 vision tower lives at vlm.model.vision_model.
            # Note: smolvla also freezes this inside its own set_requires_grad
            # when freeze_vision_encoder=True, so this is a redundant safety net.
            vision_tower = policy.model.vlm_with_expert.vlm.model.vision_model
        else:
            raise AttributeError("Cannot find vision tower to freeze")
        for param in vision_tower.parameters():
            param.requires_grad = False
        if is_main_process:
            n_frozen = sum(p.numel() for p in vision_tower.parameters())
            logging.info(f"Vision encoder frozen ({n_frozen:,} params)")

    accelerator.wait_for_everyone()

    # === Create Preprocessor and Postprocessor ===
    if is_main_process:
        logging.info("Creating preprocessor and postprocessor")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=dataset.meta.stats,
    )

    # === Create Optimizer and Scheduler ===
    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Optional: per-layer weight decay on suffix_embedder.time_mlp_{in,out} to
    # bound σ_max growth at noise floor (alternative to RMSNorm). Splits the
    # default param_group into two so AdamW applies a separate WD just to those
    # ~2M params; non-time_mlp params keep the yaml's weight_decay (e.g. 1e-10).
    time_mlp_wd = float(getattr(cfg.policy, "time_mlp_weight_decay", 0.0) or 0.0)
    if time_mlp_wd > 0.0:
        underlying = getattr(policy, "module", policy)  # unwrap DDP/FSDP
        time_mlp_params = []
        for n, p in underlying.named_parameters():
            if "suffix_embedder.time_mlp_in" in n or "suffix_embedder.time_mlp_out" in n:
                time_mlp_params.append(p)
        time_mlp_ids = {id(p) for p in time_mlp_params}

        new_groups = []
        # Keep all original groups but strip out time_mlp params (they keep
        # the yaml weight_decay etc.).
        for g in optimizer.param_groups:
            other = [p for p in g["params"] if id(p) not in time_mlp_ids]
            base = {k: v for k, v in g.items() if k != "params"}
            new_groups.append({**base, "params": other})
        # Append a dedicated group for time_mlp params with override WD; inherit
        # other settings (lr, betas, eps, ...) from the first group.
        base = {k: v for k, v in optimizer.param_groups[0].items()
                if k != "params" and k != "weight_decay"}
        new_groups.append({**base, "params": time_mlp_params, "weight_decay": time_mlp_wd})
        optimizer.param_groups = new_groups

        if is_main_process:
            logging.info(
                f"Added separate weight_decay={time_mlp_wd} on "
                f"{len(time_mlp_params)} suffix_embedder.time_mlp params"
            )

    step = 0

    # === Handle Checkpoint Resumption ===
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler
        )

    # === Log Training Configuration ===
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

    # === Create DataLoader ===
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

    # === Prepare for Distributed Training ===
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    # When DeepSpeed runs with bf16/fp16 disabled (to preserve mixed-precision
    # params), accelerator.autocast() is a no-op. Use torch.autocast explicitly.
    if cfg.deepspeed.enable and cfg.policy.dtype == "bfloat16":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = None  # fall back to accelerator.autocast() in update_policy

    # === Setup Metrics Tracking ===
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

    # === Main Training Loop ===
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

        # === Logging ===
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

        # === Checkpointing ===
        if cfg.save_checkpoint and is_saving_step:
            # unwrap_model must be called on ALL ranks so that DeepSpeed can
            # gather the full state_dict collectively (it's a NCCL all-gather).
            unwrapped_policy = accelerator.unwrap_model(policy)

            if is_main_process:
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
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

    # === Training Complete ===
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
