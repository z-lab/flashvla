from __future__ import annotations

import importlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from accelerate import Accelerator


FSDP_PROCESS_GROUP_BACKEND = "cuda:nccl,cpu:gloo"


@dataclass(frozen=True)
class FSDPWrapReport:
    """Summary of the communication groups created for one policy."""

    compute_modules: tuple[str, ...]
    fp32_modules: tuple[str, ...]
    fp32_output_modules: tuple[str, ...]
    trainable_parameters: int


@dataclass(frozen=True)
class _NamedModule:
    name: str
    module: nn.Module


@dataclass(frozen=True)
class FSDPModulePlan:
    """Resolved module objects for a policy's FSDP2 communication groups."""

    compute_modules: tuple[_NamedModule, ...]
    fp32_modules: tuple[_NamedModule, ...]
    fp32_output_modules: tuple[_NamedModule, ...]


def build_fsdp_mixed_precision_policy(mixed_precision: str, reduce_dtype: str):
    """Build the policy used by the large FSDP communication groups."""
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
        # FlashVLA casts each compute boundary explicitly. Letting FSDP cast
        # whole argument trees would also quantize auxiliary fp32 values such
        # as training targets and PI0.5's adaRMS conditioning.
        cast_forward_inputs=False,
    )


def _get_policy_metadata(policy: nn.Module, name: str) -> tuple[str, ...]:
    value = getattr(policy, name, None)
    if value is None:
        raise TypeError(
            f"{type(policy).__name__} does not define {name}; "
            "this policy has no FSDP2 sharding plan."
        )
    if not isinstance(value, (tuple, list)) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{type(policy).__name__}.{name} must be a sequence of class/name strings")
    return tuple(dict.fromkeys(value))


def _is_at_or_below(name: str, root_name: str) -> bool:
    return name == root_name or name.startswith(f"{root_name}.")


def _outermost_modules(modules: list[_NamedModule]) -> list[_NamedModule]:
    selected: list[_NamedModule] = []
    for candidate in sorted(modules, key=lambda item: item.name.count(".")):
        if any(_is_at_or_below(candidate.name, root.name) for root in selected):
            continue
        selected.append(candidate)
    return selected


def build_fsdp_module_plan(policy: nn.Module, mixed_precision: str) -> FSDPModulePlan:
    """Resolve policy metadata to concrete, non-overlapping module objects."""
    if mixed_precision not in {"no", "bf16"}:
        raise ValueError(f"Unsupported FSDP mixed precision mode: {mixed_precision!r}")

    wrap_class_names = set(_get_policy_metadata(policy, "fsdp_wrap_class_names"))
    wrap_name_suffixes = _get_policy_metadata(policy, "fsdp_wrap_name_suffixes")
    fp32_class_names = set(_get_policy_metadata(policy, "fsdp_fp32_class_names"))
    fp32_name_suffixes = _get_policy_metadata(policy, "fsdp_fp32_name_suffixes")
    fp32_output_name_suffixes = _get_policy_metadata(policy, "fsdp_fp32_output_name_suffixes")

    named_modules = [
        _NamedModule(name=name, module=module)
        for name, module in policy.named_modules()
        if name
    ]
    module_class_name = {id(item.module): type(item.module).__name__ for item in named_modules}

    fp32_modules: list[_NamedModule] = []
    if mixed_precision == "bf16":
        fp32_candidates = [
            item
            for item in named_modules
            if module_class_name[id(item.module)] in fp32_class_names
            or item.name.endswith(fp32_name_suffixes)
        ]
        fp32_modules = _outermost_modules(fp32_candidates)

    fp32_roots = tuple(item.name for item in fp32_modules)
    class_compute_modules = [
        item
        for item in named_modules
        if module_class_name[id(item.module)] in wrap_class_names
        and not any(_is_at_or_below(item.name, root_name) for root_name in fp32_roots)
    ]
    class_compute_parameter_ids = [
        {id(parameter) for parameter in item.module.parameters()}
        for item in class_compute_modules
    ]
    name_compute_modules = []
    for item in named_modules:
        if not item.name.endswith(wrap_name_suffixes):
            continue
        if any(_is_at_or_below(item.name, root_name) for root_name in fp32_roots):
            continue
        parameter_ids = {id(parameter) for parameter in item.module.parameters()}
        # A tied lm_head may share its weight with an Embedding communication
        # group despite being a distinct module object. In that case the used
        # embedding owns the parameter and the dormant head must not re-wrap it.
        if any(parameter_ids & owner_ids for owner_ids in class_compute_parameter_ids):
            continue
        name_compute_modules.append(item)

    compute_modules = list(
        {id(item.module): item for item in [*class_compute_modules, *name_compute_modules]}.values()
    )
    compute_modules.sort(key=lambda item: item.name.count("."), reverse=True)

    missing_wrap_classes = wrap_class_names - set(module_class_name.values())
    if missing_wrap_classes:
        raise ValueError(
            f"FSDP2 plan for {type(policy).__name__} could not find wrap classes: "
            f"{sorted(missing_wrap_classes)}"
        )
    missing_wrap_suffixes = {
        suffix
        for suffix in wrap_name_suffixes
        if not any(item.name.endswith(suffix) for item in named_modules)
    }
    if missing_wrap_suffixes:
        raise ValueError(
            f"FSDP2 plan for {type(policy).__name__} could not find wrap suffixes: "
            f"{sorted(missing_wrap_suffixes)}"
        )

    missing_fp32_suffixes = {
        suffix
        for suffix in fp32_name_suffixes
        if not any(item.name.endswith(suffix) for item in named_modules)
    }
    if missing_fp32_suffixes:
        raise ValueError(
            f"FSDP2 plan for {type(policy).__name__} could not find fp32 suffixes: "
            f"{sorted(missing_fp32_suffixes)}"
        )
    matched_fp32_names = {item.name for item in fp32_modules}
    missing_fp32_output_suffixes = {
        suffix
        for suffix in fp32_output_name_suffixes
        if not any(name.endswith(suffix) for name in matched_fp32_names)
    }
    if mixed_precision == "bf16" and missing_fp32_output_suffixes:
        raise ValueError(
            f"FSDP2 plan for {type(policy).__name__} could not find fp32-output suffixes "
            f"among fp32 modules: {sorted(missing_fp32_output_suffixes)}"
        )

    fp32_output_modules = tuple(
        item for item in fp32_modules if item.name.endswith(fp32_output_name_suffixes)
    )
    return FSDPModulePlan(
        compute_modules=tuple(compute_modules),
        fp32_modules=tuple(fp32_modules),
        fp32_output_modules=fp32_output_modules,
    )


def _fsdp_mesh(accelerator: Accelerator):
    mesh = getattr(accelerator, "torch_device_mesh", None)
    if mesh is not None:
        dim_names = tuple(accelerator.parallelism_config.fsdp_dim_names)
        return mesh[dim_names]

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("FSDP2 policy preparation requires an initialized distributed process group")

    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh(
        accelerator.device.type,
        mesh_shape=(torch.distributed.get_world_size(),),
    )


def fully_shard_policy(
    policy: nn.Module,
    accelerator: Accelerator,
    *,
    mixed_precision: str,
    mixed_precision_policy,
    reshard_after_forward: bool,
    cpu_offload: bool,
) -> FSDPWrapReport:
    """Apply the policy's complete heterogeneous FSDP2 plan bottom-up.

    The policy supplies class/name metadata, so the trainer never needs to know
    whether it is preparing PI0 or PI0.5. Sensitive fp32 modules are sharded
    first, transformer-sized bf16 units second, and the policy root last.
    """
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        OffloadPolicy,
        fully_shard,
    )
    from torch.distributed.tensor import DTensor

    if isinstance(policy, FSDPModule):
        raise RuntimeError(f"{type(policy).__name__} is already an FSDP2 module")

    main_policy = mixed_precision_policy or MixedPrecisionPolicy()
    prepare_for_fsdp = getattr(policy, "prepare_for_fsdp", None)
    if not callable(prepare_for_fsdp):
        raise TypeError(
            f"{type(policy).__name__} must implement prepare_for_fsdp() before FSDP2 wrapping"
        )
    prepare_for_fsdp(compute_dtype=main_policy.param_dtype or torch.float32)

    plan = build_fsdp_module_plan(policy, mixed_precision)

    mesh = _fsdp_mesh(accelerator)
    common_kwargs = {
        "mesh": mesh,
        "offload_policy": CPUOffloadPolicy() if cpu_offload else OffloadPolicy(),
        "reshard_after_forward": reshard_after_forward,
    }
    fp32_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )
    fp32_output_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=True,
    )

    fp32_output_module_ids = {id(item.module) for item in plan.fp32_output_modules}
    for item in sorted(plan.fp32_modules, key=lambda entry: entry.name.count("."), reverse=True):
        preserve_fp32_output = id(item.module) in fp32_output_module_ids
        fully_shard(
            item.module,
            mp_policy=fp32_output_policy if preserve_fp32_output else fp32_policy,
            **common_kwargs,
        )

    for item in plan.compute_modules:
        fully_shard(item.module, mp_policy=main_policy, **common_kwargs)

    root_policy = MixedPrecisionPolicy(
        param_dtype=main_policy.param_dtype,
        reduce_dtype=main_policy.reduce_dtype,
        output_dtype=main_policy.output_dtype,
        # The policy input is a batch dictionary. Casting it at the root would
        # quantize images, states, and target actions before their owning
        # modules have a chance to select the correct compute dtype.
        cast_forward_inputs=False,
    )
    root_kwargs = {
        "mesh": mesh,
        "offload_policy": common_kwargs["offload_policy"],
        "mp_policy": root_policy,
        "reshard_after_forward": reshard_after_forward,
    }
    fully_shard(policy, **root_kwargs)

    unmanaged_trainable = [
        name
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad and not isinstance(parameter, DTensor)
    ]
    if unmanaged_trainable:
        raise RuntimeError(
            "FSDP2 left trainable parameters unmanaged: "
            f"{unmanaged_trainable[:20]}"
        )

    return FSDPWrapReport(
        compute_modules=tuple(item.name for item in plan.compute_modules),
        fp32_modules=tuple(item.name for item in plan.fp32_modules),
        fp32_output_modules=tuple(item.name for item in plan.fp32_output_modules),
        trainable_parameters=sum(
            parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
        ),
    )


def load_fsdp2_sharded_optimizer_preserve_missing_state(
    fsdp_plugin,
    accelerator: Accelerator,
    optimizer,
    model: nn.Module,
    input_dir: str,
    optimizer_index: int = 0,
    adapter_only: bool = False,
) -> None:
    """Load sharded optimizer state into a fresh optimizer without inventions.

    PyTorch builds the destination schema with a zero-gradient dummy optimizer
    step. A strict DCP load then fails when a checkpoint legitimately omits a
    parameter whose gradient has never materialized. We validate every state
    field and param-group field against checkpoint metadata, allow only wholly
    absent parameter states during DCP loading, then remove the dummy states
    for exactly those parameters.
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
    storage_reader = FileSystemReader(checkpoint_dir)
    metadata = storage_reader.read_metadata()
    planner_data = metadata.planner_data or {}
    checkpoint_state_fqns = {
        path[2]
        for path in planner_data.values()
        if isinstance(path, tuple)
        and len(path) >= 4
        and path[0] == "optimizer"
        and path[1] == "state"
        and isinstance(path[2], str)
    }

    base_optimizer = optimizer
    while hasattr(base_optimizer, "optimizer"):
        base_optimizer = base_optimizer.optimizer
    if base_optimizer.state:
        raise RuntimeError(
            "FSDP2 checkpoint loading requires a fresh optimizer with empty state"
        )

    options = _prepare_sd_options(fsdp_plugin)
    optimizer_state = get_optimizer_state_dict(model, optimizer, options=options)
    template_state = optimizer_state.get("state", {})
    optimizer_param_groups = optimizer_state.get("param_groups", [])
    expected_group_params = [
        tuple(param_group["params"])
        for param_group in optimizer_param_groups
    ]
    optimizer_fqns = {
        fqn
        for param_group in optimizer_param_groups
        for fqn in param_group["params"]
    }
    missing_state_fqns = optimizer_fqns - checkpoint_state_fqns
    unexpected_fqns = checkpoint_state_fqns - optimizer_fqns
    if unexpected_fqns:
        raise KeyError(
            "Optimizer checkpoint contains parameters absent from the current model: "
            f"{sorted(unexpected_fqns)[:20]}"
        )
    missing_templates = checkpoint_state_fqns - set(template_state)
    if missing_templates:
        raise KeyError(
            "Could not build optimizer templates for checkpoint parameters: "
            f"{sorted(missing_templates)[:20]}"
        )

    checkpoint_state_fields: dict[str, set[str]] = {}
    checkpoint_param_group_fields: set[tuple[int, str]] = set()
    for path in planner_data.values():
        if not isinstance(path, tuple) or len(path) < 4 or path[0] != "optimizer":
            continue
        if path[1] == "state" and isinstance(path[2], str) and isinstance(path[3], str):
            checkpoint_state_fields.setdefault(path[2], set()).add(path[3])
        elif path[1] == "param_groups" and isinstance(path[2], int) and isinstance(path[3], str):
            checkpoint_param_group_fields.add((path[2], path[3]))

    malformed_state = {
        fqn: {
            "checkpoint": sorted(checkpoint_state_fields.get(fqn, set())),
            "expected": sorted(template_state[fqn]),
        }
        for fqn in checkpoint_state_fqns
        if checkpoint_state_fields.get(fqn, set()) != set(template_state[fqn])
    }
    if malformed_state:
        raise KeyError(
            "Optimizer checkpoint has incomplete or incompatible state fields: "
            f"{dict(list(malformed_state.items())[:10])}"
        )

    expected_param_group_fields = {
        (index, key)
        for index, param_group in enumerate(optimizer_state.get("param_groups", []))
        for key in param_group
    }
    if checkpoint_param_group_fields != expected_param_group_fields:
        raise KeyError(
            "Optimizer checkpoint param-group schema mismatch: "
            f"missing={sorted(expected_param_group_fields - checkpoint_param_group_fields)}, "
            f"unexpected={sorted(checkpoint_param_group_fields - expected_param_group_fields)}"
        )

    missing_state_params = []
    if len(optimizer_param_groups) != len(base_optimizer.param_groups):
        raise ValueError("Optimizer param-group count changed while preparing checkpoint load")
    for state_group, live_group in zip(
        optimizer_param_groups,
        base_optimizer.param_groups,
        strict=True,
    ):
        fqns = state_group["params"]
        params = live_group["params"]
        if len(fqns) != len(params):
            raise ValueError("Optimizer parameter order changed while preparing checkpoint load")
        missing_state_params.extend(
            parameter
            for fqn, parameter in zip(fqns, params, strict=True)
            if fqn in missing_state_fqns
        )

    state_dict = {"optimizer": optimizer_state}
    load(
        state_dict,
        checkpoint_id=checkpoint_dir,
        storage_reader=storage_reader,
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )
    loaded_param_groups = state_dict["optimizer"].get("param_groups", [])
    loaded_group_params = [
        tuple(param_group["params"])
        for param_group in loaded_param_groups
    ]
    if loaded_group_params != expected_group_params:
        raise ValueError(
            "Optimizer checkpoint parameter groups do not match the current optimizer"
        )
    original_requires_grad = [parameter.requires_grad for parameter in missing_state_params]
    try:
        # Torch's optimizer-state splitter indexes state for every parameter
        # whose requires_grad flag is true. Temporarily mark whole-state-missing
        # parameters dormant so it can represent the native lazy Adam state.
        for parameter in missing_state_params:
            parameter.requires_grad_(False)
        set_optimizer_state_dict(
            model,
            optimizer,
            state_dict["optimizer"],
            options=options,
        )
    finally:
        for parameter, requires_grad in zip(
            missing_state_params,
            original_requires_grad,
            strict=True,
        ):
            parameter.requires_grad_(requires_grad)
    for parameter in missing_state_params:
        base_optimizer.state.pop(parameter, None)


@contextmanager
def patch_accelerate_fsdp_optimizer_loader():
    """Scope the missing-state-safe optimizer loader to one resume."""
    accelerator_module = importlib.import_module("accelerate.accelerator")
    original_loader = accelerator_module.load_fsdp_optimizer
    accelerator_module.load_fsdp_optimizer = load_fsdp2_sharded_optimizer_preserve_missing_state
    try:
        yield
    finally:
        accelerator_module.load_fsdp_optimizer = original_loader
