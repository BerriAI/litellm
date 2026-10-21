import argparse
import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from poc import run_pilot
from poc.profiles import (
    PUBLIC_DEV_TENANT,
    HarnessConfig,
    ProfileVersions,
    compile_map_from_profiling_runs,
    harness_config_sha256,
    manifest_selection,
    validate_manifest_attempt,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER: Final = ("haiku-4.5", "sonnet-5", "opus-5")
PHASE_BUDGET_USD: Final = 180.0
BATCH_SIZE: Final = 3
WORKERS: Final = 3
MIN_FREE_BYTES: Final = 60 * 1024**3
MANIFEST_SHA256: Final = "15720dee12d40b62780837ed12a403f3951e9c5bb1ed18ec0d5d5d1f4776bff4"


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    root_dir: Path
    manifest_path: Path
    manifest_sha256: str
    tenant_id: str = PUBLIC_DEV_TENANT

    @property
    def profile_dir(self) -> Path:
        return self.root_dir / "profiling"

    @property
    def runs_path(self) -> Path:
        return self.profile_dir / "profiling_runs.jsonl"


@dataclass(frozen=True, slots=True)
class ProfileTask:
    index: int
    task_id: str
    category: str
    model_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProfileBatch:
    index: int
    tasks: tuple[ProfileTask, ...]


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def model_order_for_task(index: int) -> tuple[str, ...]:
    offset: Final = index % len(MODEL_ORDER)
    return MODEL_ORDER[offset:] + MODEL_ORDER[:offset]


def _category_tasks(
    category: str,
    manifest_category: Mapping[str, object],
    index_offset: int,
) -> tuple[ProfileTask, ...]:
    selected_ids: Final = manifest_category.get("selected_ids")
    if not isinstance(selected_ids, list) or not all(isinstance(task_id, str) for task_id in selected_ids):
        raise ValueError(f"{category} selected_ids must be strings")
    return tuple(
        ProfileTask(
            index=index_offset + index,
            task_id=task_id,
            category=category,
            model_order=model_order_for_task(index_offset + index),
        )
        for index, task_id in enumerate(selected_ids)
    )


def profile_batches(config: ProfileConfig, max_batches: int | None = None) -> tuple[ProfileBatch, ...]:
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    selected_ids, _ = manifest_selection(config.manifest_path, config.manifest_sha256)
    manifest: Final = _mapping(json.loads(config.manifest_path.read_text()), "manifest")
    categories: Final = _mapping(manifest.get("categories"), "manifest categories")
    repository_tasks: Final = _category_tasks(
        "repository_debugging",
        _mapping(categories.get("repository_debugging"), "repository_debugging"),
        0,
    )
    feature_tasks: Final = _category_tasks(
        "feature_implementation",
        _mapping(categories.get("feature_implementation"), "feature_implementation"),
        len(repository_tasks),
    )
    tasks: Final = repository_tasks + feature_tasks
    if tuple(task.task_id for task in tasks) != selected_ids:
        raise ValueError("profile categories differ from manifest selection")
    batches: Final = tuple(
        ProfileBatch(index=index, tasks=tasks[start : start + BATCH_SIZE])
        for index, start in enumerate(range(0, len(tasks), BATCH_SIZE), start=1)
    )
    return batches if max_batches is None else batches[:max_batches]


def require_disk_for_pull(free_bytes: int) -> None:
    if free_bytes < MIN_FREE_BYTES:
        raise OSError("abort profiling batch before pull: less than 60 GiB free")


def _single_result_value(result: Mapping[str, object], field: str) -> str | None:
    value: Final = result.get(field, ())
    values: Final = (value,) if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)) or not all(isinstance(item, str) for item in values):
        raise ValueError(f"{field} must contain strings")
    if len(values) > 1:
        raise ValueError(f"{field} must contain at most one value")
    return values[0] if values else None


def profile_guard(
    config: ProfileConfig,
    free_bytes: Callable[[], int] = lambda: shutil.disk_usage("/").free,
    committed_spend: Callable[[], float] | None = None,
) -> run_pilot.DispatchGuard:
    config.profile_dir.mkdir(parents=True, exist_ok=True)
    spend_reader: Final = (
        committed_spend
        if committed_spend is not None
        else lambda: run_pilot.committed_spend_usd(config.runs_path, config.profile_dir / "costs_aux.jsonl")
    )
    return run_pilot.DispatchGuard(
        config.profile_dir / "dispatch_stops.jsonl",
        budget=PHASE_BUDGET_USD,
        free_bytes=free_bytes,
        committed_spend=spend_reader,
    )


def record_task_result(
    config: ProfileConfig,
    task: ProfileTask,
    result: Mapping[str, object],
    versions: ProfileVersions,
) -> None:
    task_id: Final = result.get("task_id")
    manifest_sha256: Final = result.get("manifest_sha256", config.manifest_sha256)
    if task_id != task.task_id or not isinstance(manifest_sha256, str):
        raise ValueError("task result must match the scheduled task and include manifest_sha256")
    selected_ids, actual_sha256 = manifest_selection(config.manifest_path, config.manifest_sha256)
    validate_manifest_attempt(task.task_id, manifest_sha256, selected_ids, actual_sha256)
    model_alias: Final = result.get("model")
    if not isinstance(model_alias, str) or model_alias not in task.model_order:
        raise ValueError("task result model must match the scheduled model order")
    returned_model: Final = _single_result_value(result, "returned_models") or "unreturned"
    deployment_id: Final = _single_result_value(result, "deployment_ids")
    error_class: Final = result.get("error_class")
    is_infra: Final = error_class in ("infra", "accounting")
    success: Final = result.get("success")
    if not is_infra and not isinstance(success, bool):
        raise ValueError("graded task results must include a boolean success")
    solve_latency: Final = result.get("solve_latency_s", 0.0)
    eval_latency: Final = result.get("eval_latency_s", 0.0)
    latency: Final = float(solve_latency) + float(eval_latency)
    harness_sha256: Final = hashlib.sha256(Path(run_pilot.__file__).read_bytes()).hexdigest()
    run_pilot.append_jsonl(
        config.runs_path,
        {
            **result,
            "task_id": task.task_id,
            "tenant_id": config.tenant_id,
            "category": task.category,
            "model_alias": model_alias,
            "returned_model_version": returned_model,
            "deployment_id": deployment_id,
            "harness_version": versions.harness_version,
            "budget_version": versions.budget_version,
            "harness_config_hash": versions.harness_config_hash,
            "is_infra": is_infra,
            "success": None if is_infra else success,
            "cost_usd": result.get("cost_usd", 0.0),
            "latency_s": latency,
            "primary_label": task.category,
            "secondary_label": None,
            "manifest_sha256": manifest_sha256,
            "harness_sha256": harness_sha256,
            "model_order": task.model_order,
            "returned_model": returned_model,
            "x_litellm_model_id": deployment_id,
            "error_class": error_class,
            "raw_result": dict(result),
        },
    )


def _profile_versions(root_dir: Path) -> ProfileVersions:
    model_manifest: Final = _mapping(
        json.loads((root_dir / "poc" / "pilot_models.json").read_text()),
        "pilot model manifest",
    )
    solver_data: Final = _mapping(model_manifest.get("solvers"), "pilot solvers")
    model_versions: Final = tuple(
        (alias, str(_mapping(solver_data.get(alias), alias)["gateway_model"])) for alias in MODEL_ORDER
    )
    model_deployments: Final = tuple(
        (alias, str(_mapping(solver_data.get(alias), alias)["deployment_id"])) for alias in MODEL_ORDER
    )
    run_pilot_sha256: Final = hashlib.sha256((root_dir / "poc" / "run_pilot.py").read_bytes()).hexdigest()
    config: Final = HarnessConfig(
        harness_version="run_pilot_attempt_v1",
        budget_version="budget_1usd_50_600_v1",
        per_run_budget_usd=1.0,
        step_limit=50,
        wall_timeout_s=600,
        mini_swe_agent_version=run_pilot._package_version("mini-swe-agent"),
        guard_sha256=run_pilot_sha256,
    )
    return ProfileVersions(
        harness_version=config.harness_version,
        budget_version=config.budget_version,
        harness_config_hash=harness_config_sha256(config),
        model_versions=model_versions,
        model_deployments=model_deployments,
    )


def _write_checkpoint(config: ProfileConfig, batch: ProfileBatch, versions: ProfileVersions) -> None:
    config.profile_dir.mkdir(parents=True, exist_ok=True)
    config.runs_path.touch(exist_ok=True)
    capability_map: Final = compile_map_from_profiling_runs(
        config.runs_path,
        config.manifest_path,
        config.manifest_sha256,
        config.tenant_id,
        versions,
    )
    serialized_map: Final = json.dumps(capability_map, indent=2, sort_keys=True) + "\n"
    (config.profile_dir / f"capability_map_batch_{batch.index:03d}.json").write_text(serialized_map)
    (config.profile_dir / "capability_map_intermediate.json").write_text(serialized_map)
    progress: Final = {
        "dry_run": True,
        "manifest_sha256": config.manifest_sha256,
        "tenant_id": config.tenant_id,
        "completed_batches": batch.index,
        "next_batch": batch.index + 1,
        "tasks": [
            {
                "task_id": task.task_id,
                "category": task.category,
                "model_order": task.model_order,
                "status": "planned",
                "preflight_status": "not_run_dry_run",
            }
            for task in batch.tasks
        ],
    }
    (config.profile_dir / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")


def _completed_batches(config: ProfileConfig) -> int:
    path: Final = config.profile_dir / "progress.json"
    if not path.exists():
        return 0
    progress: Final = _mapping(json.loads(path.read_text()), "progress")
    if progress.get("manifest_sha256") != config.manifest_sha256:
        raise ValueError("progress manifest sha256 mismatch")
    if progress.get("tenant_id") != config.tenant_id:
        raise ValueError("progress tenant mismatch")
    completed: Final = progress.get("completed_batches")
    if not isinstance(completed, int) or isinstance(completed, bool) or completed < 0:
        raise ValueError("progress completed_batches must be a non-negative integer")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and checkpoint profiling batches without launching solves")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    parser.add_argument("--tenant-id", default=PUBLIC_DEV_TENANT)
    parser.add_argument("--max-batches", type=int)
    arguments = parser.parse_args()
    default_manifest: Final = arguments.root / "profiling" / "profiling_manifest.json"
    manifest_path: Final = (
        arguments.manifest
        if arguments.manifest is not None
        else default_manifest
        if default_manifest.exists()
        else arguments.root / "profiling_manifest.json"
    )
    config: Final = ProfileConfig(arguments.root, manifest_path, arguments.manifest_sha256, arguments.tenant_id)
    completed_batches: Final = _completed_batches(config)
    pending_batches: Final = profile_batches(config)[completed_batches:]
    batches: Final = pending_batches if arguments.max_batches is None else pending_batches[: arguments.max_batches]
    if arguments.max_batches is not None and arguments.max_batches < 1:
        raise ValueError("max_batches must be positive")
    versions: Final = _profile_versions(config.root_dir)
    for batch in batches:
        _write_checkpoint(config, batch, versions)
    print(
        json.dumps(
            {
                "dry_run": True,
                "batches": len(batches),
                "planned_runs": sum(len(batch.tasks) * len(MODEL_ORDER) for batch in batches),
                "workers": WORKERS,
            }
        )
    )


if __name__ == "__main__":
    main()
