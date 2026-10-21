import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Final

from datasets import load_dataset

from poc import run_pilot
from poc.prepare import DATASET_NAME, DATASET_REVISION, DATASET_SPLIT, agent_rows
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
EVAL_DATASET_REVISION: Final = "c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2"
DEFAULT_OVERLAYS: Final = MappingProxyType(
    {
        "pvlib": (
            'RUN /opt/miniconda3/envs/testbed/bin/pip install --no-cache-dir "numpy==1.26.4" '
            "&& /opt/miniconda3/envs/testbed/bin/pip check"
        ),
        "pydicom": (
            "ENV LANG=C.UTF-8 LC_ALL=C.UTF-8\n"
            "RUN /opt/miniconda3/envs/testbed/bin/pip check"
        ),
    }
)
OVERLAY_SOURCE_FILES: Final = (
    "Dockerfile.pvlib_1776_pvlib-python-1165",
    "Dockerfile.pvlib_1776_pvlib-python-1191",
    "Dockerfile.pvlib_1776_pvlib-python-1395",
    "Dockerfile.pydicom_1776_pydicom-1031",
    "Dockerfile.pydicom_1776_pydicom-793",
)
NOOP_PATCH: Final = (
    "diff --git a/routing_poc_noop.txt b/routing_poc_noop.txt\n"
    "new file mode 100644\n"
    "index 0000000..d0951a7\n"
    "--- /dev/null\n"
    "+++ b/routing_poc_noop.txt\n"
    "@@ -0,0 +1 @@\n"
    "+noop\n"
)


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    root_dir: Path
    manifest_path: Path
    manifest_sha256: str
    tenant_id: str = PUBLIC_DEV_TENANT
    overlays: Mapping[str, str] = field(default_factory=lambda: DEFAULT_OVERLAYS)

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
    image_ref: str | None = None
    image_digest: str | None = None
    agent_image_ref: str | None = None
    grader_image_ref: str | None = None
    upstream_image_id: str | None = None
    overlay_image_id: str | None = None
    final_image_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProfileBatch:
    index: int
    tasks: tuple[ProfileTask, ...]


@dataclass(frozen=True, slots=True)
class AttemptPaths:
    pilot: run_pilot.AttemptPaths
    task_rows: Mapping[str, dict[str, object]]
    gold_patches: Mapping[str, str]
    solver_configs: Mapping[str, dict[str, object]]


@dataclass(frozen=True, slots=True)
class RuntimeOps:
    disk_stats: Callable[[], Mapping[str, int]]
    inspect_image: Callable[[str], str | None]
    pull_image: Callable[[str], None]
    tag_image: Callable[[str, str], None]
    build_image: Callable[[str, str], None]
    evaluate: Callable[[str, str, str, run_pilot.AttemptPaths], tuple[str, float]]
    remove_image: Callable[[str], bool]
    image_exists: Callable[[str], bool]
    attempt: Callable[
        [str, str, int, dict[str, object], dict[str, object], object, object, dict[str, str], str,
         run_pilot.AttemptPaths],
        dict[str, object],
    ]


@dataclass(frozen=True, slots=True)
class TaskImage:
    task_id: str
    digest_ref: str
    grader_ref: str
    agent_ref: str
    upstream_digest: str
    upstream_image_id: str
    overlay_image_id: str | None
    final_image_id: str
    owned_refs: tuple[str, ...]
    restore_grader: bool


@dataclass(frozen=True, slots=True)
class BatchImagePlan:
    prepared: tuple[TaskImage, ...]
    owned_refs: tuple[str, ...]


class TransientInfraError(RuntimeError):
    def __init__(self, message: str, plan: BatchImagePlan):
        super().__init__(message)
        self.plan = plan


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _model_order(index: int) -> tuple[str, ...]:
    offset: Final = index % len(MODEL_ORDER)
    return MODEL_ORDER[offset:] + MODEL_ORDER[:offset]


model_order_for_task = _model_order


def _grader_image_ref(image_ref: str) -> str:
    return f"{image_ref.split('@', 1)[0]}:latest"


def _overlay_key(task_id: str) -> str:
    return task_id.split("__", 1)[0]


def _overlay_image_ref(task_id: str) -> str:
    return f"routing_poc/profiling-overlay:{task_id}"


def _overlay_snippet(config: ProfileConfig, task: ProfileTask) -> str | None:
    return config.overlays.get(_overlay_key(task.task_id))


def _agent_image_ref(config: ProfileConfig, task: ProfileTask) -> str | None:
    if task.image_ref is None:
        return None
    return _overlay_image_ref(task.task_id) if _overlay_snippet(config, task) is not None else _grader_image_ref(task.image_ref)


def _overlay_dockerfile(image_ref: str, snippet: str) -> str:
    return f"FROM {image_ref}\n{snippet.rstrip()}\n"


class MixedVersionAttemptError(ValueError):
    def __init__(self, task_id: str):
        super().__init__("mixed_version attempt quarantined; stopping profiling run")
        self.task_id = task_id


def profile_batches(config: ProfileConfig, max_batches: int | None = None) -> tuple[ProfileBatch, ...]:
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    selected_ids, _ = manifest_selection(config.manifest_path, config.manifest_sha256)
    manifest: Final = _mapping(json.loads(config.manifest_path.read_text()), "manifest")
    categories: Final = _mapping(manifest.get("categories"), "manifest categories")
    candidate_rows: Final = manifest.get("candidates", ())
    candidate_by_id: Final = {
        row["task_id"]: row
        for row in candidate_rows
        if isinstance(row, Mapping) and isinstance(row.get("task_id"), str)
    }
    tasks: Final = tuple(
        ProfileTask(
            index=index,
            task_id=task_id,
            category=category,
            model_order=_model_order(index),
            image_ref=(
                f"{candidate_by_id[task_id]['image']}@{candidate_by_id[task_id]['image_digest']}"
                if task_id in candidate_by_id
                and isinstance(candidate_by_id[task_id].get("image"), str)
                and isinstance(candidate_by_id[task_id].get("image_digest"), str)
                else None
            ),
            image_digest=(
                str(candidate_by_id[task_id]["image_digest"])
                if task_id in candidate_by_id and isinstance(candidate_by_id[task_id].get("image_digest"), str)
                else None
            ),
        )
        for category in ("repository_debugging", "feature_implementation")
        for ids in (_mapping(categories.get(category), category).get("selected_ids"),)
        if isinstance(ids, list)
        for task_id in ids
        for index in (selected_ids.index(task_id),)
    )
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


def _result_values(result: Mapping[str, object], field: str) -> tuple[str, ...]:
    value: Final = result.get(field, ())
    values: Final = (value,) if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)) or not all(isinstance(item, str) for item in values):
        raise ValueError(f"{field} must contain strings")
    return tuple(values)


def _required_metric(result: Mapping[str, object], field: str) -> float:
    value: Final = result.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return float(value)


def record_task_result(
    config: ProfileConfig,
    task: ProfileTask,
    result: Mapping[str, object],
    versions: ProfileVersions,
    guard_sha256: str | None = None,
) -> None:
    task_id: Final = result.get("task_id")
    manifest_sha256: Final = result.get("manifest_sha256", config.manifest_sha256)
    if task_id != task.task_id or not isinstance(manifest_sha256, str):
        raise ValueError("task result must match the scheduled task and include manifest_sha256")
    selected_ids, actual_sha256 = manifest_selection(config.manifest_path, config.manifest_sha256)
    validate_manifest_attempt(task.task_id, manifest_sha256, selected_ids, actual_sha256)
    run_id: Final = result.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("task result must include run_id")
    prior_run_ids: Final = {
        str(existing["run_id"])
        for existing in run_pilot.read_jsonl_locked(config.runs_path)
        if isinstance(existing.get("run_id"), str)
    }
    if run_id in prior_run_ids:
        return
    if result.get("cost_incomplete") is True:
        run_pilot.append_jsonl(
            config.runs_path,
            {
                **dict(result),
                "record_type": "quarantined_attempt",
                "task_id": task.task_id,
                "tenant_id": config.tenant_id,
                "category": task.category,
                "manifest_sha256": manifest_sha256,
                "cost_usd": None,
                "is_infra": True,
                "success": None,
                "error_class": "accounting",
            },
        )
        raise ValueError("attempt cost accounting is incomplete")
    model_alias: Final = result.get("model")
    if not isinstance(model_alias, str) or model_alias not in task.model_order:
        raise ValueError("task result model must match the scheduled model order")
    cost_usd: Final = _required_metric(result, "cost_usd")
    solve_latency: Final = _required_metric(result, "solve_latency_s")
    eval_latency: Final = _required_metric(result, "eval_latency_s")
    returned_models: Final = _result_values(result, "returned_models")
    deployment_ids: Final = _result_values(result, "deployment_ids")
    error_class: Final = result.get("error_class")
    is_infra: Final = error_class in ("infra", "accounting")
    success: Final = result.get("success")
    effective_guard_sha256: Final = guard_sha256 or _guard_sha256(config.root_dir, config.overlays)
    row: Final = {
        **result,
        "record_type": "attempt",
        "task_id": task.task_id,
        "tenant_id": config.tenant_id,
        "category": task.category,
        "model_alias": model_alias,
        "returned_model_version": returned_models[0] if len(returned_models) == 1 else "unreturned",
        "deployment_id": deployment_ids[0] if len(deployment_ids) == 1 else None,
        "harness_version": versions.harness_version,
        "budget_version": versions.budget_version,
        "harness_config_hash": versions.harness_config_hash,
        "is_infra": is_infra,
        "success": None if is_infra else success,
        "cost_usd": cost_usd,
        "latency_s": solve_latency,
        "eval_latency_s": eval_latency,
        "primary_label": task.category,
        "secondary_label": None,
        "manifest_sha256": manifest_sha256,
        "harness_sha256": effective_guard_sha256,
        "model_order": task.model_order,
        "returned_model": returned_models[0] if len(returned_models) == 1 else "unreturned",
        "x_litellm_model_id": deployment_ids[0] if len(deployment_ids) == 1 else None,
        "error_class": error_class,
        "raw_result": dict(result),
        "agent_image_ref": task.agent_image_ref,
        "grader_image_ref": task.grader_image_ref,
        "upstream_digest": task.image_digest,
        "upstream_image_id": task.upstream_image_id,
        "overlay_image_id": task.overlay_image_id,
        "final_image_id": task.final_image_id,
    }
    if not is_infra and not isinstance(success, bool):
        raise ValueError("graded task results must include a boolean success")
    if len(returned_models) > 1 or len(deployment_ids) > 1:
        run_pilot.append_jsonl(
            config.runs_path,
            row
            | {
                "record_type": "quarantined_attempt",
                "error_class": "mixed_version",
                "is_infra": True,
                "success": None,
                "returned_models": returned_models,
                "deployment_ids": deployment_ids,
            },
        )
        raise MixedVersionAttemptError(task.task_id)
    run_pilot.append_jsonl(config.runs_path, row)


def _guard_sha256(root_dir: Path, overlays: Mapping[str, str] | None = None) -> str:
    source_dir: Final = Path(__file__).resolve().parent
    overlay_dir: Final = (
        root_dir / "overlays"
        if (root_dir / "overlays").is_dir()
        else source_dir.parent / "overlays"
    )
    paths: Final = (
        source_dir / "guarded_model.py",
        source_dir / "run_profiling.py",
        source_dir / "run_pilot.py",
        root_dir / "poc" / "pilot_models.json",
        root_dir / "poc" / "pilot.yaml",
    ) + tuple(overlay_dir / name for name in OVERLAY_SOURCE_FILES)
    overlay_policy: Final = DEFAULT_OVERLAYS if overlays is None else overlays
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    digest.update(b"overlay_policy\0")
    digest.update(json.dumps(dict(overlay_policy), sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _profile_versions(
    root_dir: Path,
    runs_path: Path | None = None,
    guard_sha256: str | None = None,
) -> ProfileVersions:
    pilot_config: Final = _mapping(
        json.loads((root_dir / "poc" / "pilot_models.json").read_text()),
        "pilot model manifest",
    )
    if not isinstance(pilot_config.get("solvers"), Mapping):
        raise ValueError("pilot model manifest must contain solvers")
    harness_config: Final = HarnessConfig(
        harness_version="run_pilot_attempt_v1",
        budget_version="budget_1usd_50_600_v1",
        per_run_budget_usd=1.0,
        step_limit=50,
        wall_timeout_s=600,
        mini_swe_agent_version=run_pilot._package_version("mini-swe-agent"),
        guard_sha256=guard_sha256 or _guard_sha256(root_dir),
    )
    rows: Final = run_pilot.read_jsonl_locked(runs_path) if runs_path is not None else []
    model_versions: Final = tuple(
        sorted(
            {
                (str(row["model_alias"]), str(row["returned_model_version"]))
                for row in rows
                if isinstance(row.get("model_alias"), str) and isinstance(row.get("returned_model_version"), str)
            }
        )
    )
    deployments: Final = tuple(
        sorted(
            {
                (str(row["model_alias"]), str(row["deployment_id"]))
                for row in rows
                if isinstance(row.get("model_alias"), str)
            }
        )
    )
    return ProfileVersions(
        harness_config.harness_version,
        harness_config.budget_version,
        harness_config_sha256(harness_config),
        model_versions,
        deployments,
    )


def _load_dataset(name: str, revision: str, split: str) -> tuple[Mapping[str, object], ...]:
    return tuple(load_dataset(name, revision=revision, split=split))


def _task_data(
    config: ProfileConfig,
    tasks: Sequence[ProfileTask],
    agent_source: Sequence[Mapping[str, object]],
    eval_source: Sequence[Mapping[str, object]],
) -> AttemptPaths:
    selected_ids: Final = tuple(task.task_id for task in tasks)
    safe_rows: Final = agent_rows(agent_source, selected_ids)
    eval_by_id: Final = {
        row["instance_id"]: row
        for row in eval_source
        if isinstance(row.get("instance_id"), str) and row["instance_id"] in selected_ids
    }
    if set(eval_by_id) != set(selected_ids):
        raise ValueError("selected IDs are missing from the pinned evaluator dataset")
    if any(not isinstance(eval_by_id[task_id].get("patch"), str) for task_id in selected_ids):
        raise ValueError("pinned evaluator rows must contain gold patches")
    candidate_by_id: Final = {
        task.task_id: task
        for task in tasks
        if task.image_ref is not None and task.image_digest is not None
    }
    if set(candidate_by_id) != set(selected_ids):
        raise ValueError("selected manifest tasks require digest-pinned image references")
    task_rows: Final = {
        str(row["instance_id"]): dict(row)
        | {"image": _agent_image_ref(config, candidate_by_id[str(row["instance_id"])])}
        for row in safe_rows
    }
    private_root: Final = config.root_dir / ".private"
    private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if stat.S_IMODE(private_root.stat().st_mode) != 0o700:
        raise PermissionError("private profiling data directory must have mode 0700")
    data_dir: Final = private_root / "profiling_data"
    data_dir.mkdir(exist_ok=True, mode=0o700)
    if stat.S_IMODE(data_dir.stat().st_mode) != 0o700:
        raise PermissionError("profiling data directory must have mode 0700")
    agent_path: Final = data_dir / "agent_dataset"
    agent_path.mkdir(exist_ok=True, mode=0o700)
    (agent_path / "train.jsonl").write_text(
        "".join(json.dumps(task_rows[task_id]) + "\n" for task_id in selected_ids)
    )
    eval_path: Final = data_dir / "swebench_dev.json"
    eval_path.write_text(json.dumps([eval_by_id[task_id] for task_id in selected_ids]) + "\n")
    base_commits: Final = {
        task_id: str(task_rows[task_id]["base_commit"])
        for task_id in selected_ids
        if isinstance(task_rows[task_id].get("base_commit"), str)
    }
    image_digests: Final = {
        str(_agent_image_ref(config, candidate)): str(candidate.image_digest)
        for candidate in candidate_by_id.values()
    }
    solver_payload: Final = _mapping(
        json.loads((config.root_dir / "poc" / "pilot_models.json").read_text()),
        "pilot model manifest",
    )
    solvers: Final = _mapping(solver_payload.get("solvers"), "pilot solvers")
    solver_configs: Final = {
        alias: dict(_mapping(solvers.get(alias), alias))
        for alias in MODEL_ORDER
    }
    pilot_paths: Final = run_pilot.AttemptPaths(
        run_root=private_root / "profiling_runs",
        dataset_root=agent_path,
        eval_dataset=str(eval_path),
        base_commits=base_commits,
        image_digests=image_digests,
        image_overlay_ids={},
        dataset_metadata={
            "agent_dataset": f"{DATASET_NAME}@{DATASET_REVISION} {DATASET_SPLIT} (sanitized)",
            "eval_dataset": f"SWE-bench/SWE-bench@{EVAL_DATASET_REVISION} dev",
        },
        root_dir=config.root_dir,
    )
    return AttemptPaths(
        pilot_paths,
        task_rows,
        {task_id: str(eval_by_id[task_id]["patch"]) for task_id in selected_ids},
        solver_configs,
    )


def _default_disk_stats() -> Mapping[str, int]:
    usage: Final = shutil.disk_usage("/")
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def _inspect_image(reference: str) -> str | None:
    result: Final = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", reference],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _pull_image(reference: str) -> None:
    subprocess.run(["docker", "pull", reference], check=True, capture_output=True, text=True)


def _tag_image(source: str, target: str) -> None:
    subprocess.run(["docker", "tag", source, target], check=True, capture_output=True, text=True)


def _build_image(
    tag: str,
    dockerfile: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    runner(
        ["docker", "build", "--tag", tag, "-"],
        input=dockerfile,
        check=True,
        capture_output=True,
        text=True,
    )


def _remove_image(reference: str) -> bool:
    return subprocess.run(["docker", "rmi", reference], capture_output=True, check=False).returncode == 0


def _evaluate(task_id: str, patch: str, label: str, paths: run_pilot.AttemptPaths,
              env: dict[str, str]) -> tuple[str, float]:
    run_id: Final = f"{task_id}__preflight_{label}_{time.time_ns()}"
    run_dir: Final = paths.run_root / "preflight" / task_id / label
    return run_pilot._grade(task_id, f"preflight-{label}", patch, run_id, run_dir, env, paths.eval_dataset,
                            paths.root_dir or ROOT)


def _append_preflight(
    config: ProfileConfig,
    task: ProfileTask,
    status: str,
    error_class: str | None,
    gold_status: str | None,
    noop_status: str | None,
    latency_s: float,
) -> None:
    run_pilot.append_jsonl(
        config.runs_path,
        {
            "record_type": "preflight_failed" if error_class is not None else "preflight_passed",
            "task_id": task.task_id,
            "tenant_id": config.tenant_id,
            "category": task.category,
            "manifest_sha256": config.manifest_sha256,
            "status": status,
            "error_class": error_class,
            "gold_status": gold_status,
            "noop_status": noop_status,
            "latency_s": latency_s,
            "cost_usd": 0.0,
            "model_order": task.model_order,
            "agent_image_ref": task.agent_image_ref,
            "grader_image_ref": task.grader_image_ref,
            "upstream_digest": task.image_digest,
            "upstream_image_id": task.upstream_image_id,
            "overlay_image_id": task.overlay_image_id,
            "final_image_id": task.final_image_id,
        },
    )


def _cleanup_plan(config: ProfileConfig, batch: ProfileBatch, plan: BatchImagePlan, ops: RuntimeOps) -> None:
    for task_image in plan.prepared:
        if task_image.restore_grader:
            ops.tag_image(task_image.digest_ref, task_image.grader_ref)
    for reference in reversed(plan.owned_refs):
        ops.remove_image(reference)
        if ops.image_exists(reference):
            raise RuntimeError(f"pulled image remains after cleanup: {reference}")
    run_pilot.append_jsonl(
        config.profile_dir / "disk_events.jsonl",
        {"batch": batch.index, "event": "cleanup", **dict(ops.disk_stats())},
    )


def _prepared_data(data: AttemptPaths, plan: BatchImagePlan) -> AttemptPaths:
    by_task: Final = {task_image.task_id: task_image for task_image in plan.prepared}
    task_rows: Final = {
        task_id: dict(row) | {"image": by_task[task_id].agent_ref} if task_id in by_task else dict(row)
        for task_id, row in data.task_rows.items()
    }
    image_digests: Final = dict(data.pilot.image_digests) | {
        task_image.agent_ref: task_image.upstream_digest for task_image in plan.prepared
    }
    image_overlay_ids: Final = dict(data.pilot.image_overlay_ids) | {
        task_image.agent_ref: task_image.overlay_image_id
        for task_image in plan.prepared
        if task_image.overlay_image_id is not None
    }
    dataset_path: Final = data.pilot.dataset_root / "train.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text("".join(json.dumps(task_rows[task_id]) + "\n" for task_id in task_rows))
    return replace(
        data,
        task_rows=task_rows,
        pilot=replace(data.pilot, image_digests=image_digests, image_overlay_ids=image_overlay_ids),
    )


def _prepared_tasks(batch: ProfileBatch, plan: BatchImagePlan) -> tuple[ProfileTask, ...]:
    by_task: Final = {task_image.task_id: task_image for task_image in plan.prepared}
    return tuple(
        replace(
            task,
            agent_image_ref=by_task[task.task_id].agent_ref,
            grader_image_ref=by_task[task.task_id].grader_ref,
            upstream_image_id=by_task[task.task_id].upstream_image_id,
            overlay_image_id=by_task[task.task_id].overlay_image_id,
            final_image_id=by_task[task.task_id].final_image_id,
        )
        if task.task_id in by_task
        else task
        for task in batch.tasks
    )


def _assert_image_ids(task: ProfileTask, ops: RuntimeOps) -> None:
    if task.agent_image_ref is None or task.grader_image_ref is None:
        raise TransientInfraError("prepared task has no image references", BatchImagePlan((), ()))
    try:
        agent_id: Final = ops.inspect_image(task.agent_image_ref)
        grader_id: Final = ops.inspect_image(task.grader_image_ref)
    except OSError as error:
        raise TransientInfraError("image identity inspection failed", BatchImagePlan((), ())) from error
    if agent_id is None or grader_id is None or agent_id != grader_id:
        raise TransientInfraError("solver and grader image IDs differ", BatchImagePlan((), ()))


def execute_batch(
    config: ProfileConfig,
    batch: ProfileBatch,
    data: AttemptPaths,
    versions: ProfileVersions,
    ops: RuntimeOps,
    env: dict[str, str],
    base_url: str,
    guard: run_pilot.DispatchGuard,
    guard_sha256: str | None = None,
) -> None:
    effective_guard_sha256: Final = guard_sha256 or _guard_sha256(config.root_dir, config.overlays)
    existing: Final = run_pilot.read_jsonl_locked(config.runs_path)
    preflight_rows: Final = {
        str(row["task_id"]): row
        for row in existing
        if row.get("record_type") in {"preflight_passed", "preflight_failed"}
        and isinstance(row.get("task_id"), str)
    }
    try:
        plan: Final = _pull_batch(config, batch, ops, preflight_rows)
    except TransientInfraError as error:
        _cleanup_plan(config, batch, error.plan, ops)
        raise
    try:
        tasks: Final = _prepared_tasks(batch, plan)
        prepared_data: Final = _prepared_data(data, plan)
        current_rows: Final = run_pilot.read_jsonl_locked(config.runs_path)
        current_preflight: Final = {
            str(row["task_id"]): row
            for row in current_rows
            if row.get("record_type") in {"preflight_passed", "preflight_failed"}
            and isinstance(row.get("task_id"), str)
        }
        runnable: Final = tuple(
            task
            for task in tasks
            if task.task_id in current_preflight
            and current_preflight[task.task_id].get("record_type") == "preflight_passed"
            and task.image_ref is not None
            and _assert_image_ids(task, ops) is None
        )
        new_tasks: Final = tuple(task for task in tasks if task.task_id not in current_preflight)
        passed_new: Final = tuple(
            task
            for task in new_tasks
            if _preflight_task(config, batch, task, prepared_data, ops, env)
        )
        solve_tasks: Final = tuple(task for task in runnable + passed_new if task.image_ref is not None)

        def solve_model(task: ProfileTask, model: str) -> None:
            rows: Final = run_pilot.read_jsonl_locked(config.runs_path)
            pair: Final = tuple(
                row
                for row in rows
                if row.get("record_type") in {"attempt", "quarantined_attempt"}
                and row.get("task_id") == task.task_id
                and row.get("model_alias", row.get("model")) == model
            )
            if any(row.get("record_type") == "quarantined_attempt" for row in pair):
                return
            if any(row.get("is_infra") is False for row in pair):
                return
            if pair and (
                pair[-1].get("is_infra") is not True or pair[-1].get("retry_eligible") is not True
            ):
                return
            attempt_number: Final = 1 + len(pair)
            if attempt_number > 2 or not guard.acquire(task.task_id):
                return
            try:
                raw: Final = ops.attempt(
                    task.task_id,
                    model,
                    attempt_number,
                    prepared_data.task_rows[task.task_id],
                    prepared_data.solver_configs[model],
                    None,
                    None,
                    env,
                    base_url,
                    prepared_data.pilot,
                )
                result: Final = dict(raw) | {
                    "task_id": task.task_id,
                    "manifest_sha256": config.manifest_sha256,
                    "model": model,
                }
                record_task_result(config, task, result, versions, effective_guard_sha256)
                if result.get("error_class") == "accounting":
                    guard.stop_for_accounting(task.task_id)
            finally:
                guard.release()
            if result.get("error_class") == "infra" and result.get("retry_eligible") is True:
                solve_model(task, model)

        def solve_task(task: ProfileTask) -> None:
            tuple(solve_model(task, model) for model in task.model_order)

        try:
            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                tuple(executor.map(solve_task, solve_tasks))
        except MixedVersionAttemptError as error:
            guard.stop(error.task_id, "mixed_version_attempt")
    finally:
        _cleanup_plan(config, batch, plan, ops)
    _write_checkpoint(config, batch, versions)


def _pull_batch(
    config: ProfileConfig,
    batch: ProfileBatch,
    ops: RuntimeOps,
    preflight_rows: Mapping[str, Mapping[str, object]],
    prepared: tuple[TaskImage, ...] = (),
    owned: tuple[str, ...] = (),
) -> BatchImagePlan:
    if not batch.tasks:
        return BatchImagePlan(prepared, owned)
    task: Final = batch.tasks[0]
    remaining: Final = ProfileBatch(batch.index, batch.tasks[1:])
    if preflight_rows.get(task.task_id, {}).get("record_type") == "preflight_failed":
        return _pull_batch(config, remaining, ops, preflight_rows, prepared, owned)
    if task.image_ref is None:
        if task.task_id not in preflight_rows:
            _append_preflight(config, task, "preflight_failed", "env", None, None, 0.0)
        return _pull_batch(config, remaining, ops, preflight_rows, prepared, owned)
    try:
        existing_image_id: Final = ops.inspect_image(task.image_ref)
        grader_before_id: Final = ops.inspect_image(_grader_image_ref(task.image_ref))
    except OSError as error:
        raise TransientInfraError("image inspection failed", BatchImagePlan(prepared, owned)) from error
    owned_refs: tuple[str, ...] = ()
    try:
        if existing_image_id is None:
            require_disk_for_pull(int(ops.disk_stats().get("free_bytes", 0)))
            owned_refs = (task.image_ref,)
            ops.pull_image(task.image_ref)
        image_id: Final = ops.inspect_image(task.image_ref)
    except OSError as error:
        raise TransientInfraError("image pull or inspection failed", BatchImagePlan(prepared, owned + owned_refs)) from error
    except Exception as error:
        raise TransientInfraError("image pull failed", BatchImagePlan(prepared, owned + owned_refs)) from error
    if image_id is None:
        raise TransientInfraError("pulled image has no local image ID", BatchImagePlan(prepared, owned + owned_refs))
    grader_ref: Final = _grader_image_ref(task.image_ref)
    if grader_before_id is not None and grader_before_id != image_id:
        _append_preflight(config, task, "preflight_failed", "infra", None, None, 0.0)
        return _pull_batch(config, remaining, ops, preflight_rows, prepared, owned + owned_refs)
    if grader_before_id is None:
        owned_refs = owned_refs + (grader_ref,)
        try:
            ops.tag_image(task.image_ref, grader_ref)
            grader_id: Final = ops.inspect_image(grader_ref)
        except OSError as error:
            raise TransientInfraError(
                "image tag or inspection failed",
                BatchImagePlan(prepared, owned + owned_refs),
            ) from error
        except Exception as error:
            raise TransientInfraError("image tag failed", BatchImagePlan(prepared, owned + owned_refs)) from error
    else:
        grader_id = grader_before_id
    if grader_id != image_id:
        _append_preflight(config, task, "preflight_failed", "infra", None, None, 0.0)
        return _pull_batch(config, remaining, ops, preflight_rows, prepared, owned + owned_refs)
    run_pilot.append_jsonl(
        config.profile_dir / "disk_events.jsonl",
        {
            "batch": batch.index,
            "task_id": task.task_id,
            "event": "pull",
            "image_id": image_id,
            **dict(ops.disk_stats()),
        },
    )
    snippet: Final = _overlay_snippet(config, task)
    overlay_ref: Final = _overlay_image_ref(task.task_id)
    overlay_id: str | None = None
    final_id: Final = image_id
    agent_ref: Final = grader_ref
    restore_grader: Final = False
    if snippet is not None:
        owned_refs = owned_refs + (overlay_ref,)
        try:
            ops.build_image(overlay_ref, _overlay_dockerfile(task.image_ref, snippet))
            overlay_id = ops.inspect_image(overlay_ref)
            if overlay_id is None:
                raise OSError("overlay image has no local image ID")
            ops.tag_image(overlay_ref, grader_ref)
            final_id = ops.inspect_image(grader_ref)
        except OSError as error:
            raise TransientInfraError(
                "overlay build or tag failed",
                BatchImagePlan(prepared, owned + owned_refs),
            ) from error
        except Exception as error:
            raise TransientInfraError("overlay build failed", BatchImagePlan(prepared, owned + owned_refs)) from error
        if final_id != overlay_id:
            raise TransientInfraError(
                "overlay and grader image IDs differ",
                BatchImagePlan(prepared, owned + owned_refs),
            )
        agent_ref = overlay_ref
        restore_grader = grader_before_id is not None
    task_image: Final = TaskImage(
        task.task_id,
        task.image_ref,
        grader_ref,
        agent_ref,
        str(task.image_digest),
        image_id,
        overlay_id,
        final_id,
        owned_refs,
        restore_grader,
    )
    return _pull_batch(
        config,
        remaining,
        ops,
        preflight_rows,
        prepared + (task_image,),
        owned + owned_refs,
    )


def _preflight_task(
    config: ProfileConfig,
    batch: ProfileBatch,
    task: ProfileTask,
    data: AttemptPaths,
    ops: RuntimeOps,
    env: dict[str, str],
) -> bool:
    started: Final = time.monotonic()
    try:
        _assert_image_ids(task, ops)
        gold_status, gold_latency = ops.evaluate(task.task_id, data.gold_patches[task.task_id], "gold",
                                                 data.pilot)
        noop_status, noop_latency = ops.evaluate(task.task_id, NOOP_PATCH, "noop", data.pilot)
    except TransientInfraError:
        raise
    except Exception as error:
        raise TransientInfraError("preflight evaluator failed", BatchImagePlan((), ())) from error
    passed: Final = gold_status == "resolved" and noop_status == "unresolved"
    error_class: Final = None if passed else ("env" if gold_status == "unresolved" and noop_status == "unresolved" else "evaluator")
    _append_preflight(
        config,
        task,
        "preflight_passed" if passed else "preflight_failed",
        error_class,
        gold_status,
        noop_status,
        gold_latency + noop_latency,
    )
    return passed


def _write_checkpoint(config: ProfileConfig, batch: ProfileBatch, versions: ProfileVersions) -> None:
    capability_map: Final = compile_map_from_profiling_runs(
        config.runs_path,
        config.manifest_path,
        config.manifest_sha256,
        config.tenant_id,
        versions,
    )
    config.profile_dir.mkdir(parents=True, exist_ok=True)
    serialized: Final = json.dumps(capability_map, indent=2, sort_keys=True) + "\n"
    (config.profile_dir / f"capability_map_batch_{batch.index:03d}.json").write_text(serialized)
    (config.profile_dir / "capability_map_intermediate.json").write_text(serialized)
    run_pilot.append_jsonl(
        config.profile_dir / "progress.jsonl",
        {"batch": batch.index, "manifest_sha256": config.manifest_sha256, "ts": run_pilot._utc_now()},
    )


def _default_ops(
    config: ProfileConfig,
    data: AttemptPaths,
    env: dict[str, str],
) -> RuntimeOps:
    return RuntimeOps(
        disk_stats=_default_disk_stats,
        inspect_image=_inspect_image,
        pull_image=_pull_image,
        tag_image=_tag_image,
        build_image=lambda tag, dockerfile: _build_image(tag, dockerfile),
        evaluate=lambda task_id, patch, label, paths: _evaluate(task_id, patch, label, paths, env),
        remove_image=_remove_image,
        image_exists=lambda reference: _inspect_image(reference) is not None,
        attempt=run_pilot._attempt,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run capability profiling in guarded, resumable batches")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    parser.add_argument("--tenant-id", default=PUBLIC_DEV_TENANT)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    default_manifest: Final = arguments.root / "profiling" / "profiling_manifest.json"
    manifest_path: Final = arguments.manifest or (
        default_manifest if default_manifest.exists() else arguments.root / "profiling_manifest.json"
    )
    config: Final = ProfileConfig(arguments.root, manifest_path, arguments.manifest_sha256, arguments.tenant_id)
    batches: Final = profile_batches(config, arguments.max_batches)
    if not arguments.execute:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "guard_path": "poc/run_pilot.py:DispatchGuard.acquire -> global_budget_allows",
                    "batches": [
                        {
                            "batch": batch.index,
                            "tasks": [
                                {"task_id": task.task_id, "model_order": task.model_order} for task in batch.tasks
                            ],
                        }
                        for batch in batches
                    ],
                    "workers": WORKERS,
                },
                indent=2,
            )
        )
        return
    guard_sha256: Final = _guard_sha256(config.root_dir, config.overlays)
    versions: Final = _profile_versions(config.root_dir, config.runs_path, guard_sha256)
    agent_source: Final = tuple(load_dataset(DATASET_NAME, revision=DATASET_REVISION, split=DATASET_SPLIT))
    eval_source: Final = _load_dataset("SWE-bench/SWE-bench", EVAL_DATASET_REVISION, "dev")
    data: Final = _task_data(config, tuple(task for batch in batches for task in batch.tasks), agent_source, eval_source)
    key_path: Final = config.root_dir / ".private" / "profiling_key"
    env: Final = os.environ.copy()
    env.pop("LITELLM_GATEWAY_ADMIN_KEY", None)
    env["LITELLM_PROXY_API_KEY"] = run_pilot._read_pilot_key(key_path)
    env["PYTHONPATH"] = str(config.root_dir)
    env["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "1"
    base_url: Final = os.environ.get("LITELLM_GATEWAY_BASE_URL", "https://gateway.litellm-sandbox.ai").rstrip("/")
    guard: Final = profile_guard(config, free_bytes=lambda: _default_disk_stats()["free_bytes"])
    ops: Final = _default_ops(config, data, env)
    for batch in batches:
        execute_batch(config, batch, data, versions, ops, env, base_url, guard, guard_sha256)


if __name__ == "__main__":
    main()
