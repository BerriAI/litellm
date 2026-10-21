import argparse
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Final
from collections.abc import Mapping
from datasets import load_dataset

from poc import run_pilot
from poc.prepare import DATASET_NAME, DATASET_REVISION, DATASET_SPLIT, agent_rows
from poc.run_profiling import (
    DEFAULT_OVERLAYS,
    EVAL_DATASET_REVISION,
    MANIFEST_SHA256,
    MODEL_ORDER,
    AttemptPaths,
    BatchImagePlan,
    ProfileBatch,
    ProfileConfig,
    ProfileTask,
    RuntimeOps,
    TaskImage,
    _assert_image_ids,
    _default_ops,
    _guard_sha256,
    _mapping,
    _overlay_dockerfile,
    _profile_versions,
    profile_batches,
    profile_guard,
    record_task_result,
    require_disk_for_pull,
)
from poc.profiles import ProfileVersions


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _correction_rows(path: Path) -> tuple[Mapping[str, object], str]:
    raw: Final = path.read_bytes()
    payload: Final = _object(json.loads(raw), "environment corrections")
    return payload, _sha256(raw)


def _repair_metadata(
    task: ProfileTask,
    correction: Mapping[str, object],
    correction_sha256: str,
    dockerfile_sha256: str,
    repair_script_sha256: str,
    upstream_image_id: str,
    overlay_image_id: str,
    repair_ref: str,
    grader_ref: str,
) -> dict[str, object]:
    evidence: Final = correction.get("evidence")
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        raise ValueError("correction evidence must be a list of strings")
    cause: Final = correction.get("cause")
    applies_on_top_of: Final = correction.get("applies_on_top_of")
    if not isinstance(cause, str) or not isinstance(applies_on_top_of, str):
        raise ValueError("correction cause and applies_on_top_of must be strings")
    return {
        "env_correction_sha256": correction_sha256,
        "corrections_file_sha256": correction_sha256,
        "dockerfile_sha256": dockerfile_sha256,
        "repair_script_sha256": repair_script_sha256,
        "cause": cause,
        "evidence": evidence,
        "applies_on_top_of": applies_on_top_of,
        "upstream_digest": task.image_digest,
        "upstream_image_id": upstream_image_id,
        "overlay_image_id": overlay_image_id,
        "final_image_id": overlay_image_id,
        "agent_image_ref": repair_ref,
        "grader_image_ref": grader_ref,
    }


def _validate_repair_eligibility(
    task: ProfileTask,
    rows: tuple[dict[str, object], ...],
    correction_sha256: str,
) -> None:
    if not any(
        row.get("task_id") == task.task_id
        and row.get("record_type") == "preflight_failed"
        and row.get("error_class") == "env"
        for row in rows
    ):
        raise ValueError(f"{task.task_id} has no prior environment preflight failure")
    attempts: Final = tuple(
        row
        for row in rows
        if row.get("task_id") == task.task_id
        and row.get("record_type") in {"attempt", "quarantined_attempt"}
    )
    links: Final = {
        (row.get("model_alias"), row.get("attempt"))
        for row in rows
        if row.get("task_id") == task.task_id
        and row.get("record_type") == "env_correction_attempt_link"
        and row.get("env_correction_sha256") == correction_sha256
    }
    if any((row.get("model_alias", row.get("model")), row.get("attempt")) not in links for row in attempts):
        raise ValueError(f"{task.task_id} has attempts outside this environment correction")


def _is_eligible(
    task_id: str,
    tasks: Mapping[str, ProfileTask],
    corrections: Mapping[str, object],
    rows: tuple[dict[str, object], ...],
    correction_sha256: str,
) -> bool:
    task: Final = tasks.get(task_id)
    if task is None:
        return False
    try:
        _object(corrections.get(task_id), f"correction for {task_id}")
        _validate_repair_eligibility(task, rows, correction_sha256)
    except ValueError:
        return False
    return True


def _prepared_repair_task(task: ProfileTask, image: TaskImage) -> ProfileTask:
    return replace(
        task,
        agent_image_ref=image.agent_ref,
        grader_image_ref=image.grader_ref,
        upstream_image_id=image.upstream_image_id,
        overlay_image_id=image.overlay_image_id,
        final_image_id=image.final_image_id,
    )


def _append_repair_preflight(
    config: ProfileConfig,
    task: ProfileTask,
    data: AttemptPaths,
    ops: RuntimeOps,
    env: dict[str, str],
    metadata: Mapping[str, object],
) -> bool:
    with tempfile.TemporaryDirectory(prefix="profiling-repair-preflight-") as temporary:
        temporary_config: Final = replace(config, root_dir=Path(temporary))
        batch: Final = ProfileBatch(task.index // 3 + 1, (task,))
        passed: Final = run_profiling_preflight(temporary_config, batch, task, data, ops, env)
        rows: Final = run_pilot.read_jsonl_locked(temporary_config.runs_path)
        if not rows:
            raise RuntimeError("repair preflight produced no row")
        run_pilot.append_jsonl(config.runs_path, dict(rows[-1]) | dict(metadata))
        return passed


def _repair_task_data(
    config: ProfileConfig,
    task: ProfileTask,
    agent_source: tuple[Mapping[str, object], ...],
    eval_source: tuple[Mapping[str, object], ...],
) -> AttemptPaths:
    if task.image_ref is None or task.image_digest is None:
        raise ValueError("repair task must have a manifest digest image")
    safe_rows: Final = agent_rows(agent_source, (task.task_id,))
    eval_by_id: Final = {
        row["instance_id"]: row
        for row in eval_source
        if isinstance(row.get("instance_id"), str) and row["instance_id"] == task.task_id
    }
    if task.task_id not in eval_by_id or not isinstance(eval_by_id[task.task_id].get("patch"), str):
        raise ValueError("selected task is missing from the pinned evaluator dataset")
    private_root: Final = config.root_dir / ".private"
    private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if stat.S_IMODE(private_root.stat().st_mode) != 0o700:
        raise PermissionError("private profiling data directory must have mode 0700")
    repair_root: Final = private_root / "profiling_repair"
    repair_root.mkdir(exist_ok=True, mode=0o700)
    if stat.S_IMODE(repair_root.stat().st_mode) != 0o700:
        raise PermissionError("private repair data directory must have mode 0700")
    task_root: Final = repair_root / task.task_id
    task_root.mkdir(exist_ok=True, mode=0o700)
    if stat.S_IMODE(task_root.stat().st_mode) != 0o700:
        raise PermissionError("task repair data directory must have mode 0700")
    agent_path: Final = task_root / "agent_dataset"
    agent_path.mkdir(exist_ok=True, mode=0o700)
    if stat.S_IMODE(agent_path.stat().st_mode) != 0o700:
        raise PermissionError("private repair dataset directory must have mode 0700")
    task_row: Final = dict(safe_rows[0]) | {
        "image": f"routing_poc/profiling-overlay:{task.task_id}-envfix1"
    }
    (agent_path / "train.jsonl").write_text(json.dumps(task_row) + "\n")
    eval_path: Final = task_root / "swebench_dev.json"
    eval_path.write_text(json.dumps([eval_by_id[task.task_id]]) + "\n")
    pilot_config: Final = _mapping(
        json.loads((config.root_dir / "poc" / "pilot_models.json").read_text()),
        "pilot model manifest",
    )
    solvers: Final = _mapping(pilot_config.get("solvers"), "pilot solvers")
    solver_configs: Final = {
        model: dict(_mapping(solvers.get(model), model))
        for model in MODEL_ORDER
    }
    repair_ref: Final = str(task_row["image"])
    pilot_paths: Final = run_pilot.AttemptPaths(
        run_root=task_root / "runs",
        dataset_root=agent_path,
        eval_dataset=str(eval_path),
        base_commits={task.task_id: str(safe_rows[0]["base_commit"])},
        image_digests={repair_ref: str(task.image_digest)},
        image_overlay_ids={},
        dataset_metadata={
            "agent_dataset": f"{DATASET_NAME}@{DATASET_REVISION} {DATASET_SPLIT} (sanitized)",
            "eval_dataset": f"SWE-bench/SWE-bench@{EVAL_DATASET_REVISION} dev",
        },
        root_dir=config.root_dir,
    )
    return AttemptPaths(
        pilot_paths,
        {task.task_id: task_row},
        {task.task_id: str(eval_by_id[task.task_id]["patch"])},
        solver_configs,
    )


def run_profiling_preflight(
    config: ProfileConfig,
    batch: ProfileBatch,
    task: ProfileTask,
    data: AttemptPaths,
    ops: RuntimeOps,
    env: dict[str, str],
) -> bool:
    from poc.run_profiling import _preflight_task

    return _preflight_task(config, batch, task, data, ops, env)


def _append_attempt_link(
    config: ProfileConfig,
    task: ProfileTask,
    model: str,
    attempt: int,
    result: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    run_id: Final = result.get("run_id")
    if not isinstance(run_id, str):
        return
    recorded: Final = any(row.get("run_id") == run_id for row in run_pilot.read_jsonl_locked(config.runs_path))
    if recorded:
        run_pilot.append_jsonl(
            config.runs_path,
            {
                "record_type": "env_correction_attempt_link",
                "task_id": task.task_id,
                "model_alias": model,
                "attempt": attempt,
                "run_id": run_id,
                "timestamp": run_pilot._utc_now(),
                **dict(metadata),
            },
        )


def _run_models(
    config: ProfileConfig,
    task: ProfileTask,
    data: AttemptPaths,
    ops: RuntimeOps,
    env: dict[str, str],
    base_url: str,
    guard: run_pilot.DispatchGuard,
    versions: ProfileVersions,
    guard_sha256: str,
    metadata: Mapping[str, object],
) -> None:
    for model in task.model_order:
        rows: Final = run_pilot.read_jsonl_locked(config.runs_path)
        pair: Final = tuple(
            row
            for row in rows
            if row.get("task_id") == task.task_id
            and row.get("record_type") in {"attempt", "quarantined_attempt"}
            and row.get("model_alias", row.get("model")) == model
        )
        if any(row.get("record_type") == "quarantined_attempt" for row in pair):
            continue
        if any(row.get("is_infra") is False for row in pair):
            continue
        if pair and (pair[-1].get("is_infra") is not True or pair[-1].get("retry_eligible") is not True):
            continue
        attempt_number: Final = 1 + len(pair)
        if attempt_number > 2:
            continue
        while attempt_number <= 2:
            if not guard.acquire(task.task_id):
                break
            try:
                result: Final = ops.attempt(
                    task.task_id,
                    model,
                    attempt_number,
                    data.task_rows[task.task_id],
                    data.solver_configs[model],
                    None,
                    None,
                    env,
                    base_url,
                    data.pilot,
                )
                completed: Final = dict(result) | {
                    "task_id": task.task_id,
                    "manifest_sha256": config.manifest_sha256,
                    "model": model,
                    "attempt": attempt_number,
                }
                try:
                    record_task_result(config, task, completed, versions, guard_sha256)
                finally:
                    _append_attempt_link(config, task, model, attempt_number, completed, metadata)
                if completed.get("error_class") == "accounting":
                    guard.stop_for_accounting(task.task_id)
            finally:
                guard.release()
            if completed.get("error_class") != "infra" or completed.get("retry_eligible") is not True:
                break
            attempt_number += 1


def repair_task(
    config: ProfileConfig,
    task: ProfileTask,
    corrections: Mapping[str, object],
    correction_sha256: str,
    data: AttemptPaths,
    ops: RuntimeOps,
    guard: run_pilot.DispatchGuard,
    versions: ProfileVersions,
    env: dict[str, str],
    base_url: str,
) -> bool:
    correction: Final = _object(corrections.get(task.task_id), f"correction for {task.task_id}")
    snippet: Final = correction.get("snippet")
    base_snippet: Final = config.overlays.get(task.task_id.split("__", 1)[0])
    if not isinstance(snippet, str) or not isinstance(base_snippet, str):
        raise ValueError("task correction and repository overlay must both be available")
    rows: Final = tuple(run_pilot.read_jsonl_locked(config.runs_path))
    _validate_repair_eligibility(task, rows, correction_sha256)
    if task.image_ref is None or task.image_digest is None:
        raise ValueError("repair task must have a manifest digest image")
    grader_ref: Final = f"{task.image_ref.split('@', 1)[0]}:latest"
    repair_ref: Final = f"routing_poc/profiling-overlay:{task.task_id}-envfix1"
    upstream_id_before: Final = ops.inspect_image(task.image_ref)
    grader_before_id: Final = ops.inspect_image(grader_ref)
    owned_refs: tuple[str, ...] = ()
    try:
        if upstream_id_before is None:
            require_disk_for_pull(int(ops.disk_stats().get("free_bytes", 0)))
            owned_refs = owned_refs + (task.image_ref,)
            ops.pull_image(task.image_ref)
        upstream_id: Final = ops.inspect_image(task.image_ref)
        if upstream_id is None:
            raise RuntimeError("manifest image pull returned no local image ID")
        if ops.inspect_image(repair_ref) is not None:
            raise RuntimeError(f"refusing to overwrite existing repair image {repair_ref}")
        dockerfile: Final = _overlay_dockerfile(task.image_ref, f"{base_snippet.rstrip()}\n{snippet.strip()}")
        dockerfile_sha256: Final = _sha256(dockerfile.encode())
        owned_refs = owned_refs + (repair_ref,)
        ops.build_image(repair_ref, dockerfile)
        overlay_id: Final = ops.inspect_image(repair_ref)
        if overlay_id is None:
            raise RuntimeError("repair overlay build returned no local image ID")
        if grader_before_id is None:
            owned_refs = owned_refs + (grader_ref,)
        ops.tag_image(repair_ref, grader_ref)
        image: Final = TaskImage(
            task.task_id,
            task.image_ref,
            grader_ref,
            repair_ref,
            task.image_digest,
            upstream_id,
            overlay_id,
            overlay_id,
            owned_refs,
            grader_before_id is not None,
        )
        prepared_task: Final = _prepared_repair_task(task, image)
        _assert_image_ids(prepared_task, ops)
        metadata: Final = _repair_metadata(
            task,
            correction,
            correction_sha256,
            dockerfile_sha256,
            _sha256(Path(__file__).read_bytes()),
            upstream_id,
            overlay_id,
            repair_ref,
            grader_ref,
        )
        if not any(
            row.get("record_type") == "env_correction"
            and row.get("task_id") == task.task_id
            and row.get("env_correction_sha256") == correction_sha256
            for row in rows
        ):
            run_pilot.append_jsonl(
                config.runs_path,
                {
                    "record_type": "env_correction",
                    "task_id": task.task_id,
                    "manifest_sha256": config.manifest_sha256,
                    "cost_usd": 0.0,
                    "timestamp": run_pilot._utc_now(),
                    **metadata,
                },
            )
        prepared_data: Final = run_profiling_prepared_data(data, BatchImagePlan((image,), owned_refs))
        passed: Final = _append_repair_preflight(config, prepared_task, prepared_data, ops, env, metadata)
        if passed:
            _run_models(config, prepared_task, prepared_data, ops, env, base_url, guard, versions, _guard_sha256(config.root_dir, DEFAULT_OVERLAYS), metadata)
        return passed
    finally:
        if grader_before_id is not None:
            ops.tag_image(grader_before_id, grader_ref)
        for reference in reversed(owned_refs):
            ops.remove_image(reference)
            if ops.image_exists(reference):
                raise RuntimeError(f"repair-owned image remains after cleanup: {reference}")


def run_profiling_prepared_data(data: AttemptPaths, plan: BatchImagePlan) -> AttemptPaths:
    from poc.run_profiling import _prepared_data

    return _prepared_data(data, plan)


def _task_map(config: ProfileConfig) -> dict[str, ProfileTask]:
    return {
        task.task_id: task
        for batch in profile_batches(config)
        for task in batch.tasks
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair a verified task environment for profiling")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("task_ids", nargs="+")
    arguments = parser.parse_args()
    default_manifest: Final = arguments.root / "profiling" / "profiling_manifest.json"
    manifest_path: Final = arguments.manifest or (
        default_manifest
        if default_manifest.is_file()
        else Path(__file__).resolve().parents[1] / "profiling" / "profiling_manifest.json"
    )
    config: Final = ProfileConfig(arguments.root, manifest_path, arguments.manifest_sha256)
    correction_path: Final = arguments.root / "profiling" / "env_corrections.json"
    corrections, correction_sha256 = _correction_rows(correction_path)
    tasks: Final = _task_map(config)
    rows: Final = tuple(run_pilot.read_jsonl_locked(config.runs_path))
    plan: Final = tuple(
        {
            "task_id": task_id,
            "model_order": tasks[task_id].model_order if task_id in tasks else None,
            "repair_image": f"routing_poc/profiling-overlay:{task_id}-envfix1",
            "eligible": _is_eligible(task_id, tasks, corrections, rows, correction_sha256),
        }
        for task_id in arguments.task_ids
    )
    if not arguments.execute:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2))
        return
    if any(not item["eligible"] for item in plan):
        raise SystemExit("one or more requested tasks are not eligible for environment repair")
    selected_tasks: Final = tuple(tasks[task_id] for task_id in arguments.task_ids)
    agent_source: Final = tuple(load_dataset(DATASET_NAME, revision=DATASET_REVISION, split=DATASET_SPLIT))
    eval_source: Final = tuple(
        load_dataset("SWE-bench/SWE-bench", revision=EVAL_DATASET_REVISION, split="dev")
    )
    env: Final = os.environ.copy()
    env.pop("LITELLM_GATEWAY_ADMIN_KEY", None)
    env["LITELLM_PROXY_API_KEY"] = run_pilot._read_pilot_key(arguments.root / ".private" / "profiling_key")
    env["PYTHONPATH"] = str(arguments.root)
    env["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "1"
    base_url: Final = os.environ.get("LITELLM_GATEWAY_BASE_URL", "https://gateway.litellm-sandbox.ai").rstrip("/")
    guard_sha256: Final = _guard_sha256(arguments.root, DEFAULT_OVERLAYS)
    versions: Final = _profile_versions(arguments.root, config.runs_path, guard_sha256)
    guard: Final = profile_guard(config)
    for task in selected_tasks:
        data: Final = _repair_task_data(config, task, agent_source, eval_source)
        ops: Final = _default_ops(config, data, env)
        repair_task(config, task, corrections, correction_sha256, data, ops, guard, versions, env, base_url)


if __name__ == "__main__":
    main()
