import hashlib
import json
from pathlib import Path

import pytest

from poc import run_pilot
from poc.run_profiling import (
    PHASE_BUDGET_USD,
    AttemptPaths,
    ProfileConfig,
    ProfileBatch,
    ProfileTask,
    RuntimeOps,
    MixedVersionAttemptError,
    execute_batch,
    model_order_for_task,
    profile_batches,
    profile_guard,
    record_task_result,
    require_disk_for_pull,
)
from poc.profiles import ProfileVersions


def _config(tmp_path: Path, task_ids: tuple[str, ...] = ("task-0", "task-1", "task-2", "task-3")) -> ProfileConfig:
    manifest = tmp_path / "profiling_manifest.json"
    manifest_bytes = json.dumps(
        {
            "categories": {
                "repository_debugging": {"selected_ids": task_ids},
                "feature_implementation": {"selected_ids": []},
                "shell_build": {"selected_ids": []},
                "algorithms_data": {"selected_ids": []},
                "unknown": {"selected_ids": []},
            }
        }
    ).encode()
    manifest.write_bytes(manifest_bytes)
    return ProfileConfig(
        root_dir=tmp_path,
        manifest_path=manifest,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


def test_max_batches_preserves_manifest_order_and_balances_model_order(tmp_path: Path) -> None:
    config = _config(tmp_path)

    batches = profile_batches(config, max_batches=1)

    assert len(batches) == 1
    assert tuple(task.task_id for task in batches[0].tasks) == ("task-0", "task-1", "task-2")
    assert tuple(model_order_for_task(index) for index in range(3)) == (
        ("haiku-4.5", "sonnet-5", "opus-5"),
        ("sonnet-5", "opus-5", "haiku-4.5"),
        ("opus-5", "haiku-4.5", "sonnet-5"),
    )
    assert batches[0].tasks[1].model_order == model_order_for_task(1)


def test_runner_record_boundary_checks_membership_and_manifest_sha(tmp_path: Path) -> None:
    config = _config(tmp_path, ("task-0",))
    task = ProfileTask(0, "task-0", "repository_debugging", model_order_for_task(0))
    versions = ProfileVersions("harness-v1", "budget-v1", "config-hash", ())

    with pytest.raises(ValueError, match="scheduled task"):
        record_task_result(
            config,
            task,
            {"task_id": "outside-manifest", "manifest_sha256": config.manifest_sha256},
            versions,
        )
    outside_task = ProfileTask(1, "outside-manifest", "repository_debugging", model_order_for_task(1))
    with pytest.raises(ValueError, match="not selected in the manifest"):
        record_task_result(
            config,
            outside_task,
            {"task_id": "outside-manifest", "manifest_sha256": config.manifest_sha256},
            versions,
        )
    with pytest.raises(ValueError, match="attempt manifest sha256 mismatch"):
        record_task_result(
            config,
            task,
            {"task_id": "task-0", "manifest_sha256": "wrong-digest"},
            versions,
        )

    record_task_result(
        config,
        task,
        {
            "task_id": "task-0",
            "run_id": "run-0",
            "manifest_sha256": config.manifest_sha256,
            "model": "haiku-4.5",
            "returned_models": ["anthropic/claude-haiku-4-5-20251001"],
            "deployment_ids": ["deployment-1"],
            "error_class": "solver",
            "success": True,
            "cost_usd": 0.25,
            "solve_latency_s": 3.0,
            "eval_latency_s": 2.0,
        },
        versions,
    )
    result = json.loads(config.runs_path.read_text())
    assert result["tenant_id"] == "public_dev_poc"
    assert result["category"] == "repository_debugging"
    assert result["model_order"] == ["haiku-4.5", "sonnet-5", "opus-5"]
    assert result["returned_model_version"] == "anthropic/claude-haiku-4-5-20251001"
    assert result["x_litellm_model_id"] == "deployment-1"
    assert result["raw_result"]["model"] == "haiku-4.5"


def test_attempt_cost_and_latency_are_required(tmp_path: Path) -> None:
    config = _config(tmp_path, ("task-0",))
    task = ProfileTask(0, "task-0", "repository_debugging", model_order_for_task(0))
    versions = ProfileVersions("harness-v1", "budget-v1", "config-hash", ())
    result = {
        "task_id": "task-0",
        "run_id": "run-0",
        "manifest_sha256": config.manifest_sha256,
        "model": "haiku-4.5",
        "returned_models": ["concrete-haiku"],
        "deployment_ids": ["deployment-1"],
        "error_class": "solver",
        "success": True,
        "solve_latency_s": 3.0,
        "eval_latency_s": 2.0,
    }

    with pytest.raises(ValueError, match="cost_usd must be finite"):
        record_task_result(config, task, result, versions)
    assert not config.runs_path.exists()


def test_mixed_versions_are_quarantined_with_cost_and_stop_signal(tmp_path: Path) -> None:
    config = _config(tmp_path, ("task-0",))
    task = ProfileTask(0, "task-0", "repository_debugging", model_order_for_task(0))
    versions = ProfileVersions("harness-v1", "budget-v1", "config-hash", ())
    result = {
        "task_id": "task-0",
        "run_id": "run-mixed",
        "manifest_sha256": config.manifest_sha256,
        "model": "haiku-4.5",
        "returned_models": ["concrete-haiku-a", "concrete-haiku-b"],
        "deployment_ids": ["deployment-1"],
        "error_class": "solver",
        "success": True,
        "cost_usd": 0.37,
        "solve_latency_s": 3.0,
        "eval_latency_s": 2.0,
    }

    with pytest.raises(MixedVersionAttemptError):
        record_task_result(config, task, result, versions)
    row = json.loads(config.runs_path.read_text())
    assert row["record_type"] == "quarantined_attempt"
    assert row["error_class"] == "mixed_version"
    assert row["cost_usd"] == 0.37
    assert row["success"] is None


def test_injected_batch_preflight_cleanup_and_attempts(tmp_path: Path) -> None:
    task_id = "task-0"
    image_ref = "swebench/example@sha256:abc"
    manifest = tmp_path / "profiling_manifest.json"
    manifest_row = {
        "categories": {
            "repository_debugging": {"selected_ids": [task_id]},
            "feature_implementation": {"selected_ids": []},
            "shell_build": {"selected_ids": []},
            "algorithms_data": {"selected_ids": []},
            "unknown": {"selected_ids": []},
        },
        "candidates": [
            {"task_id": task_id, "image": "swebench/example:latest", "image_digest": "sha256:abc"}
        ],
    }
    manifest_bytes = json.dumps(manifest_row).encode()
    manifest.write_bytes(manifest_bytes)
    config = ProfileConfig(tmp_path, manifest, hashlib.sha256(manifest_bytes).hexdigest())
    task = ProfileTask(0, task_id, "repository_debugging", model_order_for_task(0), image_ref, "sha256:abc")
    batch = ProfileBatch(1, (task,))
    pilot_paths = run_pilot.AttemptPaths(
        run_root=tmp_path / "raw",
        dataset_root=tmp_path / "agent_dataset",
        eval_dataset=str(tmp_path / "eval.json"),
        base_commits={task_id: "commit"},
        image_digests={image_ref: "sha256:abc"},
        image_overlay_ids={},
        dataset_metadata={},
        root_dir=tmp_path,
    )
    data = AttemptPaths(
        pilot_paths,
        {task_id: {"instance_id": task_id, "image": image_ref}},
        {task_id: "gold patch"},
        {model: {} for model in model_order_for_task(0)},
    )
    images: list[str] = []
    evaluated: list[str] = []
    removed: list[str] = []
    attempted: list[tuple[str, int]] = []

    def inspect(reference: str) -> str | None:
        return "sha256:image-id" if reference in images else None

    def pull(reference: str) -> None:
        images.append(reference)

    def evaluate(_task: str, _patch: str, label: str, _paths: object) -> tuple[str, float]:
        evaluated.append(label)
        return ("resolved" if label == "gold" else "unresolved", 1.0)

    def remove(reference: str) -> bool:
        images.remove(reference)
        removed.append(reference)
        return True

    def attempt(
        task_id_arg: str,
        model: str,
        number: int,
        _task_row: dict[str, object],
        _model_data: dict[str, object],
        _offline: object,
        _jev: object,
        _env: dict[str, str],
        _base_url: str,
        _paths: object,
    ) -> dict[str, object]:
        attempted.append((model, number))
        return {
            "task_id": task_id_arg,
            "run_id": f"{model}-{number}",
            "model": model,
            "returned_models": [f"concrete-{model}"],
            "deployment_ids": [f"deployment-{model}"],
            "error_class": "solver",
            "success": True,
            "cost_usd": 0.1,
            "solve_latency_s": 1.0,
            "eval_latency_s": 1.0,
            "retry_eligible": False,
        }

    operations = RuntimeOps(
        disk_stats=lambda: {"free_bytes": 100 * 1024**3},
        inspect_image=inspect,
        pull_image=pull,
        evaluate=evaluate,
        remove_image=remove,
        image_exists=lambda reference: reference in images,
        attempt=attempt,
    )
    versions = ProfileVersions("harness-v1", "budget-v1", "config-hash", ())
    guard = profile_guard(
        config,
        free_bytes=lambda: 100 * 1024**3,
        committed_spend=lambda: 0.0,
    )
    for model, run_id, error_class, success, retry_eligible in (
        ("haiku-4.5", "seed-graded", "solver", True, False),
        ("sonnet-5", "seed-infra", "infra", None, True),
    ):
        record_task_result(
            config,
            task,
            {
                "task_id": task_id,
                "run_id": run_id,
                "manifest_sha256": config.manifest_sha256,
                "model": model,
                "returned_models": [f"concrete-{model}"],
                "deployment_ids": [f"deployment-{model}"],
                "error_class": error_class,
                "success": success,
                "cost_usd": 0.1,
                "solve_latency_s": 1.0,
                "eval_latency_s": 1.0,
                "retry_eligible": retry_eligible,
            },
            versions,
        )

    execute_batch(config, batch, data, versions, operations, {}, "https://gateway.invalid", guard)

    assert evaluated == ["gold", "noop"]
    assert removed == [image_ref]
    assert not images
    records = [json.loads(line) for line in config.runs_path.read_text().splitlines()]
    assert sum(row["record_type"] == "attempt" for row in records) == 4
    assert attempted == [("sonnet-5", 2), ("opus-5", 1)]
    assert (config.profile_dir / "capability_map_intermediate.json").exists()


def test_profile_guard_enforces_phase_cap_and_pre_pull_disk_reserve(tmp_path: Path) -> None:
    config = _config(tmp_path, ("task-0",))
    guard = profile_guard(
        config,
        free_bytes=lambda: 100 * 1024**3,
        committed_spend=lambda: PHASE_BUDGET_USD,
    )

    assert guard.acquire("task-0") is False
    require_disk_for_pull(60 * 1024**3)
    with pytest.raises(OSError, match="less than 60 GiB"):
        require_disk_for_pull(60 * 1024**3 - 1)


def test_phase_guard_counts_failed_retry_spend(tmp_path: Path) -> None:
    config = _config(tmp_path, ("task-0",))
    run_pilot.append_jsonl(config.runs_path, {"cost_usd": 179.5, "error_class": "infra"})
    run_pilot.append_jsonl(config.runs_path, {"cost_usd": 0.5, "error_class": "solver"})
    guard = profile_guard(config, free_bytes=lambda: 100 * 1024**3)

    assert not guard.acquire("task-after-retries")
