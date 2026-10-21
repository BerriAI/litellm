import hashlib
import json
from pathlib import Path

import pytest

from poc.run_profiling import (
    PHASE_BUDGET_USD,
    ProfileConfig,
    ProfileTask,
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
