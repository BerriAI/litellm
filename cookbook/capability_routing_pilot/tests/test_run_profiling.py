import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from poc import run_pilot
from poc.run_profiling import (
    DEFAULT_OVERLAYS,
    PHASE_BUDGET_USD,
    AttemptPaths,
    ProfileConfig,
    ProfileBatch,
    ProfileTask,
    RuntimeOps,
    TransientInfraError,
    _build_image,
    _cleanup_plan,
    _pull_batch,
    _guard_sha256,
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
    (tmp_path / "poc").mkdir()
    (tmp_path / "poc" / "pilot_models.json").write_text("{}")
    (tmp_path / "poc" / "pilot.yaml").write_text("")
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
    assert result["latency_s"] == 3.0
    assert result["eval_latency_s"] == 2.0
    assert result["harness_sha256"] == _guard_sha256(config.root_dir)
    assert result["harness_sha256"] != hashlib.sha256(Path(run_pilot.__file__).read_bytes()).hexdigest()


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
    images: dict[str, str] = {}
    evaluated: list[str] = []
    removed: list[str] = []
    attempted: list[tuple[str, int]] = []

    def inspect(reference: str) -> str | None:
        return images.get(reference)

    def pull(reference: str) -> None:
        images[reference] = "sha256:image-id"

    def tag(source: str, target: str) -> None:
        images[target] = images[source]

    def evaluate(_task: str, _patch: str, label: str, _paths: object) -> tuple[str, float]:
        evaluated.append(label)
        return ("resolved" if label == "gold" else "unresolved", 1.0)

    def remove(reference: str) -> bool:
        images.pop(reference, None)
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
        tag_image=tag,
        build_image=lambda _tag, _dockerfile: None,
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
    assert removed == ["swebench/example:latest", image_ref]
    assert not images
    records = [json.loads(line) for line in config.runs_path.read_text().splitlines()]
    assert sum(row["record_type"] == "attempt" for row in records) == 4
    assert attempted == [("sonnet-5", 2), ("opus-5", 1)]
    checkpoint = json.loads((config.profile_dir / "capability_map_intermediate.json").read_text())
    assert any(profile["graded_count"] > 0 for profile in checkpoint["profiles"])
    assert (config.profile_dir / "capability_map_intermediate.json").exists()


def test_overlay_agent_and_grader_refs_share_final_image_id(tmp_path: Path) -> None:
    image_ref = "swebench/pvlib-python@sha256:abc"
    task = ProfileTask(
        0,
        "pvlib__pvlib-python-1191",
        "feature_implementation",
        model_order_for_task(0),
        image_ref,
        "sha256:abc",
    )
    config = ProfileConfig(tmp_path, tmp_path / "manifest.json", "manifest-sha")
    images: dict[str, str] = {}

    def tag(source: str, target: str) -> None:
        images[target] = images[source]

    operations = RuntimeOps(
        disk_stats=lambda: {"free_bytes": 100 * 1024**3},
        inspect_image=lambda reference: images.get(reference),
        pull_image=lambda reference: images.__setitem__(reference, "upstream-id"),
        tag_image=tag,
        build_image=lambda tag, _dockerfile: images.__setitem__(tag, "overlay-id"),
        evaluate=lambda *_args: ("unresolved", 0.1),
        remove_image=lambda reference: not images.pop(reference, None),
        image_exists=lambda reference: reference in images,
        attempt=lambda *_args: {},
    )
    plan = _pull_batch(config, ProfileBatch(1, (task,)), operations, {})
    prepared = plan.prepared[0]
    assert prepared.agent_ref == "routing_poc/profiling-overlay:pvlib__pvlib-python-1191"
    assert prepared.grader_ref == "swebench/pvlib-python:latest"
    assert prepared.overlay_image_id == prepared.final_image_id == "overlay-id"
    _cleanup_plan(config, ProfileBatch(1, (task,)), plan, operations)
    assert not images


def test_transient_pull_leaves_rows_absent_and_retry_prepares_task(tmp_path: Path) -> None:
    task = ProfileTask(0, "task-0", "repository_debugging", model_order_for_task(0), "repo/image@sha256:x", "sha256:x")
    config = ProfileConfig(tmp_path, tmp_path / "manifest.json", "manifest-sha")
    failed = True
    images: dict[str, str] = {}

    def pull(reference: str) -> None:
        if failed:
            raise OSError("registry unavailable")
        images[reference] = "upstream-id"

    def tag(source: str, target: str) -> None:
        images[target] = images[source]

    operations = RuntimeOps(
        disk_stats=lambda: {"free_bytes": 100 * 1024**3},
        inspect_image=lambda reference: images.get(reference),
        pull_image=pull,
        tag_image=tag,
        build_image=lambda _tag, _dockerfile: None,
        evaluate=lambda *_args: ("unresolved", 0.1),
        remove_image=lambda reference: not images.pop(reference, None),
        image_exists=lambda reference: reference in images,
        attempt=lambda *_args: {},
    )
    with pytest.raises(TransientInfraError):
        _pull_batch(config, ProfileBatch(1, (task,)), operations, {})
    assert not config.runs_path.exists()
    failed = False
    plan = _pull_batch(config, ProfileBatch(1, (task,)), operations, {})
    assert plan.prepared[0].final_image_id == "upstream-id"


def test_preflight_failed_task_preserves_later_pull_plan_and_cleanup(tmp_path: Path) -> None:
    first = ProfileTask(0, "task-0", "repository_debugging", model_order_for_task(0), "repo/first@sha256:a", "sha256:a")
    later = ProfileTask(1, "task-1", "repository_debugging", model_order_for_task(1), "repo/later@sha256:b", "sha256:b")
    batch = ProfileBatch(1, (first, later))
    config = ProfileConfig(tmp_path, tmp_path / "manifest.json", "manifest-sha")
    images: dict[str, str] = {}

    def tag(source: str, target: str) -> None:
        images[target] = images[source]

    operations = RuntimeOps(
        disk_stats=lambda: {"free_bytes": 100 * 1024**3},
        inspect_image=lambda reference: images.get(reference),
        pull_image=lambda reference: images.__setitem__(reference, reference),
        tag_image=tag,
        build_image=lambda _tag, _dockerfile: None,
        evaluate=lambda *_args: ("unresolved", 0.1),
        remove_image=lambda reference: not images.pop(reference, None),
        image_exists=lambda reference: reference in images,
        attempt=lambda *_args: {},
    )
    plan = _pull_batch(
        config,
        batch,
        operations,
        {"task-0": {"record_type": "preflight_failed"}},
    )

    assert tuple(task_image.task_id for task_image in plan.prepared) == ("task-1",)
    assert plan.owned_refs == ("repo/later@sha256:b", "repo/later:latest")
    _cleanup_plan(config, batch, plan, operations)
    assert not images


def test_overlay_build_uses_empty_docker_context() -> None:
    runner = Mock()
    dockerfile = "FROM repo/image@sha256:digest\nRUN true\n"

    _build_image("routing_poc/profiling-overlay:task-0", dockerfile, runner)

    runner.assert_called_once_with(
        ["docker", "build", "--tag", "routing_poc/profiling-overlay:task-0", "-"],
        input=dockerfile,
        check=True,
        capture_output=True,
        text=True,
    )


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
