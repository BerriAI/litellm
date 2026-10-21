import hashlib
import json
from pathlib import Path
from typing import Final

import pytest

from poc import run_pilot
from poc.repair_task_env import _repair_task_data, main, repair_task
from poc.run_profiling import (
    DEFAULT_OVERLAYS,
    AttemptPaths,
    ProfileConfig,
    ProfileTask,
    RuntimeOps,
    _guard_sha256,
    model_order_for_task,
)
from poc.profiles import ProfileVersions


TASK_ID: Final = "pydicom__pydicom-1413"
REPAIR_REF: Final = f"routing_poc/profiling-overlay:{TASK_ID}-envfix1"


def _scenario(
    tmp_path: Path,
    gold_status: str = "resolved",
    fail_once: str | None = None,
) -> tuple[
    ProfileConfig,
    ProfileTask,
    dict[str, object],
    str,
    AttemptPaths,
    RuntimeOps,
    run_pilot.DispatchGuard,
    ProfileVersions,
    list[str],
    list[str],
]:
    manifest_path: Final = tmp_path / "profiling" / "profiling_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_bytes: Final = json.dumps(
        {
            "categories": {
                "repository_debugging": {"selected_ids": [TASK_ID]},
                "feature_implementation": {"selected_ids": []},
                "shell_build": {"selected_ids": []},
                "algorithms_data": {"selected_ids": []},
                "unknown": {"selected_ids": []},
            },
            "candidates": [
                {
                    "task_id": TASK_ID,
                    "image": "swebench/pydicom",
                    "image_digest": "sha256:upstream-digest",
                }
            ],
        },
        separators=(",", ":"),
    ).encode()
    manifest_path.write_bytes(manifest_bytes)
    config: Final = ProfileConfig(
        root_dir=tmp_path,
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    prior: Final = {
        "record_type": "preflight_failed",
        "task_id": TASK_ID,
        "error_class": "env",
        "gold_status": "unresolved",
        "noop_status": "unresolved",
        "cost_usd": 0.0,
        "historical_marker": "unchanged",
    }
    run_pilot.append_jsonl(config.runs_path, prior)
    task: Final = ProfileTask(
        0,
        TASK_ID,
        "repository_debugging",
        model_order_for_task(0),
        "swebench/pydicom@sha256:upstream-digest",
        "sha256:upstream-digest",
    )
    correction: Final = {
        TASK_ID: {
            "snippet": 'RUN /opt/miniconda3/envs/testbed/bin/pip install --no-cache-dir "pytest==7.4.4"',
            "cause": "pytest setup compatibility",
            "evidence": ["pydicom/tests/test_valuerep.py::TestBadValueRead::test_read_bad_value_in_VR_default"],
            "applies_on_top_of": "pydicom repo overlay",
        }
    }
    correction_path: Final = tmp_path / "profiling" / "env_corrections.json"
    correction_bytes: Final = json.dumps(correction).encode()
    correction_path.write_bytes(correction_bytes)
    correction_sha256: Final = hashlib.sha256(correction_bytes).hexdigest()
    pilot_paths: Final = run_pilot.AttemptPaths(
        run_root=tmp_path / ".private" / "profiling_runs",
        dataset_root=tmp_path / ".private" / "agent_dataset",
        eval_dataset=str(tmp_path / ".private" / "swebench_dev.json"),
        base_commits={TASK_ID: "base"},
        image_digests={"swebench/pydicom:latest": "sha256:upstream-digest"},
        image_overlay_ids={},
        dataset_metadata={},
        root_dir=tmp_path,
    )
    data: Final = AttemptPaths(
        pilot_paths,
        {TASK_ID: {"instance_id": TASK_ID, "image": "swebench/pydicom:latest"}},
        {TASK_ID: ""},
        {model: {"gateway_model": model} for model in task.model_order},
    )
    images: Final = {}
    attempt_models: Final = []
    attempt_images: Final = []
    failures: Final = {fail_once: True} if fail_once is not None else {}

    def disk_stats() -> dict[str, int]:
        return {"free_bytes": 100 * 1024**3}

    def inspect_image(reference: str) -> str | None:
        return images.get(reference)

    def pull_image(reference: str) -> None:
        images[reference] = "upstream-image-id"

    def tag_image(source: str, target: str) -> None:
        images[target] = images.get(source, source)

    def build_image(tag: str, dockerfile: str) -> None:
        assert dockerfile.startswith("FROM swebench/pydicom@sha256:upstream-digest\n")
        images[tag] = "corrected-image-id"

    def evaluate(task_id: str, patch: str, label: str, paths: run_pilot.AttemptPaths) -> tuple[str, float]:
        del task_id, patch, paths
        return (gold_status if label == "gold" else "unresolved", 0.1)

    def remove_image(reference: str) -> bool:
        return images.pop(reference, None) is not None

    def image_exists(reference: str) -> bool:
        return reference in images

    def attempt(
        task_id: str,
        model: str,
        attempt_number: int,
        task_row: dict[str, object],
        model_data: dict[str, object],
        category_offline: object,
        category_jev: object,
        env: dict[str, str],
        base_url: str,
        paths: run_pilot.AttemptPaths,
    ) -> dict[str, object]:
        del model_data, category_offline, category_jev, env, base_url, paths
        attempt_models.append(model)
        attempt_images.append(str(task_row["image"]))
        if failures.get(model):
            failures[model] = False
            raise RuntimeError("injected interruption")
        return {
            "task_id": task_id,
            "run_id": f"{model}-{attempt_number}-{len(attempt_models)}",
            "model": model,
            "returned_models": [f"provider/{model}"],
            "deployment_ids": [f"deployment-{model}"],
            "error_class": "solver",
            "success": True,
            "cost_usd": 0.1,
            "solve_latency_s": 1.0,
            "eval_latency_s": 0.2,
        }

    ops: Final = RuntimeOps(
        disk_stats,
        inspect_image,
        pull_image,
        tag_image,
        build_image,
        evaluate,
        remove_image,
        image_exists,
        attempt,
    )
    guard: Final = run_pilot.DispatchGuard(
        config.profile_dir / "dispatch_stops.jsonl",
        budget=180.0,
        free_bytes=lambda: 100 * 1024**3,
        committed_spend=lambda: 0.0,
    )
    versions: Final = ProfileVersions("harness", "budget", "config", ())
    return (
        config,
        task,
        correction,
        correction_sha256,
        data,
        ops,
        guard,
        versions,
        attempt_models,
        attempt_images,
    )


def _repair(scenario: tuple[object, ...]) -> bool:
    config, task, correction, correction_sha256, data, ops, guard, versions, _, _ = scenario
    return repair_task(
        config,
        task,
        correction,
        correction_sha256,
        data,
        ops,
        guard,
        versions,
        {},
        "https://gateway.invalid",
    )


def test_repair_appends_correction_and_preflight_without_changing_old_rows(tmp_path: Path) -> None:
    scenario: Final = _scenario(tmp_path)
    config: Final = scenario[0]
    prior_line: Final = config.runs_path.read_text().splitlines()[0]

    assert _repair(scenario) is True

    rows: Final = run_pilot.read_jsonl_locked(config.runs_path)
    assert config.runs_path.read_text().splitlines()[0] == prior_line
    correction_row: Final = next(row for row in rows if row.get("record_type") == "env_correction")
    assert correction_row["task_id"] == TASK_ID
    assert correction_row["cost_usd"] == 0.0
    assert correction_row["upstream_digest"] == "sha256:upstream-digest"
    assert correction_row["overlay_image_id"] == "corrected-image-id"
    assert len([row for row in rows if row.get("record_type") == "preflight_passed"]) == 1
    ops: Final = scenario[5]
    assert not ops.image_exists("swebench/pydicom@sha256:upstream-digest")
    assert not ops.image_exists(REPAIR_REF)
    assert not ops.image_exists("swebench/pydicom:latest")
    assert all(
        row.get("env_correction_sha256") == scenario[3]
        for row in rows
        if row.get("record_type") == "preflight_passed"
    )


def test_gold_failure_after_correction_writes_no_attempts_or_cost(tmp_path: Path) -> None:
    scenario: Final = _scenario(tmp_path, gold_status="unresolved")
    config: Final = scenario[0]

    assert _repair(scenario) is False

    rows: Final = run_pilot.read_jsonl_locked(config.runs_path)
    assert not [row for row in rows if row.get("record_type") in {"attempt", "quarantined_attempt"}]
    assert all(row.get("cost_usd") == 0.0 for row in rows if row.get("task_id") == TASK_ID)
    assert rows[-1]["record_type"] == "preflight_failed"


def test_attempts_use_corrected_image_frozen_guard_and_cyclic_order(tmp_path: Path) -> None:
    scenario: Final = _scenario(tmp_path)
    config, _, _, _, _, _, _, _, attempt_models, attempt_images = scenario

    assert _repair(scenario) is True

    rows: Final = run_pilot.read_jsonl_locked(config.runs_path)
    attempts: Final = tuple(row for row in rows if row.get("record_type") == "attempt")
    expected_guard: Final = _guard_sha256(config.root_dir, DEFAULT_OVERLAYS)
    assert tuple(attempt_models) == model_order_for_task(0)
    assert attempt_images == [REPAIR_REF] * 3
    assert tuple(row["model_alias"] for row in attempts) == model_order_for_task(0)
    assert all(row["agent_image_ref"] == REPAIR_REF for row in attempts)
    assert all(row["harness_sha256"] == expected_guard for row in attempts)


def test_resume_skips_models_with_recorded_repair_attempts(tmp_path: Path) -> None:
    scenario: Final = _scenario(tmp_path, fail_once="sonnet-5")
    attempt_models: Final = scenario[8]

    with pytest.raises(RuntimeError, match="injected interruption"):
        _repair(scenario)
    first_run_models: Final = tuple(attempt_models)

    assert _repair(scenario) is True

    assert first_run_models == ("haiku-4.5", "sonnet-5")
    assert tuple(attempt_models) == (
        "haiku-4.5",
        "sonnet-5",
        "sonnet-5",
        "opus-5",
    )


def test_task_without_correction_entry_is_refused(tmp_path: Path) -> None:
    scenario: Final = _scenario(tmp_path)
    config, task, _, correction_sha256, data, ops, guard, versions, _, _ = scenario

    with pytest.raises(ValueError, match="correction for"):
        repair_task(
            config,
            task,
            {},
            correction_sha256,
            data,
            ops,
            guard,
            versions,
            {},
            "https://gateway.invalid",
        )


def test_repair_dataset_is_task_scoped_and_keeps_agent_rows_sanitized(tmp_path: Path) -> None:
    scenario: Final = _scenario(tmp_path)
    config, task = scenario[:2]
    poc_dir: Final = tmp_path / "poc"
    poc_dir.mkdir()
    (poc_dir / "pilot_models.json").write_text(
        json.dumps({"solvers": {model: {"gateway_model": model} for model in task.model_order}})
    )
    agent_source: Final = (
        {
            "instance_id": TASK_ID,
            "repo": "pydicom/pydicom",
            "base_commit": "base",
            "problem_statement": "reproduce the test issue",
            "version": "3.0",
            "created_at": "2024-01-01T00:00:00Z",
            "patch": "must not enter the agent dataset",
        },
    )
    eval_source: Final = ({"instance_id": TASK_ID, "patch": "local evaluator fixture"},)

    data: Final = _repair_task_data(config, task, agent_source, eval_source)

    dataset_path: Final = data.pilot.dataset_root / "train.jsonl"
    agent_row: Final = json.loads(dataset_path.read_text())
    assert dataset_path.is_relative_to(tmp_path / ".private" / "profiling_repair" / TASK_ID)
    assert not (tmp_path / ".private" / "profiling_data").exists()
    assert agent_row["image"] == REPAIR_REF
    assert "patch" not in agent_row
    assert json.loads(Path(data.pilot.eval_dataset).read_text())[0]["patch"] == "local evaluator fixture"


def test_repair_dry_run_does_not_write_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    scenario: Final = _scenario(tmp_path)
    config, _, correction, _, _, _, _, _, _, _ = scenario
    correction_path: Final = tmp_path / "profiling" / "env_corrections.json"
    correction_path.write_text(json.dumps(correction))
    before: Final = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        "sys.argv",
        [
            "repair_task_env",
            "--root",
            str(tmp_path),
            "--manifest-sha256",
            config.manifest_sha256,
            TASK_ID,
        ],
    )

    main()

    after: Final = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert json.loads(capsys.readouterr().out)["plan"][0]["eligible"] is True
