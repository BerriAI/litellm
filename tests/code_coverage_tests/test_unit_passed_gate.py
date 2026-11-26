import json
import os
import subprocess
from pathlib import Path
from typing import Final

import pytest
import yaml

_WORKFLOWS_DIR: Final = Path(__file__).resolve().parents[2] / ".github" / "workflows"
_TIERS: Final = (
    ("test-linting.yml", "lint-passed", frozenset()),
    ("test-unit.yml", "unit-passed", frozenset({"coverage"})),
    ("test-merge-smoke.yml", "smoke-passed", frozenset({"windows-sdk", "proxy-build-from-pip"})),
)
_Tier = tuple[str, str, frozenset[str]]


def _jobs(tier: _Tier) -> dict[str, dict[str, object]]:
    workflow: Final = yaml.safe_load((_WORKFLOWS_DIR / tier[0]).read_text())
    return workflow["jobs"]


def _gate_script(tier: _Tier) -> str:
    steps: Final = _jobs(tier)[tier[1]]["steps"]
    assert isinstance(steps, list) and len(steps) == 1
    return steps[0]["run"]


def _run_gate(tier: _Tier, results: dict[str, str]) -> subprocess.CompletedProcess[str]:
    needs: Final = {job: {"result": result, "outputs": {}} for job, result in results.items()}
    return subprocess.run(
        ("bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", _gate_script(tier)),
        env={**os.environ, "NEEDS": json.dumps(needs)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _needed_jobs(tier: _Tier) -> tuple[str, ...]:
    needs: Final = _jobs(tier)[tier[1]]["needs"]
    assert isinstance(needs, list)
    return tuple(needs)


@pytest.mark.parametrize("tier", _TIERS, ids=("lint", "unit", "smoke"))
def test_the_gate_passes_when_every_needed_job_succeeded(tier: _Tier) -> None:
    jobs: Final = _needed_jobs(tier)
    assert jobs

    result: Final = _run_gate(tier, {job: "success" for job in jobs})

    assert result.returncode == 0, result.stdout + result.stderr
    assert all(f"{job}: success" in result.stdout for job in jobs), result.stdout


@pytest.mark.parametrize("outcome", ("failure", "cancelled", "skipped"))
@pytest.mark.parametrize("tier", _TIERS, ids=("lint", "unit", "smoke"))
def test_the_gate_fails_when_any_needed_job_did_not_succeed(tier: _Tier, outcome: str) -> None:
    jobs: Final = _needed_jobs(tier)
    assert len(jobs) > 1

    result: Final = _run_gate(tier, {**{job: "success" for job in jobs}, jobs[-1]: outcome})

    assert result.returncode != 0
    assert f"{jobs[-1]}: {outcome}" in result.stdout, result.stdout


@pytest.mark.parametrize("tier", _TIERS, ids=("lint", "unit", "smoke"))
def test_the_gate_waits_for_every_other_job_and_reports_even_when_they_fail(tier: _Tier) -> None:
    jobs: Final = _jobs(tier)
    gate: Final = jobs[tier[1]]

    assert set(_needed_jobs(tier)) == set(jobs) - {tier[1]} - set(tier[2])
    assert gate["if"] == "always()"
    assert gate["permissions"] == {}


@pytest.mark.parametrize(
    ("skipped_by_plan", "passes"),
    ((["unit", "helm"], True), (["helm"], False), ([], False)),
    ids=("planned skips pass", "an unplanned skip fails", "no plan fails"),
)
def test_the_unit_gate_accepts_a_skipped_job_only_when_the_plan_skipped_it(
    skipped_by_plan: list[str], passes: bool
) -> None:
    tier: Final = _TIERS[1]
    needs: Final = {
        **{job: {"result": "success", "outputs": {}} for job in _needed_jobs(tier)},
        "plan": {"result": "success", "outputs": {"skipped": json.dumps(skipped_by_plan)}},
        "unit": {"result": "skipped", "outputs": {}},
        "helm": {"result": "skipped", "outputs": {}},
    }

    result: Final = subprocess.run(
        ("bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", _gate_script(tier)),
        env={**os.environ, "NEEDS": json.dumps(needs)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert (result.returncode == 0) is passes, result.stdout + result.stderr


def _run_plan_step(tmp_path: Path, step_id: str, plan_steps: object, env: dict[str, str]) -> dict[str, str]:
    assert isinstance(plan_steps, list)
    step: Final = next(step for step in plan_steps if step.get("id") == step_id)
    output: Final = tmp_path / "github_output"
    subprocess.run(
        ("bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]),
        env={**os.environ, **env, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return dict(line.split("=", 1) for line in output.read_text().splitlines())


_EVERY_CHANGE_RUNS: Final = {
    "BACKEND": "run",
    "UI": "run",
    "DASHBOARD_BUILD": "run",
    "REDIS_COMPAT": "run",
    "PYTHON_310": "run",
    "HELM": "run",
}


@pytest.mark.parametrize(
    ("decisions", "skipped"),
    (
        ({}, []),
        ({"BACKEND": "", "UI": "", "HELM": ""}, []),
        (
            {"UI": "skip", "DASHBOARD_BUILD": "skip", "REDIS_COMPAT": "skip", "PYTHON_310": "skip", "HELM": "skip"},
            ["ui-unit", "dashboard-build", "redis-compat", "unit-python-310", "helm"],
        ),
        (
            {key: "skip" for key in _EVERY_CHANGE_RUNS},
            [
                "unit",
                "lint-runtime",
                "openai-sdk-compat",
                "ui-unit",
                "dashboard-build",
                "redis-compat",
                "unit-python-310",
                "helm",
            ],
        ),
        ({"HELM": "skip"}, ["helm"]),
    ),
    ids=("everything changed", "undecided runs", "backend-only change", "docs-only change", "one category skipped"),
)
def test_the_unit_plan_skips_exactly_the_jobs_whose_files_did_not_change(
    tmp_path: Path, decisions: dict[str, str], skipped: list[str]
) -> None:
    outputs: Final = _run_plan_step(
        tmp_path, "plan", _jobs(_TIERS[1])["plan"]["steps"], {"BASE_REF": "main", **_EVERY_CHANGE_RUNS, **decisions}
    )

    assert json.loads(outputs["skipped"]) == skipped
    assert set(skipped) <= set(_needed_jobs(_TIERS[1]))


@pytest.mark.parametrize(
    ("base_ref", "release_base"),
    (
        ("main", "false"),
        ("litellm_feature", "false"),
        ("", "false"),
        ("rc/1.90.0", "true"),
        ("stable/1.86.x", "true"),
        ("lts/1.108.x", "true"),
    ),
)
def test_the_unit_plan_marks_release_branch_bases_without_skipping_lint_runtime(
    tmp_path: Path, base_ref: str, release_base: str
) -> None:
    outputs: Final = _run_plan_step(
        tmp_path, "plan", _jobs(_TIERS[1])["plan"]["steps"], {"BASE_REF": base_ref, **_EVERY_CHANGE_RUNS}
    )

    assert outputs["release-base"] == release_base
    assert json.loads(outputs["skipped"]) == []


@pytest.mark.parametrize(
    ("ui", "skipped"),
    (
        ("run", ["e2e-harness", "ui-api-types"]),
        ("", ["e2e-harness", "ui-api-types"]),
        ("skip", ["e2e-harness", "ui-api-types", "ui"]),
    ),
    ids=("dashboard changed", "undecided runs", "dashboard untouched"),
)
def test_the_lint_plan_skips_the_ui_lint_only_when_no_dashboard_file_changed(
    tmp_path: Path, ui: str, skipped: list[str]
) -> None:
    fake_bin: Final = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh: Final = fake_bin / "gh"
    fake_gh.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_gh.chmod(0o755)

    outputs: Final = _run_plan_step(
        tmp_path,
        "plan",
        _jobs(_TIERS[0])["plan"]["steps"],
        {"PR_NUMBER": "1", "UI": ui, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )

    assert outputs["mode"] == "core"
    assert json.loads(outputs["skipped"]) == skipped
