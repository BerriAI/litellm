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
    ((["unit", "docs"], True), (["docs"], False), ([], False)),
    ids=("planned skips pass", "an unplanned skip fails", "no plan fails"),
)
def test_the_unit_gate_accepts_a_skipped_job_only_when_the_plan_skipped_it(skipped_by_plan: list[str], passes: bool) -> None:
    tier: Final = _TIERS[1]
    needs: Final = {
        **{job: {"result": "success", "outputs": {}} for job in _needed_jobs(tier)},
        "plan": {"result": "success", "outputs": {"skipped": json.dumps(skipped_by_plan)}},
        "unit": {"result": "skipped", "outputs": {}},
        "docs": {"result": "skipped", "outputs": {}},
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


@pytest.mark.parametrize(
    ("base_ref", "skipped"),
    (
        ("main", []),
        ("litellm_feature", []),
        ("", []),
        ("rc/1.90.0", ["lint-runtime"]),
        ("stable/1.86.x", ["lint-runtime"]),
        ("lts/1.108.x", ["lint-runtime"]),
    ),
)
def test_the_unit_plan_skips_lint_runtime_only_on_release_branches(
    tmp_path: Path, base_ref: str, skipped: list[str]
) -> None:
    plan_steps: Final = _jobs(_TIERS[1])["plan"]["steps"]
    assert isinstance(plan_steps, list)
    step: Final = next(step for step in plan_steps if step.get("id") == "plan")
    output: Final = tmp_path / "github_output"

    subprocess.run(
        ("bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]),
        env={**os.environ, "BASE_REF": base_ref, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )

    outputs: Final = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert json.loads(outputs["skipped"]) == skipped
    assert outputs["lint-runtime"] == ("false" if skipped else "true")
