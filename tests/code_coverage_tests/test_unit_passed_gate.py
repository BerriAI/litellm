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
    ("test-merge-smoke.yml", "smoke-passed", frozenset()),
)
_Tier = tuple[str, str, frozenset[str]]


def _jobs(tier: _Tier) -> dict[str, dict[str, object]]:
    workflow: Final = yaml.safe_load((_WORKFLOWS_DIR / tier[0]).read_text())
    return workflow["jobs"]


def _gate_script(tier: _Tier) -> str:
    steps: Final = _jobs(tier)[tier[1]]["steps"]
    assert isinstance(steps, list) and len(steps) == 1
    return steps[0]["run"]


def _run_gate(
    tier: _Tier, results: dict[str, str], event: str = "pull_request"
) -> subprocess.CompletedProcess[str]:
    needs: Final = {job: {"result": result, "outputs": {}} for job, result in results.items()}
    return subprocess.run(
        ("bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", _gate_script(tier)),
        env={**os.environ, "NEEDS": json.dumps(needs), "EVENT_NAME": event},
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


@pytest.mark.parametrize("outcome", ("failure", "cancelled"))
@pytest.mark.parametrize("tier", _TIERS, ids=("lint", "unit", "smoke"))
def test_the_gate_fails_when_any_needed_job_did_not_succeed(tier: _Tier, outcome: str) -> None:
    jobs: Final = _needed_jobs(tier)
    assert len(jobs) > 1

    for failed_job in jobs:
        result: Final = _run_gate(tier, {**{job: "success" for job in jobs}, failed_job: outcome})

        assert result.returncode != 0, result.stdout
        assert f"{failed_job}: {outcome}" in result.stdout, result.stdout


def _pr_skippable_jobs(tier: _Tier) -> frozenset[str]:
    if tier[1] == "smoke-passed":
        return frozenset(_needed_jobs(tier))
    if tier[1] == "unit-passed":
        return frozenset({"openai-sdk-compat", "redis-compat"})
    return frozenset()


@pytest.mark.parametrize("tier", _TIERS, ids=("lint", "unit", "smoke"))
def test_only_intended_pull_request_skips_pass(tier: _Tier) -> None:
    jobs: Final = _needed_jobs(tier)
    for skipped_job in jobs:
        result: Final = _run_gate(tier, {**{job: "success" for job in jobs}, skipped_job: "skipped"})
        assert (result.returncode == 0) == (skipped_job in _pr_skippable_jobs(tier)), result.stdout


@pytest.mark.parametrize("tier", _TIERS, ids=("lint", "unit", "smoke"))
def test_all_intended_pull_request_skips_pass_together(tier: _Tier) -> None:
    results: Final = {job: "skipped" if job in _pr_skippable_jobs(tier) else "success" for job in _needed_jobs(tier)}
    result: Final = _run_gate(tier, results)
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("event", ("merge_group", "push"))
@pytest.mark.parametrize("tier", _TIERS, ids=("lint", "unit", "smoke"))
def test_skipped_dependencies_fail_outside_pull_requests(tier: _Tier, event: str) -> None:
    jobs: Final = _needed_jobs(tier)
    for skipped_job in jobs:
        result: Final = _run_gate(tier, {**{job: "success" for job in jobs}, skipped_job: "skipped"}, event)
        assert result.returncode != 0, result.stdout


@pytest.mark.parametrize("tier", _TIERS, ids=("lint", "unit", "smoke"))
def test_the_gate_waits_for_every_other_job_and_reports_even_when_they_fail(tier: _Tier) -> None:
    jobs: Final = _jobs(tier)
    gate: Final = jobs[tier[1]]

    assert set(_needed_jobs(tier)) == set(jobs) - {tier[1]} - set(tier[2])
    assert gate["if"] == "always()"
    assert gate["permissions"] == {}
