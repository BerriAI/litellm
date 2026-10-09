import json
import os
import subprocess
from pathlib import Path
from typing import Final

import pytest
import yaml

_UNIT_WORKFLOW: Final = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "test-unit.yml"
_GATE_JOB: Final = "unit-passed"


def _jobs() -> dict[str, dict[str, object]]:
    return yaml.safe_load(_UNIT_WORKFLOW.read_text())["jobs"]


def _gate_script() -> str:
    steps: Final = _jobs()[_GATE_JOB]["steps"]
    assert isinstance(steps, list) and len(steps) == 1
    return steps[0]["run"]


def _run_gate(results: dict[str, str]) -> subprocess.CompletedProcess[str]:
    needs: Final = {job: {"result": result, "outputs": {}} for job, result in results.items()}
    return subprocess.run(
        ("bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", _gate_script()),
        env={**os.environ, "NEEDS": json.dumps(needs)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _needed_jobs() -> tuple[str, ...]:
    needs: Final = _jobs()[_GATE_JOB]["needs"]
    assert isinstance(needs, list)
    return tuple(needs)


def test_the_gate_passes_when_every_needed_job_succeeded() -> None:
    jobs: Final = _needed_jobs()
    assert jobs

    result: Final = _run_gate(dict.fromkeys(jobs, "success"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert all(f"{job}: success" in result.stdout for job in jobs), result.stdout


@pytest.mark.parametrize("outcome", ("failure", "cancelled", "skipped"))
def test_the_gate_fails_when_any_needed_job_did_not_succeed(outcome: str) -> None:
    jobs: Final = _needed_jobs()
    assert len(jobs) > 1

    result: Final = _run_gate({**dict.fromkeys(jobs, "success"), jobs[-1]: outcome})

    assert result.returncode != 0
    assert f"{jobs[-1]}: {outcome}" in result.stdout, result.stdout


def test_the_gate_waits_for_every_other_job_and_reports_even_when_they_fail() -> None:
    jobs: Final = _jobs()
    gate: Final = jobs[_GATE_JOB]

    assert set(_needed_jobs()) == set(jobs) - {_GATE_JOB}
    assert gate["if"] == "always()"
