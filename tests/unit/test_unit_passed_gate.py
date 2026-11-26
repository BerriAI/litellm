import json
import os
import subprocess
from pathlib import Path
from typing import Final

import pytest
import yaml

_WORKFLOW: Final = yaml.safe_load(
    (Path(__file__).resolve().parents[2] / ".github" / "workflows" / "test-unit.yml").read_text()
)


def _plan_outputs(tmp_path: Path, base_ref: str) -> dict[str, str]:
    output: Final = tmp_path / "github_output"
    step: Final = next(step for step in _WORKFLOW["jobs"]["plan"]["steps"] if step.get("id") == "plan")
    subprocess.run(
        ("bash", "-e", "-c", step["run"]),
        env={**os.environ, "BASE_REF": base_ref, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(line.split("=", 1) for line in output.read_text().splitlines())


def _unit_passed(outputs: dict[str, str], skipped_jobs: frozenset[str]) -> bool:
    needs: Final = {
        job: {"result": "skipped" if job in skipped_jobs else "success"}
        for job in _WORKFLOW["jobs"]["unit-passed"]["needs"]
    }
    result: Final = subprocess.run(
        ("bash", "-e", "-c", _WORKFLOW["jobs"]["unit-passed"]["steps"][0]["run"]),
        env={**os.environ, "NEEDS": json.dumps({**needs, "plan": {"result": "success", "outputs": outputs}})},
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


@pytest.mark.parametrize("base_ref", ("main", "litellm_feature", ""))
def test_every_unit_job_must_succeed_outside_release_branches(tmp_path: Path, base_ref: str) -> None:
    outputs: Final = _plan_outputs(tmp_path, base_ref)
    assert _unit_passed(outputs, frozenset())
    assert not _unit_passed(outputs, frozenset({"lint-runtime"}))
    assert not _unit_passed(outputs, frozenset({"unit"}))


@pytest.mark.parametrize("base_ref", ("rc/1.90.0", "stable/1.86.x", "lts/1.108.x"))
def test_release_branches_may_skip_only_lint_runtime(tmp_path: Path, base_ref: str) -> None:
    outputs: Final = _plan_outputs(tmp_path, base_ref)
    assert outputs["lint-runtime"] == "false"
    assert _unit_passed(outputs, frozenset({"lint-runtime"}))
    assert not _unit_passed(outputs, frozenset({"lint-runtime", "unit"}))
    assert not _unit_passed(outputs, frozenset({"ui-unit"}))
