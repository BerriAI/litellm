"""Regression tests for the CodeQL workflow's py/log-injection exclusion.

On a full Python scan, run for pushes to main and nightly, py/log-injection passes
CodeQL's 2 GiB result-set limit and fails the whole analysis. The workflow drops that
one query from full scans with a yq step that edits `.github/codeql/codeql-config.yml`.
Pull request scans are diff-informed and finish the query, so they keep it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Final

import pytest
import yaml

ROOT: Final = Path(__file__).resolve().parents[2]
WORKFLOW: Final = ROOT / ".github" / "workflows" / "codeql.yml"
CONFIG: Final = ROOT / ".github" / "codeql" / "codeql-config.yml"
EXCLUSION: Final = {"exclude": {"id": "py/log-injection"}}


def _exclusion_step() -> dict[str, str]:
    steps: Final = [
        step
        for job in yaml.safe_load(WORKFLOW.read_text())["jobs"].values()
        for step in job.get("steps", [])
        if "py/log-injection" in step.get("run", "")
    ]
    assert len(steps) == 1, steps
    return steps[0]


def test_full_python_scans_skip_log_injection_and_pull_request_scans_keep_it() -> None:
    condition: Final = _exclusion_step()["if"]
    assert "matrix.language == 'python'" in condition
    assert "github.event_name != 'pull_request'" in condition


@pytest.mark.skipif(shutil.which("yq") is None, reason="needs yq, which GitHub's Ubuntu runners ship")
def test_exclusion_step_drops_only_log_injection(tmp_path: Path) -> None:
    copy: Final = tmp_path / ".github" / "codeql" / "codeql-config.yml"
    copy.parent.mkdir(parents=True)
    copy.write_text(CONFIG.read_text())

    subprocess.run(["bash", "-c", _exclusion_step()["run"]], cwd=tmp_path, check=True)

    before: Final = yaml.safe_load(CONFIG.read_text())
    after: Final = yaml.safe_load(copy.read_text())
    assert after["query-filters"] == [*before["query-filters"], EXCLUSION]
    assert {key: value for key, value in after.items() if key != "query-filters"} == {
        key: value for key, value in before.items() if key != "query-filters"
    }
