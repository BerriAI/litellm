from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path
from typing import Final

import yaml

REPO: Final = Path(__file__).resolve().parents[2]
WORKFLOW: Final = REPO / ".github" / "workflows" / "test-terraform-provider.yml"


def _route_files_outside_tests() -> tuple[str, ...]:
    listed: Final = subprocess.run(
        ["git", "grep", "-l", "-E", r"APIRouter\(", "--", "*.py", ":(exclude)tests/**"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(listed.stdout.splitlines())


def test_a_pull_request_changing_any_route_file_runs_the_provider_endpoint_audit() -> None:
    pull_request_paths: Final = yaml.safe_load(WORKFLOW.read_text())[True]["pull_request"]["paths"]
    route_files: Final = _route_files_outside_tests()
    assert route_files
    unmatched: Final = tuple(
        name for name in route_files if not any(fnmatch.fnmatch(name, pattern) for pattern in pull_request_paths)
    )
    assert unmatched == ()
