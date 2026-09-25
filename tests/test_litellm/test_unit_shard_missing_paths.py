import os
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
import yaml

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_BASE_WORKFLOW: Final = _REPO_ROOT / ".github" / "workflows" / "_test-unit-base.yml"
_SHARD_ENV: Final = MappingProxyType(
    {"MAX_FAILURES": "10", "RERUNS": "0", "DIST": "loadscope", "TEST_TIMEOUT_SECONDS": "60", "COVERAGE_CORE": "sysmon"}
)
_UV_SHIM: Final = f'#!/usr/bin/env bash\nshift 2\nexec "{sys.executable}" -m "$@"\n'
_PASSING_TEST: Final = "def test_passes():\n    assert True\n"
_FAILING_TEST: Final = "def test_fails():\n    assert False\n"


def _run_tests_script() -> str:
    workflow: Final = yaml.safe_load(_BASE_WORKFLOW.read_text())
    return next(step["run"] for step in workflow["jobs"]["run"]["steps"] if step.get("name") == "Run tests")


def _run_shard(tmp_path: Path, test_path: str, workers: str) -> subprocess.CompletedProcess[str]:
    shim_dir: Final = tmp_path / "bin"
    shim_dir.mkdir()
    (shim_dir / "uv").write_text(_UV_SHIM)
    (shim_dir / "uv").chmod(0o755)
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\naddopts = '-p no:cacheprovider'\n")
    return subprocess.run(
        ("bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", _run_tests_script()),
        cwd=tmp_path,
        env={
            **os.environ,
            **_SHARD_ENV,
            "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
            "TEST_PATH": test_path,
            "WORKERS": workers,
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _write_passing_test(tmp_path: Path) -> Path:
    present: Final = tmp_path / "tests" / "present"
    present.mkdir(parents=True)
    (present / "test_present.py").write_text(_PASSING_TEST)
    return present


@pytest.mark.parametrize("workers", ("0", "2"), ids=("serial", "xdist"))
def test_a_missing_path_is_dropped_and_the_existing_paths_still_run(tmp_path: Path, workers: str) -> None:
    _write_passing_test(tmp_path)

    result: Final = _run_shard(tmp_path, "tests/gone tests/present", workers)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout, result.stdout
    assert "::warning::tests/gone does not exist" in result.stdout


def test_ignore_flags_survive_the_path_filter(tmp_path: Path) -> None:
    present: Final = _write_passing_test(tmp_path)
    (present / "test_ignored.py").write_text(_FAILING_TEST)

    result: Final = _run_shard(tmp_path, "tests/present --ignore=tests/present/test_ignored.py", "0")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout, result.stdout
