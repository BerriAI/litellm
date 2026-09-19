import shlex
import subprocess
import sys
from pathlib import Path
from string import Template
from types import MappingProxyType
from typing import Final

import pytest
import yaml

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_BASE_WORKFLOW: Final = _REPO_ROOT / ".github" / "workflows" / "_test-unit-base.yml"
_SHARD_ENV: Final = MappingProxyType({"WORKERS": "2", "RERUNS": "2", "DIST": "loadscope", "TEST_TIMEOUT_SECONDS": "1"})
_HANG_GUARD_FLAGS: Final = frozenset(("-n", "--dist", "--reruns", "--reruns-delay", "--timeout", "--rerun-except"))
_HUNG_TEST_MODULE: Final = """
import threading

import pytest


@pytest.fixture
def hangs_on_teardown():
    yield
    threading.Event().wait()


def test_body_waits_forever():
    threading.Event().wait()


def test_fixture_teardown_waits_forever(hangs_on_teardown):
    assert True


def test_passes():
    assert True
"""


def _run_tests_script() -> str:
    workflow: Final = yaml.safe_load(_BASE_WORKFLOW.read_text())
    return next(step["run"] for step in workflow["jobs"]["run"]["steps"] if step.get("name") == "Run tests")


def _pytest_invocations(script: str) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(shlex.split(line)) for line in script.replace("\\\n", " ").splitlines() if " pytest " in line)


def _hang_guard_args(invocation: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        Template(token).safe_substitute(_SHARD_ENV)
        for previous, token in zip(("", *invocation), invocation)
        if token.split("=", 1)[0] in _HANG_GUARD_FLAGS or previous in _HANG_GUARD_FLAGS
    )


_INVOCATIONS: Final = _pytest_invocations(_run_tests_script())


@pytest.mark.parametrize(
    "invocation", _INVOCATIONS, ids=tuple("xdist" if "-n" in invocation else "serial" for invocation in _INVOCATIONS)
)
def test_a_hung_test_fails_fast_and_names_itself_under_the_shard_flags(
    invocation: tuple[str, ...], tmp_path: Path
) -> None:
    hung_module: Final = tmp_path / "test_hung.py"
    hung_module.write_text(_HUNG_TEST_MODULE)

    result: Final = subprocess.run(
        (sys.executable, "-m", "pytest", str(hung_module), "-p", "no:cacheprovider", *_hang_guard_args(invocation)),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert result.returncode == 1, result.stdout
    assert "FAILED test_hung.py::test_body_waits_forever" in result.stdout
    assert "ERROR test_hung.py::test_fixture_teardown_waits_forever" in result.stdout
    assert "Timeout (>1.0s) from pytest-timeout" in result.stdout
    assert "1 failed, 2 passed, 1 error" in result.stdout
