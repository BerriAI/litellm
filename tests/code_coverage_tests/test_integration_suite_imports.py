from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
INTEGRATION_ROOT: Final = REPO_ROOT / "tests" / "integration"
COLLECTED_COUNT: Final = re.compile(r"^(\d+) tests? collected", re.MULTILINE)


def _integration_test_files() -> tuple[Path, ...]:
    return tuple(sorted(INTEGRATION_ROOT.rglob("test_*.py")))


def _collect_without_injected_pythonpath(target: str) -> subprocess.CompletedProcess[str]:
    env: Final = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    return subprocess.run(
        (sys.executable, "-m", "pytest", target, "--collect-only", "-q", "-p", "no:cacheprovider"),
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_collected(result: subprocess.CompletedProcess[str], target: str) -> None:
    assert result.returncode == 0, f"{target} exited {result.returncode}\n{result.stdout}\n{result.stderr}"
    match: Final = COLLECTED_COUNT.search(result.stdout)
    assert match is not None, f"{target} reported no collection summary\n{result.stdout}"
    assert int(match.group(1)) > 0, f"{target} collected nothing, so nothing was verified\n{result.stdout}"


def test_the_integration_suite_still_has_files_to_guard() -> None:
    assert _integration_test_files()


@pytest.mark.parametrize(
    "target",
    [str(path.relative_to(REPO_ROOT)) for path in _integration_test_files()],
)
def test_each_integration_file_collects_the_way_its_docs_document_it(target: str) -> None:
    _assert_collected(_collect_without_injected_pythonpath(target), target)


def test_the_whole_integration_directory_collects_without_an_injected_pythonpath() -> None:
    target: Final = "tests/integration"
    _assert_collected(_collect_without_injected_pythonpath(target), target)
