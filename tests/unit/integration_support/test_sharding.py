from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest

from tests.integration._support.sharding import ShardExecution, shard_errors
from tests.integration.conftest import COLLECTED, IntegrationReportPlugin, _shard_files
from tests.integration.conftest import INVENTORY as WORKER_INVENTORY

INVENTORY: Final = ("test_owned.py::test_first", "test_owned.py::test_skipped", "test_owned.py::test_last")
INTEGRATION_PACKAGE: Final = Path(__file__).resolve().parents[2] / "integration"
HEALTHY: Final = "tests/integration/providers/test_healthy.py"
BROKEN: Final = "tests/integration/providers/test_broken.py"
RENAMED: Final = "tests/integration/providers/test_renamed.py"
CLASS_ERROR: Final = "tests/integration/providers/test_class_error.py"
MODULES: Final = MappingProxyType(
    {
        HEALTHY: "def test_one():\n    pass\n\n\ndef test_two():\n    pass\n",
        BROKEN: "from tests.integration._support.process import _removed_helper\n\n\ndef test_never():\n    pass\n",
        CLASS_ERROR: (
            "import pytest\n\n\nclass TestBroken:\n"
            "    @pytest.mark.parametrize('unused', [1])\n    def test_case(self):\n        pass\n"
        ),
    }
)


def _execution(
    index: int,
    collected: tuple[str, ...],
    *,
    inventory: tuple[str, ...] = INVENTORY,
    passed: tuple[str, ...] | None = None,
    skipped: tuple[str, ...] = (),
    shard_count: int = 2,
    complete: bool = True,
    exitstatus: int = 0,
) -> ShardExecution:
    return ShardExecution(
        inventory=inventory,
        collected=collected,
        passed=tuple(node for node in collected if node not in skipped) if passed is None else passed,
        skipped=skipped,
        shard_count=shard_count,
        shard_index=index,
        complete=complete,
        exitstatus=exitstatus,
    )


def test_shard_qualification_retains_passes_and_skip_identities() -> None:
    executions: Final = (
        _execution(0, INVENTORY[:2], skipped=INVENTORY[1:2]),
        _execution(1, INVENTORY[2:]),
    )
    assert shard_errors(executions, 2) == ()


def test_a_selected_subset_can_leave_a_shard_empty_without_losing_the_inventory() -> None:
    executions: Final = (_execution(0, INVENTORY), _execution(1, ()))
    assert shard_errors(executions, 2) == ()
    assert shard_errors((_execution(0, INVENTORY[:1]), _execution(1, ())), 2)


def test_an_empty_assignment_is_distinct_from_no_assignment_or_a_missing_file(tmp_path: Path) -> None:
    assignment: Final = tmp_path / "node-files.txt"
    assert _shard_files(None) is None
    with pytest.raises(pytest.UsageError, match="Cannot read"):
        _shard_files(assignment)
    assignment.write_text("")
    assert _shard_files(assignment) == frozenset()


@pytest.mark.parametrize("content", ("test_owned.py\ntest_owned.py\n", "test_owned.py\n\n", "\n"))
def test_an_invalid_assignment_is_rejected_instead_of_running_extra_tests(tmp_path: Path, content: str) -> None:
    assignment: Final = tmp_path / "node-files.txt"
    assignment.write_text(content)
    with pytest.raises(pytest.UsageError, match="duplicate or blank"):
        _shard_files(assignment)


@pytest.mark.parametrize(
    "executions",
    (
        (),
        (_execution(0, (), inventory=()), _execution(1, (), inventory=())),
        (_execution(0, INVENTORY),),
        (_execution(0, INVENTORY[:2]), _execution(0, INVENTORY[2:])),
        (_execution(0, INVENTORY[:1]), _execution(1, INVENTORY[2:])),
        (_execution(0, INVENTORY), _execution(1, INVENTORY[2:])),
        (_execution(0, INVENTORY[:2]), _execution(1, INVENTORY[2:], inventory=INVENTORY[:2])),
        (_execution(0, INVENTORY[:2]), _execution(1, INVENTORY[2:], complete=False)),
        (_execution(0, INVENTORY[:2]), _execution(1, INVENTORY[2:], exitstatus=1)),
        (_execution(0, INVENTORY[:2]), _execution(1, INVENTORY[2:], shard_count=3)),
        (_execution(0, INVENTORY[:2], inventory=(*INVENTORY, INVENTORY[0])), _execution(1, INVENTORY[2:])),
        (_execution(0, INVENTORY[:2]), _execution(1, INVENTORY[2:], passed=())),
        (_execution(0, INVENTORY[:2]), _execution(1, INVENTORY[2:], skipped=INVENTORY[2:] * 2)),
        (_execution(0, (*INVENTORY[:2], "test_foreign.py::test_other")), _execution(1, INVENTORY[2:])),
    ),
)
def test_shard_qualification_rejects_lost_duplicate_foreign_or_failed_cases(
    executions: tuple[ShardExecution, ...],
) -> None:
    assert shard_errors(executions, 2)


@dataclass(frozen=True, slots=True)
class WorkerResult:
    workeroutput: Mapping[str, object]


def test_worker_results_retain_the_full_inventory_and_reject_disagreement() -> None:
    config: Final = pytest.Config(pytest.PytestPluginManager())
    config.stash[COLLECTED] = INVENTORY[:1]
    plugin: Final = IntegrationReportPlugin(config)
    complete: Final = WorkerResult(MappingProxyType({"integration_inventory": INVENTORY}))
    plugin.pytest_testnodedown(complete, None)
    plugin.pytest_testnodedown(complete, None)
    assert config.stash[WORKER_INVENTORY] == INVENTORY
    assert config.stash[COLLECTED] == INVENTORY[:1]
    incomplete: Final = WorkerResult(MappingProxyType({"integration_inventory": INVENTORY[:1]}))
    with pytest.raises(pytest.UsageError, match="inventories disagree"):
        plugin.pytest_testnodedown(incomplete, None)


XDIST: Final = ("-n", "2", "--dist=loadgroup")
SERIAL: Final = ("-p", "no:xdist")


def _run_shard(
    root: Path, selected: tuple[str, ...], distribution: tuple[str, ...] = XDIST
) -> subprocess.CompletedProcess[str]:
    package: Final = root / "tests" / "integration"
    (package / "providers").mkdir(parents=True)
    (root / "pytest.ini").write_text("[pytest]\n")
    (root / "tests" / "__init__.py").write_text("")
    for name in ("__init__.py", "conftest.py", "run.py", "_support"):
        (package / name).symlink_to(INTEGRATION_PACKAGE / name)
    for file in selected:
        if file in MODULES:
            (root / file).write_text(MODULES[file])
    (root / "shard.txt").write_text("".join(f"{file}\n" for file in selected))
    return subprocess.run(
        (
            sys.executable,
            "-P",
            "-m",
            "pytest",
            "tests/integration/providers",
            "-p",
            "no:cacheprovider",
            *distribution,
            "--integration-shard-count=2",
            "--integration-shard-index=0",
            "--integration-shard-files",
            "shard.txt",
        ),
        cwd=root,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(root),
            "PYTHONPATH": os.pathsep.join((str(root), str(root / "tests"))),
        },
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


@pytest.mark.parametrize(
    ("distribution", "exit_code", "outcome"),
    (
        (XDIST, pytest.ExitCode.TESTS_FAILED, "2 passed, 1 error"),
        (SERIAL, pytest.ExitCode.INTERRUPTED, "Interrupted: 1 error during collection"),
    ),
    ids=("xdist", "serial"),
)
def test_a_broken_import_in_a_shard_is_a_collection_error_not_a_worker_crash(
    tmp_path: Path, distribution: tuple[str, ...], exit_code: pytest.ExitCode, outcome: str
) -> None:
    result: Final = _run_shard(tmp_path, (HEALTHY, BROKEN), distribution)
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert outcome in result.stdout
    assert result.stdout.count(f"ERROR collecting {BROKEN}") == 1
    assert "cannot import name '_removed_helper'" in result.stdout
    assert "INTERNALERROR" not in result.stdout + result.stderr


def test_a_shard_file_that_collects_no_tests_still_fails_as_a_collection_error(tmp_path: Path) -> None:
    result: Final = _run_shard(tmp_path, (HEALTHY, RENAMED))
    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout + result.stderr
    assert result.stdout.count(f"ERROR collecting {RENAMED}") == 1
    assert "2 passed" in result.stdout
    assert "INTERNALERROR" not in result.stdout + result.stderr


def test_a_class_level_collection_error_is_not_also_reported_as_a_file_without_tests(tmp_path: Path) -> None:
    result: Final = _run_shard(tmp_path, (HEALTHY, CLASS_ERROR), SERIAL)
    assert result.returncode == pytest.ExitCode.INTERRUPTED, result.stdout + result.stderr
    assert f"ERROR {CLASS_ERROR}::TestBroken" in result.stdout
    assert "which has no tests" not in result.stdout


def test_a_healthy_shard_runs_every_selected_test(tmp_path: Path) -> None:
    result: Final = _run_shard(tmp_path, (HEALTHY,))
    assert result.returncode == pytest.ExitCode.OK, result.stdout + result.stderr
    assert "2 passed" in result.stdout
