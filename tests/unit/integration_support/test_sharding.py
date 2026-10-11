from __future__ import annotations

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
