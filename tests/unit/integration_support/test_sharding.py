from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import pytest

from tests.integration._support.ordering import CaseTiming
from tests.integration._support.sharding import ShardExecution, merged_timings, shard_errors
from tests.integration.conftest import COLLECTED, IntegrationReportPlugin
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


@pytest.mark.parametrize(
    "executions",
    (
        (),
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


def test_timing_snapshot_includes_every_executed_pass_and_skip_once() -> None:
    first: Final = CaseTiming(nodeid=INVENTORY[0], seconds=40.0)
    skipped: Final = CaseTiming(nodeid=INVENTORY[1], seconds=0.0)
    last: Final = CaseTiming(nodeid=INVENTORY[2], seconds=1.0)
    executions: Final = (
        _execution(0, INVENTORY[:2], skipped=INVENTORY[1:2]).model_copy(update={"timings": (first, skipped)}),
        _execution(1, INVENTORY[2:]).model_copy(update={"timings": (last,)}),
    )
    assert shard_errors(executions, 2) == ()
    assert merged_timings(executions) == tuple(sorted((first, skipped, last), key=lambda timing: timing.nodeid))


@pytest.mark.parametrize(
    "timings",
    (
        (),
        (CaseTiming(nodeid=INVENTORY[0], seconds=1.0),),
        (CaseTiming(nodeid=INVENTORY[0], seconds=1.0),) * 2,
        (CaseTiming(nodeid="foreign", seconds=1.0), CaseTiming(nodeid=INVENTORY[1], seconds=0.0)),
    ),
)
def test_incomplete_duplicate_or_foreign_measurements_cannot_seed_the_timing_cache(
    timings: tuple[CaseTiming, ...],
) -> None:
    execution: Final = _execution(0, INVENTORY[:2]).model_copy(update={"timings": timings})
    assert merged_timings((execution,)) == ()


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
