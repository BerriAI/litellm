from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.sharding import ShardExecution, shard_errors

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
