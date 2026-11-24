from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Final

import pytest

from litellm.proxy.db.advisory_lock import (
    AdvisoryLockTransaction,
    AdvisoryLockUnavailable,
    PrismaAdvisoryLock,
    UncommittedResult,
)

LOCK_TIMEOUT: Final = timedelta(seconds=60)


class _Transaction:
    def __init__(self, events: list[object], lock_fails: bool) -> None:
        self.events = events
        self.lock_fails = lock_fails

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        if self.lock_fails:
            raise TimeoutError("lock wait timed out")
        self.events.append(("locked", "pg_advisory_xact_lock" in query, args))
        return [{"locked": 1}]


class _Database:
    def __init__(self, transaction_fails: bool = False, lock_fails: bool = False, commit_fails: bool = False) -> None:
        self.events: list[object] = []
        self.transactions: list[_Transaction] = []
        self.timeouts: list[timedelta] = []
        self.transaction_fails = transaction_fails
        self.lock_fails = lock_fails
        self.commit_fails = commit_fails

    @asynccontextmanager
    async def tx(self, *, timeout: timedelta) -> AsyncIterator[_Transaction]:
        self.timeouts.append(timeout)
        if self.transaction_fails:
            raise ConnectionError("the database is unreachable")
        self.events.append("begin")
        transaction: Final = _Transaction(self.events, self.lock_fails)
        self.transactions.append(transaction)
        yield transaction
        if self.commit_fails:
            raise ConnectionError("the commit did not reach the database")
        self.events.append("commit")


async def test_the_action_runs_on_the_transaction_that_holds_the_lock_for_its_key():
    database: Final = _Database()
    used: Final[list[AdvisoryLockTransaction]] = []

    async def action(transaction: AdvisoryLockTransaction) -> str:
        used.append(transaction)
        database.events.append("action")
        return "done"

    result: Final = await PrismaAdvisoryLock(database, timeout=LOCK_TIMEOUT).while_held("login-a", action)

    assert result == "done"
    assert used == database.transactions
    assert database.events == ["begin", ("locked", True, ("login-a",)), "action", "commit"]
    assert database.timeouts == [LOCK_TIMEOUT]


@pytest.mark.parametrize(
    "database",
    [_Database(transaction_fails=True), _Database(lock_fails=True)],
    ids=["transaction-fails", "lock-fails"],
)
async def test_without_the_lock_the_action_never_runs(database):
    actions: Final[list[str]] = []

    async def action(transaction: AdvisoryLockTransaction) -> str:
        actions.append("action")
        return "ran without the lock"

    result: Final = await PrismaAdvisoryLock(database, timeout=LOCK_TIMEOUT).while_held("login-a", action)

    assert result == AdvisoryLockUnavailable()
    assert actions == []


async def test_an_action_that_finished_before_the_commit_failed_returns_its_result_as_uncommitted():
    database: Final = _Database(commit_fails=True)

    async def action(transaction: AdvisoryLockTransaction) -> str:
        database.events.append("action")
        return "done"

    result: Final = await PrismaAdvisoryLock(database, timeout=LOCK_TIMEOUT).while_held("login-a", action)

    assert result == UncommittedResult("done")
    assert database.events == ["begin", ("locked", True, ("login-a",)), "action"]
