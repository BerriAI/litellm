import json
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Final

import pytest

from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import OAuthToken
from litellm.proxy.mcp_server.outbound_credentials.postgres_refresh_coordinator import (
    AdvisoryLockRefreshCoordinator,
    PrismaAdvisoryLock,
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
    def __init__(self, transaction_fails: bool = False, lock_fails: bool = False) -> None:
        self.events: list[object] = []
        self.timeouts: list[timedelta] = []
        self.transaction_fails = transaction_fails
        self.lock_fails = lock_fails

    @asynccontextmanager
    async def tx(self, *, timeout: timedelta) -> AsyncIterator[_Transaction]:
        self.timeouts.append(timeout)
        if self.transaction_fails:
            raise ConnectionError("the database is unreachable")
        self.events.append("begin")
        yield _Transaction(self.events, self.lock_fails)
        self.events.append("commit")


async def test_the_refresh_runs_while_the_transaction_holds_the_advisory_lock_for_its_flight():
    database: Final = _Database()
    refreshed: Final = OAuthToken(access_token="new")

    async def refresh() -> OAuthToken | None:
        database.events.append("refresh")
        return refreshed

    async def reread() -> OAuthToken | None:
        raise AssertionError("a held lock never rereads")

    token: Final = await AdvisoryLockRefreshCoordinator(PrismaAdvisoryLock(database, timeout=LOCK_TIMEOUT)).run(
        "chatgpt-subscription", "login-a", refresh, reread
    )

    assert token is refreshed
    assert database.events == [
        "begin",
        ("locked", True, (json.dumps(["chatgpt-subscription", "login-a"]),)),
        "refresh",
        "commit",
    ]
    assert database.timeouts == [LOCK_TIMEOUT]


@pytest.mark.parametrize(
    "database",
    [_Database(transaction_fails=True), _Database(lock_fails=True)],
    ids=["transaction-fails", "lock-fails"],
)
async def test_without_the_advisory_lock_the_stored_token_is_reread_and_nothing_is_refreshed(database):
    stored: Final = OAuthToken(access_token="stored")
    refreshes: Final[list[str]] = []

    async def refresh() -> OAuthToken | None:
        refreshes.append("refresh")
        return OAuthToken(access_token="redeemed-without-the-lock")

    async def reread() -> OAuthToken | None:
        return stored

    token: Final = await AdvisoryLockRefreshCoordinator(PrismaAdvisoryLock(database, timeout=LOCK_TIMEOUT)).run(
        "chatgpt-subscription", "login-a", refresh, reread
    )

    assert token is stored
    assert refreshes == []
