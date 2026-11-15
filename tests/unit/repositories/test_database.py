import asyncio
import logging
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm._logging import verbose_proxy_logger
from litellm.proxy.db.pgbouncer import PGBOUNCER_POOLED_ENV_VAR
from litellm.repositories.database import Database

WRITER_URL: Final = "postgresql://litellm:secret@db.invalid:5432/proxy?connection_limit=3"
READER_URL: Final = "postgresql://reader:secret@replica.invalid:5432/proxy?connection_limit=3"
NATIVE_CONNECTIONS: Final = 2


def _prisma(aclose: AsyncMock | None = None) -> MagicMock:
    return MagicMock(aclose=aclose or AsyncMock())


@pytest.fixture
def static_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IAM_TOKEN_DB_AUTH", raising=False)
    monkeypatch.delenv("AZURE_POSTGRESQL_AUTH", raising=False)
    monkeypatch.delenv(PGBOUNCER_POOLED_ENV_VAR, raising=False)


@pytest.mark.asyncio
@pytest.mark.usefixtures("static_password")
async def test_closing_the_database_closes_prisma() -> None:
    aclose: Final = AsyncMock()
    database: Final = await Database.open(_prisma(aclose), WRITER_URL, None, NATIVE_CONNECTIONS)

    aclose.assert_not_awaited()
    await database.aclose()

    aclose.assert_awaited_once()


def _task_names(database: Database) -> tuple[str, ...]:
    return tuple(task.get_coro().__qualname__ for task in database.tasks)


@pytest.mark.asyncio
@pytest.mark.usefixtures("static_password")
async def test_static_password_runs_no_background_task() -> None:
    database: Final = await Database.open(_prisma(), WRITER_URL, READER_URL, NATIVE_CONNECTIONS)

    assert _task_names(database) == ()
    await database.aclose()


@pytest.mark.asyncio
async def test_closing_the_database_stops_both_token_refreshes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    monkeypatch.delenv("AZURE_POSTGRESQL_AUTH", raising=False)
    monkeypatch.delenv(PGBOUNCER_POOLED_ENV_VAR, raising=False)
    database: Final = await Database.open(_prisma(), WRITER_URL, READER_URL, NATIVE_CONNECTIONS)
    assert _task_names(database) == ("keep_token_fresh", "keep_token_fresh")

    await database.aclose()

    assert all(task.done() for task in database.tasks)
    assert database.tasks[0].cancelled() and database.tasks[1].cancelled()


@pytest.mark.asyncio
async def test_rejected_native_settings_still_close_prisma() -> None:
    aclose: Final = AsyncMock()

    with pytest.raises(ValueError, match="must name a host and a user"):
        await Database.open(_prisma(aclose), "postgresql://db.invalid/proxy", None, NATIVE_CONNECTIONS)

    aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.usefixtures("static_password")
async def test_rejected_reader_settings_still_close_prisma() -> None:
    aclose: Final = AsyncMock()

    with pytest.raises(ValueError, match="must name a host and a user"):
        await Database.open(_prisma(aclose), WRITER_URL, "postgresql://replica.invalid/proxy", NATIVE_CONNECTIONS)

    aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_zero_native_connections_open_no_native_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    aclose: Final = AsyncMock()

    database: Final = await Database.open(
        _prisma(aclose), "postgresql://db.invalid/proxy", "postgresql://replica.invalid/proxy", 0
    )

    assert (database.native, database.tasks) == (None, ())
    await database.aclose()
    aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.usefixtures("static_password")
async def test_an_unreachable_native_pool_is_reported_without_failing_boot(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=verbose_proxy_logger.name):
        database: Final = await Database.open(
            _prisma(), "postgresql://litellm:secret@127.0.0.1:1/proxy?pool_timeout=1", None, NATIVE_CONNECTIONS
        )
        await asyncio.wait_for(database.boot_probe, timeout=10)
        await database.aclose()

    assert [
        record.getMessage() for record in caplog.records if "Native database pool cannot reach" in record.getMessage()
    ] == [
        "Native database pool cannot reach the database, so operations ported to it will fail until it connects."
        " Prisma is checked separately. Cause: database query failed: pool timed out while waiting for an open"
        " connection"
    ]
