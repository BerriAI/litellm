import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
import pytest_asyncio
from prisma import Prisma

from litellm.repositories.lens_export_repository import LensExportRepository
from tests.integration._support.database import scratch_database

_MIGRATION: Final = (
    Path(__file__).resolve().parents[3]
    / "litellm/proxy-extras/litellm_proxy_extras/migrations/20261118100000_lens_ledger_export/migration.sql"
)


@pytest.fixture
def export_database() -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    with scratch_database("lens_ledger_export") as url, psycopg.connect(url, autocommit=True) as connection:
        connection.execute(_MIGRATION.read_text())
        yield connection


@pytest_asyncio.fixture(loop_scope="function")
async def export_client(export_database: psycopg.Connection[tuple[object, ...]]) -> AsyncIterator[Prisma]:
    url: Final = urlunsplit(urlsplit(os.environ["DATABASE_URL"])._replace(path=f"/{export_database.info.dbname}"))
    client: Final = Prisma(datasource={"url": url})
    await client.connect()
    try:
        yield client
    finally:
        await client.disconnect()


def _seed(connection: psycopg.Connection[tuple[object, ...]], *event_ids: str) -> None:
    for event_id in event_ids:
        connection.execute(
            'INSERT INTO "LiteLLM_LensLedgerExport" (event_id, request_id, event) VALUES (%s, %s, %s)',
            (event_id, f"request-{event_id}", '{"spend": 0.5}'),
        )


def _remaining(connection: psycopg.Connection[tuple[object, ...]]) -> list[tuple[object, ...]]:
    return connection.execute(
        'SELECT event_id, attempts, parked_at IS NOT NULL FROM "LiteLLM_LensLedgerExport" ORDER BY event_id'
    ).fetchall()


@pytest.mark.asyncio
async def test_two_exporters_lease_disjoint_events_and_only_the_owner_acknowledges(
    export_database: psycopg.Connection[tuple[object, ...]], export_client: Prisma
) -> None:
    _seed(export_database, "a", "b")
    first: Final = LensExportRepository(export_client, "first")  # pyright: ignore[reportArgumentType]  # Prisma satisfies the transaction protocol
    second: Final = LensExportRepository(export_client, "second")  # pyright: ignore[reportArgumentType]  # Prisma satisfies the transaction protocol

    claimed_first: Final = await first.claim_events(1, 60)
    claimed_second: Final = await second.claim_events(10, 60)

    assert {event.event_id for event in claimed_first} | {event.event_id for event in claimed_second} == {"a", "b"}
    assert len(claimed_first) == len(claimed_second) == 1
    assert await second.claim_events(10, 60) == ()
    assert await second.acknowledge_events([claimed_first[0].event_id]) == 0
    assert await first.acknowledge_events([claimed_first[0].event_id]) == 1
    assert [row[0] for row in _remaining(export_database)] == [claimed_second[0].event_id]


@pytest.mark.asyncio
async def test_an_expired_lease_is_reclaimed_and_the_old_owner_loses_it(
    export_database: psycopg.Connection[tuple[object, ...]], export_client: Prisma
) -> None:
    _seed(export_database, "a")
    crashed: Final = LensExportRepository(export_client, "crashed")  # pyright: ignore[reportArgumentType]  # Prisma satisfies the transaction protocol
    survivor: Final = LensExportRepository(export_client, "survivor")  # pyright: ignore[reportArgumentType]  # Prisma satisfies the transaction protocol
    await crashed.claim_events(10, 60)
    export_database.execute(
        """UPDATE "LiteLLM_LensLedgerExport" SET lease_until = CURRENT_TIMESTAMP - INTERVAL '1 second'"""
    )

    assert [event.event_id for event in await survivor.claim_events(10, 60)] == ["a"]
    assert await crashed.acknowledge_events(["a"]) == 0
    assert await survivor.acknowledge_events(["a"]) == 1
    assert _remaining(export_database) == []


@pytest.mark.asyncio
async def test_retries_back_off_and_park_after_the_attempt_limit(
    export_database: psycopg.Connection[tuple[object, ...]], export_client: Prisma
) -> None:
    _seed(export_database, "a")
    repository: Final = LensExportRepository(export_client, "worker")  # pyright: ignore[reportArgumentType]  # Prisma satisfies the transaction protocol

    await repository.claim_events(10, 60)
    await repository.retry_events(["a"], "HTTP 503", max_attempts=2)
    assert _remaining(export_database) == [("a", 1, False)]
    assert export_database.execute(
        "SELECT next_attempt_at >= created_at + INTERVAL '5 seconds', lease_owner IS NULL "
        'FROM "LiteLLM_LensLedgerExport"'
    ).fetchone() == (True, True)

    export_database.execute('UPDATE "LiteLLM_LensLedgerExport" SET next_attempt_at = CURRENT_TIMESTAMP')
    await repository.claim_events(10, 60)
    await repository.retry_events(["a"], "HTTP 503", max_attempts=2)
    export_database.execute('UPDATE "LiteLLM_LensLedgerExport" SET next_attempt_at = CURRENT_TIMESTAMP')

    assert _remaining(export_database) == [("a", 2, True)]
    assert await repository.claim_events(10, 60) == ()


@pytest.mark.asyncio
async def test_cleanup_deletes_only_parked_rows_past_retention(
    export_database: psycopg.Connection[tuple[object, ...]], export_client: Prisma
) -> None:
    _seed(export_database, "old_parked", "new_parked", "pending")
    export_database.execute(
        """UPDATE "LiteLLM_LensLedgerExport" SET parked_at = CURRENT_TIMESTAMP - INTERVAL '3 days'
        WHERE event_id = 'old_parked'"""
    )
    export_database.execute(
        """UPDATE "LiteLLM_LensLedgerExport" SET parked_at = CURRENT_TIMESTAMP WHERE event_id = 'new_parked'"""
    )

    assert await LensExportRepository(export_client, "worker").cleanup(retention_days=2) == 1  # pyright: ignore[reportArgumentType]  # Prisma satisfies the transaction protocol
    assert [row[0] for row in _remaining(export_database)] == ["new_parked", "pending"]
