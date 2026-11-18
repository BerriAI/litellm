import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
import pytest_asyncio

from litellm.repositories.lens_export_repository import LensExportRepository
from tests.integration._support.database import scratch_database

if TYPE_CHECKING:
    from prisma import Prisma

_MIGRATION: Final = (
    Path(__file__).resolve().parents[3]
    / "litellm/proxy-extras/litellm_proxy_extras/migrations/20261118100000_lens_ledger_export/migration.sql"
)
Connection = psycopg.Connection[tuple[object, ...]]


@pytest.fixture
def export_database() -> Iterator[Connection]:
    with scratch_database("lens_ledger_export") as url, psycopg.connect(url, autocommit=True) as connection:
        connection.execute(_MIGRATION.read_text())
        yield connection


@pytest_asyncio.fixture(loop_scope="function")
async def export_client(export_database: Connection) -> AsyncIterator["Prisma"]:
    from prisma import Prisma

    url: Final = urlunsplit(urlsplit(os.environ["DATABASE_URL"])._replace(path=f"/{export_database.info.dbname}"))
    client: Final = Prisma(datasource={"url": url})
    await client.connect()
    try:
        yield client
    finally:
        await client.disconnect()


def _repository(client: "Prisma", owner: str) -> LensExportRepository:
    return LensExportRepository(client, owner)  # pyright: ignore[reportArgumentType]  # Prisma satisfies the transaction protocol


def _seed(connection: Connection, *event_ids: str) -> None:
    for event_id in event_ids:
        connection.execute(
            'INSERT INTO "LiteLLM_LensLedgerExport" (event_id, request_id, event) VALUES (%s, %s, %s)',
            (event_id, f"request-{event_id}", '{"spend": 0.5}'),
        )


def _seed_body(connection: Connection, event_id: str, size: int) -> None:
    connection.execute(
        'INSERT INTO "LiteLLM_LensLedgerPayload" (event_id, field, team_id, api_key, event_time_ms, content, bytes, '
        "sha256) VALUES (%s, 'messages', 'team', 'key', 1, %s, %s, %s)",
        (event_id, "x" * size, size, "0" * 64),
    )


def _events(connection: Connection) -> list[tuple[object, ...]]:
    return connection.execute(
        'SELECT event_id, attempts, parked_at IS NOT NULL FROM "LiteLLM_LensLedgerExport" ORDER BY event_id'
    ).fetchall()


def _due_now(connection: Connection) -> None:
    connection.execute('UPDATE "LiteLLM_LensLedgerExport" SET next_attempt_at = CURRENT_TIMESTAMP')


@pytest.mark.asyncio
async def test_two_exporters_lease_disjoint_events_and_only_the_owner_acknowledges(
    export_database: Connection, export_client: "Prisma"
) -> None:
    _seed(export_database, "a", "b")
    first: Final = _repository(export_client, "first")
    second: Final = _repository(export_client, "second")

    claimed_first: Final = await first.claim_events(1, 60)
    claimed_second: Final = await second.claim_events(10, 60)

    assert {event.event_id for event in claimed_first} | {event.event_id for event in claimed_second} == {"a", "b"}
    assert await second.claim_events(10, 60) == ()
    assert await second.acknowledge_events([claimed_first[0].event_id]) == 0
    assert await first.acknowledge_events([claimed_first[0].event_id]) == 1
    assert [row[0] for row in _events(export_database)] == [claimed_second[0].event_id]


@pytest.mark.asyncio
async def test_a_crashed_owners_lease_expires_and_a_released_lease_is_free_at_once(
    export_database: Connection, export_client: "Prisma"
) -> None:
    _seed(export_database, "crashed", "stopped")
    crashed: Final = _repository(export_client, "crashed")
    stopped: Final = _repository(export_client, "stopped")
    survivor: Final = _repository(export_client, "survivor")
    await crashed.claim_events(1, 60)
    await stopped.claim_events(1, 60)
    export_database.execute(
        """UPDATE "LiteLLM_LensLedgerExport" SET lease_until = CURRENT_TIMESTAMP - INTERVAL '1 second'
        WHERE lease_owner = 'crashed'"""
    )

    assert await stopped.release() == 1
    assert sorted(event.event_id for event in await survivor.claim_events(10, 60)) == ["crashed", "stopped"]
    assert await crashed.acknowledge_events(["crashed"]) == 0


@pytest.mark.asyncio
async def test_transient_failures_back_off_but_never_park_or_delete(
    export_database: Connection, export_client: "Prisma"
) -> None:
    _seed(export_database, "a")
    repository: Final = _repository(export_client, "worker")

    await repository.claim_events(10, 60)
    await repository.retry_events(["a"], "HTTP 503")
    assert export_database.execute(
        'SELECT next_attempt_at > created_at, lease_owner IS NULL FROM "LiteLLM_LensLedgerExport"'
    ).fetchone() == (True, True)
    for _ in range(20):
        _due_now(export_database)
        await repository.claim_events(10, 60)
        await repository.retry_events(["a"], "HTTP 503")

    assert _events(export_database) == [("a", 21, False)]
    assert export_database.execute(
        "SELECT next_attempt_at <= CURRENT_TIMESTAMP + INTERVAL '300 seconds' FROM \"LiteLLM_LensLedgerExport\""
    ).fetchone() == (True,)


@pytest.mark.asyncio
async def test_parked_rows_stay_until_an_operator_requeues_them(
    export_database: Connection, export_client: "Prisma"
) -> None:
    _seed(export_database, "malformed")
    repository: Final = _repository(export_client, "worker")
    await repository.claim_events(10, 60)
    await repository.park_events(["malformed"], "Rejected")

    assert await repository.claim_events(10, 60) == ()
    assert (await repository.backlog()).parked_events == 1
    assert await repository.requeue_parked("events", "Rejected") == 1
    assert [event.event_id for event in await repository.claim_events(10, 60)] == ["malformed"]


@pytest.mark.asyncio
async def test_body_claims_stop_at_the_byte_budget_but_always_take_one(
    export_database: Connection, export_client: "Prisma"
) -> None:
    _seed_body(export_database, "a", 600)
    _seed_body(export_database, "b", 600)
    _seed_body(export_database, "c", 600)
    repository: Final = _repository(export_client, "uploader")

    assert [payload.event_id for payload in await repository.claim_payloads(10, 60, max_bytes=1300)] == ["a", "b"]
    assert [payload.event_id for payload in await repository.claim_payloads(10, 60, max_bytes=100)] == ["c"]


@pytest.mark.asyncio
async def test_body_settlement_belongs_to_the_current_lease_holder(
    export_database: Connection, export_client: "Prisma"
) -> None:
    _seed_body(export_database, "a", 10)
    stale: Final = _repository(export_client, "stale")
    current: Final = _repository(export_client, "current")
    [stale_claim] = await stale.claim_payloads(10, 60, max_bytes=1024)
    export_database.execute(
        """UPDATE "LiteLLM_LensLedgerPayload" SET lease_until = CURRENT_TIMESTAMP - INTERVAL '1 second'"""
    )
    [current_claim] = await current.claim_payloads(10, 60, max_bytes=1024)

    assert await stale.retry_payloads((stale_claim,), "HTTP 503") == 0
    assert await stale.park_payloads((stale_claim,), "HTTP 422") == 0
    assert await stale.acknowledge_payloads((stale_claim,)) == 0
    assert export_database.execute(
        'SELECT lease_owner, attempts, parked_at IS NULL FROM "LiteLLM_LensLedgerPayload"'
    ).fetchone() == ("current", 0, True)
    assert await current.acknowledge_payloads((current_claim,)) == 1
