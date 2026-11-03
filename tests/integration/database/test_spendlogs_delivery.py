import asyncio
import os
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
import pytest_asyncio
from prisma import Prisma

from litellm.repositories.accounting_delivery_repository import (
    AccountingDeliveryRepository,
    DeliveryProjection,
    LedgerTransaction,
)

from litellm.repositories.spend_log_insert import spend_log_insert_statement
from tests.integration._support.database import scratch_database

_MIGRATIONS: Final = Path(__file__).resolve().parents[3] / "litellm-proxy-extras/litellm_proxy_extras/migrations"


@pytest.fixture
def delivery_database() -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    with scratch_database("spendlogs_delivery") as url, psycopg.connect(url, autocommit=True) as connection:
        connection.execute((_MIGRATIONS / "20261027000000_add_accounting_ledger_tables/migration.sql").read_text())
        connection.execute((_MIGRATIONS / "20261103000000_remove_accounting_events/migration.sql").read_text())
        connection.execute("""CREATE TABLE "LiteLLM_SpendLogs" (
            request_id text PRIMARY KEY, call_type text NOT NULL, spend float8 NOT NULL DEFAULT 0,
            status text,
            metadata jsonb DEFAULT '{}', "startTime" timestamp(3) NOT NULL,
            created_at timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        connection.execute((_MIGRATIONS / "20261103010000_spendlogs_accounting_delivery/migration.sql").read_text())
        yield connection


def _insert(connection: psycopg.Connection[tuple[object, ...]], rows: Sequence[Mapping[str, object]]) -> None:
    statement: Final = spend_log_insert_statement(rows)
    connection.execute(statement[0].replace("$1", "%s"), (statement[1],))


def _row(request_id: str, path: str = "ledger") -> Mapping[str, object]:
    return {
        "request_id": request_id,
        "call_type": "acompletion",
        "spend": 0.5,
        "startTime": "2026-11-02T12:00:00+00:00",
        "metadata": '{"accounting_savings":{"compression":0.2}}',
        "accounting_path": path,
    }


def test_insert_creates_only_new_ledger_deliveries(delivery_database: psycopg.Connection[tuple[object, ...]]) -> None:
    _insert(delivery_database, (_row("ledger"), _row("legacy", "legacy")))
    _insert(delivery_database, (_row("ledger"), _row("legacy")))
    assert delivery_database.execute('SELECT request_id, state::text FROM "LiteLLM_AccountingDelivery"').fetchall() == [
        ("ledger", "pending")
    ]
    assert delivery_database.execute(
        'SELECT request_id, spend, accounting_path FROM "LiteLLM_SpendLogs" ORDER BY request_id'
    ).fetchall() == [("ledger", 0.5, "ledger"), ("legacy", 0.5, "legacy")]
    assert delivery_database.execute(
        "SELECT metadata->'accounting_savings'->>'compression' FROM \"LiteLLM_SpendLogs\" WHERE request_id='ledger'"
    ).fetchone() == ("0.2",)


def test_duplicate_after_receipt_cleanup_does_not_recreate_delivery(
    delivery_database: psycopg.Connection[tuple[object, ...]],
) -> None:
    _insert(delivery_database, (_row("request"),))
    delivery_database.execute('DELETE FROM "LiteLLM_AccountingDelivery"')
    _insert(delivery_database, (_row("request"),))
    assert delivery_database.execute('SELECT COUNT(*) FROM "LiteLLM_AccountingDelivery"').fetchone() == (0,)
    assert delivery_database.execute('SELECT COUNT(*) FROM "LiteLLM_SpendLogs"').fetchone() == (1,)


def test_delivery_failure_rolls_back_every_inserted_spend_row(
    delivery_database: psycopg.Connection[tuple[object, ...]],
) -> None:
    delivery_database.execute("""ALTER TABLE "LiteLLM_AccountingDelivery"
        ADD CONSTRAINT reject_poison CHECK (request_id <> 'poison')""")
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert(delivery_database, (_row("good"), _row("poison")))
    assert delivery_database.execute('SELECT COUNT(*) FROM "LiteLLM_AccountingDelivery"').fetchone() == (0,)
    assert delivery_database.execute('SELECT COUNT(*) FROM "LiteLLM_SpendLogs"').fetchone() == (0,)


@pytest.mark.asyncio
async def test_worker_rolls_back_destinations_and_parks_only_isolated_failure(
    delivery_database: psycopg.Connection[tuple[object, ...]],
) -> None:
    url: Final = urlunsplit(urlsplit(os.environ["DATABASE_URL"])._replace(path=f"/{delivery_database.info.dbname}"))
    client: Final = Prisma(datasource={"url": url})
    delivery_database.execute("CREATE TABLE totals (spend float8 NOT NULL)")
    delivery_database.execute("INSERT INTO totals VALUES (0)")
    _insert(delivery_database, (_row("bad"), _row("good")))

    async def apply(transaction: LedgerTransaction, rows: tuple[DeliveryProjection, ...]) -> None:
        await transaction.execute_raw(
            "UPDATE totals SET spend = spend + $1::float8", sum(float(str(row.row["spend"])) for row in rows)
        )
        if any(row.row["request_id"] == "bad" for row in rows):
            raise ValueError("Invalid projected destination")

    await client.connect()
    try:
        repository: Final = AccountingDeliveryRepository(client)
        failed: Final = await repository.apply_pending(apply, max_attempts=2)
        assert failed.failed == 2
        assert failed.parked == ()
        assert delivery_database.execute("SELECT spend FROM totals").fetchone() == (0.0,)
        assert delivery_database.execute(
            'SELECT DISTINCT state::text, attempts FROM "LiteLLM_AccountingDelivery"'
        ).fetchall() == [("pending", 1)]
        delivery_database.execute('UPDATE "LiteLLM_AccountingDelivery" SET next_attempt_at = CURRENT_TIMESTAMP')
        isolated: Final = await repository.apply_pending(apply, max_attempts=2)
        assert isolated.parked == ("bad",)
        applied: Final = await repository.apply_pending(apply, max_attempts=2)
        assert applied.applied == 1
        assert delivery_database.execute("SELECT spend FROM totals").fetchone() == (0.5,)
        assert delivery_database.execute(
            'SELECT request_id, state::text FROM "LiteLLM_AccountingDelivery" ORDER BY request_id'
        ).fetchall() == [("bad", "parked"), ("good", "applied")]
        assert (await repository.apply_pending(apply)).applied == 0
    finally:
        await client.disconnect()


@pytest_asyncio.fixture(loop_scope="function")
async def delivery_client(delivery_database: psycopg.Connection[tuple[object, ...]]) -> AsyncIterator[Prisma]:
    url: Final = urlunsplit(urlsplit(os.environ["DATABASE_URL"])._replace(path=f"/{delivery_database.info.dbname}"))
    client: Final = Prisma(datasource={"url": url})
    delivery_database.execute("CREATE TABLE totals (spend float8 NOT NULL)")
    delivery_database.execute("INSERT INTO totals VALUES (0)")
    await client.connect()
    try:
        yield client
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_two_workers_claim_disjoint_rows_and_apply_once(
    delivery_database: psycopg.Connection[tuple[object, ...]],
    delivery_client: Prisma,
) -> None:
    _insert(delivery_database, (_row("first"), _row("second")))
    entered: Final[asyncio.Queue[str]] = asyncio.Queue()
    release: Final = asyncio.Event()

    async def apply(transaction: LedgerTransaction, rows: tuple[DeliveryProjection, ...]) -> None:
        assert len(rows) == 1
        await entered.put(str(rows[0].row["request_id"]))
        await release.wait()
        await transaction.execute_raw("UPDATE totals SET spend = spend + $1::float8", float(str(rows[0].row["spend"])))

    repository: Final = AccountingDeliveryRepository(delivery_client)
    first: Final = asyncio.create_task(repository.apply_pending(apply, batch_size=1))
    first_id: Final = await asyncio.wait_for(entered.get(), timeout=10)
    second: Final = asyncio.create_task(repository.apply_pending(apply, batch_size=1))
    try:
        second_id: Final = await asyncio.wait_for(entered.get(), timeout=10)
        assert first_id != second_id
    finally:
        release.set()
        results: Final = await asyncio.gather(first, second)
    assert sum(result.applied for result in results) == 2
    assert delivery_database.execute("SELECT spend FROM totals").fetchone() == (1.0,)
    assert delivery_database.execute(
        "SELECT COUNT(*) FROM \"LiteLLM_AccountingDelivery\" WHERE state = 'applied'"
    ).fetchone() == (2,)


@pytest.mark.asyncio
async def test_cancelled_worker_rolls_back_and_restart_applies_once(
    delivery_database: psycopg.Connection[tuple[object, ...]],
    delivery_client: Prisma,
) -> None:
    _insert(delivery_database, (_row("restart"),))
    wrote: Final = asyncio.Event()
    hold: Final = asyncio.Event()

    async def interrupted(transaction: LedgerTransaction, rows: tuple[DeliveryProjection, ...]) -> None:
        await transaction.execute_raw("UPDATE totals SET spend = spend + $1::float8", float(str(rows[0].row["spend"])))
        wrote.set()
        await hold.wait()

    repository: Final = AccountingDeliveryRepository(delivery_client)
    task: Final = asyncio.create_task(repository.apply_pending(interrupted))
    await asyncio.wait_for(wrote.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert delivery_database.execute("SELECT spend FROM totals").fetchone() == (0.0,)
    assert delivery_database.execute('SELECT state::text, attempts FROM "LiteLLM_AccountingDelivery"').fetchone() == (
        "pending",
        0,
    )

    async def complete(transaction: LedgerTransaction, rows: tuple[DeliveryProjection, ...]) -> None:
        await transaction.execute_raw("UPDATE totals SET spend = spend + $1::float8", float(str(rows[0].row["spend"])))

    assert (await repository.apply_pending(complete)).applied == 1
    assert (await repository.apply_pending(complete)).applied == 0
    assert delivery_database.execute("SELECT spend FROM totals").fetchone() == (0.5,)


@pytest.mark.parametrize("state", ["pending", "parked", "applied", "missing"])
@pytest.mark.asyncio
async def test_batch_takeover_preserves_pending_work_or_queues_difference(
    delivery_database: psycopg.Connection[tuple[object, ...]],
    delivery_client: Prisma,
    state: str,
) -> None:
    from types import SimpleNamespace

    from litellm.repositories.table_repositories import SpendLogsRepository

    initial: Final = {**_row("batch"), "spend": 0.0, "call_type": "aretrieve_batch", "status": "success"}
    _insert(delivery_database, (initial,))
    if state == "missing":
        delivery_database.execute('DELETE FROM "LiteLLM_AccountingDelivery"')
    else:
        delivery_database.execute(
            'UPDATE "LiteLLM_AccountingDelivery" SET state = %s::"AccountingDeliveryState"', (state,)
        )
    repository: Final = SpendLogsRepository(SimpleNamespace(db=delivery_client, writer_db=delivery_client))
    taken: Final = await repository.take_over_batch({**initial, "spend": 1.0, "accounting_path": "legacy"})
    assert taken.charged
    assert taken.accounting_path == "ledger"
    assert delivery_database.execute('SELECT spend, accounting_path FROM "LiteLLM_SpendLogs"').fetchone() == (
        1.0,
        "ledger",
    )
    receipt: Final = delivery_database.execute(
        'SELECT state::text, previous_row FROM "LiteLLM_AccountingDelivery"'
    ).fetchone()
    assert receipt is not None
    assert receipt[0] == ("parked" if state == "parked" else "pending")
    if state in {"applied", "missing"}:
        assert isinstance(receipt[1], dict)
        assert receipt[1]["spend"] == 0
        assert receipt[1]["accounting_path"] == "ledger"
    else:
        assert receipt[1] is None
    assert not (await repository.take_over_batch({**initial, "spend": 1.0})).charged


@pytest.mark.asyncio
async def test_legacy_batch_takeover_stays_legacy_across_flag_change(
    delivery_database: psycopg.Connection[tuple[object, ...]],
    delivery_client: Prisma,
) -> None:
    from types import SimpleNamespace

    from litellm.repositories.table_repositories import SpendLogsRepository

    initial: Final = {
        **_row("legacy-batch", "legacy"),
        "spend": 0.0,
        "call_type": "aretrieve_batch",
        "status": "success",
    }
    _insert(delivery_database, (initial,))
    repository: Final = SpendLogsRepository(SimpleNamespace(db=delivery_client, writer_db=delivery_client))
    result: Final = await repository.take_over_batch({**initial, "spend": 0.5, "accounting_path": "ledger"})
    assert result.charged
    assert result.accounting_path == "legacy"
    assert delivery_database.execute('SELECT COUNT(*) FROM "LiteLLM_AccountingDelivery"').fetchone() == (0,)
    assert delivery_database.execute('SELECT spend, accounting_path FROM "LiteLLM_SpendLogs"').fetchone() == (
        0.5,
        "legacy",
    )


@pytest.mark.asyncio
async def test_retention_preserves_unfinished_rows_and_deletes_settled_rows(
    delivery_database: psycopg.Connection[tuple[object, ...]], delivery_client: Prisma
) -> None:
    from datetime import datetime

    from litellm.repositories.table_repositories import SpendLogsRepository

    _insert(delivery_database, tuple(_row(state) for state in ("pending", "parked", "applied")))
    _insert(delivery_database, (_row("legacy", "legacy"),))
    delivery_database.execute('UPDATE "LiteLLM_AccountingDelivery" SET state = request_id::"AccountingDeliveryState"')
    async with delivery_client.tx() as transaction:
        result = await SpendLogsRepository.delete_settled_before(transaction, datetime(2026, 11, 3), 100)
    assert result.deleted == 2
    assert result.postponed
    assert delivery_database.execute('SELECT request_id FROM "LiteLLM_SpendLogs" ORDER BY request_id').fetchall() == [
        ("parked",),
        ("pending",),
    ]


@pytest.mark.asyncio
async def test_partition_drop_waits_for_all_deliveries_to_settle(
    delivery_database: psycopg.Connection[tuple[object, ...]], delivery_client: Prisma
) -> None:
    from litellm.repositories.table_repositories import SpendLogsRepository

    _insert(delivery_database, (_row("pending"),))
    delivery_database.execute('CREATE TABLE "LiteLLM_SpendLogs_p20261102" AS SELECT * FROM "LiteLLM_SpendLogs"')
    async with delivery_client.tx() as transaction:
        assert not await SpendLogsRepository.drop_settled_partition(transaction, "LiteLLM_SpendLogs_p20261102")
    assert delivery_database.execute('SELECT COUNT(*) FROM "LiteLLM_SpendLogs_p20261102"').fetchone() == (1,)
    delivery_database.execute("UPDATE \"LiteLLM_AccountingDelivery\" SET state = 'applied'")
    async with delivery_client.tx() as transaction:
        assert await SpendLogsRepository.drop_settled_partition(transaction, "LiteLLM_SpendLogs_p20261102")
    assert delivery_database.execute("SELECT to_regclass('\"LiteLLM_SpendLogs_p20261102\"')").fetchone() == (None,)


@pytest.mark.asyncio
async def test_cleanup_skips_a_batch_takeover_holding_the_spend_row_lock(
    delivery_database: psycopg.Connection[tuple[object, ...]], delivery_client: Prisma
) -> None:
    from datetime import datetime

    from litellm.repositories.table_repositories import SpendLogsRepository

    _insert(delivery_database, (_row("takeover"),))
    delivery_database.execute("UPDATE \"LiteLLM_AccountingDelivery\" SET state = 'applied'")
    async with delivery_client.tx() as takeover:
        await takeover.query_raw('SELECT request_id FROM "LiteLLM_SpendLogs" FOR UPDATE')
        async with delivery_client.tx() as cleanup:
            result = await SpendLogsRepository.delete_settled_before(cleanup, datetime(2026, 11, 3), 100)
        assert result.deleted == 0
        await takeover.execute_raw("UPDATE \"LiteLLM_AccountingDelivery\" SET state = 'pending'")
    async with delivery_client.tx() as cleanup:
        result = await SpendLogsRepository.delete_settled_before(cleanup, datetime(2026, 11, 3), 100)
    assert result.deleted == 0
    assert result.postponed
    assert delivery_database.execute('SELECT COUNT(*) FROM "LiteLLM_SpendLogs"').fetchone() == (1,)


@pytest.mark.asyncio
async def test_receipt_cleanup_keeps_pending_and_parked_rows_and_reports_lag(
    delivery_database: psycopg.Connection[tuple[object, ...]], delivery_client: Prisma
) -> None:
    _insert(delivery_database, tuple(_row(state) for state in ("pending", "parked", "applied")))
    delivery_database.execute(
        'UPDATE "LiteLLM_AccountingDelivery" '
        'SET state = request_id::"AccountingDeliveryState", '
        "created_at = CURRENT_TIMESTAMP - INTERVAL '40 days', "
        "applied_at = CURRENT_TIMESTAMP - INTERVAL '40 days'"
    )
    repository = AccountingDeliveryRepository(delivery_client)
    assert await repository.cleanup_applied() == 1
    metrics = await repository.metrics()
    assert metrics.pending_count == 1
    assert metrics.parked_count == 1
    assert metrics.oldest_pending_seconds >= 40 * 86400
    assert delivery_database.execute(
        'SELECT request_id FROM "LiteLLM_AccountingDelivery" ORDER BY request_id'
    ).fetchall() == [("parked",), ("pending",)]


@pytest.mark.asyncio
async def test_batch_takeover_waits_for_worker_then_applies_only_difference(
    delivery_database: psycopg.Connection[tuple[object, ...]],
    delivery_client: Prisma,
) -> None:
    from types import SimpleNamespace
    from litellm.repositories.table_repositories import SpendLogsRepository

    initial = {**_row("batch-race"), "spend": 0.0, "call_type": "aretrieve_batch", "status": "success"}
    _insert(delivery_database, (initial,))
    loaded = asyncio.Event()
    release = asyncio.Event()

    async def apply(transaction: LedgerTransaction, rows: tuple[DeliveryProjection, ...]) -> None:
        loaded.set()
        await release.wait()
        await transaction.execute_raw(
            "UPDATE totals SET spend = spend + $1::float8",
            sum(float(str(row.row["spend"])) - float(str((row.previous_row or {}).get("spend", 0))) for row in rows),
        )

    worker = AccountingDeliveryRepository(delivery_client)
    repository = SpendLogsRepository(SimpleNamespace(db=delivery_client, writer_db=delivery_client))
    applying = asyncio.create_task(worker.apply_pending(apply))
    await asyncio.wait_for(loaded.wait(), 10)
    takeover = asyncio.create_task(repository.take_over_batch({**initial, "spend": 1.0}))
    try:
        await asyncio.sleep(0.1)
        assert not takeover.done()
    finally:
        release.set()
        result, taken = await asyncio.gather(applying, takeover)
    assert result.applied == 1 and taken.charged
    assert (await worker.apply_pending(apply)).applied == 1
    assert not (await repository.take_over_batch({**initial, "spend": 1.0})).charged
    assert (await worker.apply_pending(apply)).applied == 0
    assert delivery_database.execute("SELECT spend FROM totals").fetchone() == (1.0,)


def test_downgrade_requires_drained_buffers_disabled_writers_and_no_unsettled_rows(
    delivery_database: psycopg.Connection[tuple[object, ...]],
) -> None:
    import subprocess
    import sys

    url = urlunsplit(urlsplit(os.environ["DATABASE_URL"])._replace(path=f"/{delivery_database.info.dbname}"))
    command = [sys.executable, "scripts/check_spendlogs_ledger_downgrade.py"]
    environment = {**os.environ, "DATABASE_URL": url}
    assert subprocess.run(command, env=environment, capture_output=True).returncode == 1
    assert (
        subprocess.run(command + ["--all-writers-ledger-disabled"], env=environment, capture_output=True).returncode
        == 1
    )
    confirmed = command + ["--all-writers-ledger-disabled", "--ledger-buffers-drained"]
    _insert(delivery_database, (_row("pending"), _row("parked")))
    delivery_database.execute("UPDATE \"LiteLLM_AccountingDelivery\" SET state='parked' WHERE request_id='parked'")
    blocked = subprocess.run(confirmed, env=environment, capture_output=True, text=True)
    assert blocked.returncode == 1 and "1 pending and 1 parked" in blocked.stdout, blocked.stderr
    delivery_database.execute("UPDATE \"LiteLLM_AccountingDelivery\" SET state='applied'")
    passed = subprocess.run(confirmed, env=environment, capture_output=True, text=True)
    assert passed.returncode == 0 and "no pending or parked" in passed.stdout, passed.stderr
