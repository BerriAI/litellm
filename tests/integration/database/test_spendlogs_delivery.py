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
def delivery_database(request: pytest.FixtureRequest) -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    with scratch_database("spendlogs_delivery") as url, psycopg.connect(url, autocommit=True) as connection:
        if getattr(request, "param", "public") == "ledger_tenant":
            connection.execute("CREATE SCHEMA ledger_tenant")
            connection.execute("SET search_path TO ledger_tenant")
        connection.execute((_MIGRATIONS / "20261027000000_add_accounting_ledger_tables/migration.sql").read_text())
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
    from psycopg.types.json import Jsonb

    statement: Final = spend_log_insert_statement(rows)
    with connection.transaction():
        inserted: Final = {
            row[0] for row in connection.execute(statement[0].replace("$1", "%s"), (statement[1],)).fetchall()
        }
        for row in rows:
            if row["request_id"] in inserted and row.get("accounting_path") == "ledger":
                connection.execute(
                    'INSERT INTO "LiteLLM_SpendLogDelivery" (delivery_id, request_id, accounting_payload, is_batch) '
                    "VALUES (%s, %s, %s, %s)",
                    (row["request_id"], row["request_id"], Jsonb(dict(row)), row.get("call_type") == "aretrieve_batch"),
                )


def _with_payload(row: Mapping[str, object]) -> Mapping[str, object]:
    return {**row, "accounting_payload": dict(row)}


def _row(request_id: str, path: str = "ledger") -> Mapping[str, object]:
    return {
        "request_id": request_id,
        "call_type": "acompletion",
        "spend": 0.5,
        "startTime": "2026-11-02T12:00:00+00:00",
        "metadata": '{"accounting_savings":{"compression":0.2}}',
        "accounting_path": path,
    }


@pytest.mark.parametrize("mode", ["local", "url", "disabled"])
@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.asyncio
async def test_delivery_first_acceptance_deduplicates_all_storage_modes(
    delivery_database, delivery_client, mode, partitioned
):
    from types import SimpleNamespace

    from litellm.repositories.table_repositories import SpendLogsRepository

    if partitioned:
        delivery_database.execute('ALTER TABLE "LiteLLM_SpendLogs" RENAME TO spend_template')
        delivery_database.execute(
            'CREATE TABLE "LiteLLM_SpendLogs" (LIKE spend_template INCLUDING DEFAULTS INCLUDING CONSTRAINTS, PRIMARY KEY(request_id, "startTime")) PARTITION BY RANGE ("startTime")'
        )
        delivery_database.execute('CREATE TABLE "LiteLLM_SpendLogs_p20261102" PARTITION OF "LiteLLM_SpendLogs" DEFAULT')
        delivery_database.execute("DROP TABLE spend_template")
    repository = SpendLogsRepository(SimpleNamespace(db=delivery_client, writer_db=delivery_client))
    row = {**_with_payload(_row("shared")), "accounting_log_disabled": mode == "disabled"}
    results = await asyncio.gather(
        repository.insert_with_deliveries((row,), write_spend_logs=mode != "url"),
        repository.insert_with_deliveries(
            ({**row, "startTime": "2026-11-03T12:00:00"},), write_spend_logs=mode != "url"
        ),
    )
    assert sorted(results) == [0, 1]
    assert delivery_database.execute('SELECT COUNT(*) FROM "LiteLLM_SpendLogs"').fetchone() == (int(mode == "local"),)
    assert delivery_database.execute(
        "SELECT accounting_payload->>'spend', state::text FROM \"LiteLLM_SpendLogDelivery\""
    ).fetchall() == [("0.5", "pending")]
    if partitioned:
        delivery_database.execute('DROP TABLE "LiteLLM_SpendLogs_p20261102"')
    else:
        delivery_database.execute('DELETE FROM "LiteLLM_SpendLogs"')

    async def apply(transaction, deliveries):
        await transaction.execute_raw("UPDATE totals SET spend = spend + $1::float8", float(deliveries[0].row["spend"]))

    worker = AccountingDeliveryRepository(delivery_client)
    assert (await worker.apply_pending(apply)).applied == 1
    assert (await worker.apply_pending(apply)).applied == 0
    assert delivery_database.execute("SELECT spend FROM totals").fetchone() == (0.5,)


@pytest.mark.parametrize("mode", ["local", "url", "disabled"])
@pytest.mark.asyncio
async def test_duplicate_rows_in_one_flush_keep_first_accounting_payload(delivery_database, delivery_client, mode):
    from types import SimpleNamespace

    from litellm.repositories.table_repositories import SpendLogsRepository

    repository = SpendLogsRepository(SimpleNamespace(db=delivery_client, writer_db=delivery_client))
    first = {**_with_payload(_row("repeated")), "accounting_log_disabled": mode == "disabled"}
    later = {
        **_with_payload({**_row("repeated"), "spend": 9.0, "startTime": "2026-11-03T12:00:00+00:00"}),
        "accounting_log_disabled": mode == "disabled",
    }
    assert await repository.insert_with_deliveries((first, later), write_spend_logs=mode != "url") == 1
    assert delivery_database.execute(
        'SELECT accounting_payload FROM "LiteLLM_SpendLogDelivery"'
    ).fetchone() == (first["accounting_payload"],)
    assert delivery_database.execute('SELECT spend FROM "LiteLLM_SpendLogs"').fetchall() == (
        [(first["spend"],)] if mode == "local" else []
    )


@pytest.mark.asyncio
async def test_existing_local_charge_rolls_back_new_delivery(delivery_database, delivery_client):
    from types import SimpleNamespace

    from litellm.repositories.table_repositories import SpendLogsRepository

    _insert(delivery_database, (_row("legacy", "legacy"),))
    repository = SpendLogsRepository(SimpleNamespace(db=delivery_client, writer_db=delivery_client))
    assert await repository.insert_with_deliveries((_with_payload(_row("legacy")),)) == 0
    assert delivery_database.execute('SELECT COUNT(*) FROM "LiteLLM_SpendLogDelivery"').fetchone() == (0,)
    assert delivery_database.execute('SELECT spend, accounting_path FROM "LiteLLM_SpendLogs"').fetchall() == [
        (0.5, "legacy")
    ]


@pytest.mark.parametrize("table", ["LiteLLM_SpendLogs", "LiteLLM_SpendLogDelivery"])
@pytest.mark.asyncio
async def test_transaction_a_rolls_back_payload_and_log_on_failure(delivery_database, delivery_client, table):
    from types import SimpleNamespace

    from prisma.errors import PrismaError
    from psycopg import sql

    from litellm.repositories.table_repositories import SpendLogsRepository

    delivery_database.execute(
        sql.SQL("ALTER TABLE {} ADD CONSTRAINT reject_poison CHECK (request_id <> 'poison')").format(
            sql.Identifier(table)
        )
    )
    repository = SpendLogsRepository(SimpleNamespace(db=delivery_client, writer_db=delivery_client))
    with pytest.raises(PrismaError):
        await repository.insert_with_deliveries(tuple(_with_payload(_row(name)) for name in ("good", "poison")))
    assert delivery_database.execute('SELECT COUNT(*) FROM "LiteLLM_SpendLogDelivery"').fetchone() == (0,)
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
            'SELECT DISTINCT state::text, attempts FROM "LiteLLM_SpendLogDelivery"'
        ).fetchall() == [("pending", 1)]
        delivery_database.execute('UPDATE "LiteLLM_SpendLogDelivery" SET next_attempt_at = CURRENT_TIMESTAMP')
        isolated: Final = await repository.apply_pending(apply, max_attempts=2)
        assert isolated.parked == ("bad",)
        applied: Final = await repository.apply_pending(apply, max_attempts=2)
        assert applied.applied == 1
        assert delivery_database.execute("SELECT spend FROM totals").fetchone() == (0.5,)
        assert delivery_database.execute(
            'SELECT request_id, state::text FROM "LiteLLM_SpendLogDelivery" ORDER BY request_id'
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
        "SELECT COUNT(*) FROM \"LiteLLM_SpendLogDelivery\" WHERE state = 'applied'"
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
    assert delivery_database.execute('SELECT state::text, attempts FROM "LiteLLM_SpendLogDelivery"').fetchone() == (
        "pending",
        0,
    )

    async def complete(transaction: LedgerTransaction, rows: tuple[DeliveryProjection, ...]) -> None:
        await transaction.execute_raw("UPDATE totals SET spend = spend + $1::float8", float(str(rows[0].row["spend"])))

    assert (await repository.apply_pending(complete)).applied == 1
    assert (await repository.apply_pending(complete)).applied == 0
    assert delivery_database.execute("SELECT spend FROM totals").fetchone() == (0.5,)


@pytest.mark.parametrize("state", ["pending", "parked", "applied"])
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
        delivery_database.execute('DELETE FROM "LiteLLM_SpendLogDelivery"')
    else:
        delivery_database.execute(
            'UPDATE "LiteLLM_SpendLogDelivery" SET state = %s::"AccountingDeliveryState"', (state,)
        )
    repository: Final = SpendLogsRepository(SimpleNamespace(db=delivery_client, writer_db=delivery_client))
    taken: Final = await repository.take_over_batch(
        _with_payload({**initial, "spend": 1.0, "accounting_path": "legacy"})
    )
    assert taken.charged
    assert taken.accounting_path == "ledger"
    assert delivery_database.execute('SELECT spend, accounting_path FROM "LiteLLM_SpendLogs"').fetchone() == (
        1.0,
        "ledger",
    )
    receipt: Final = delivery_database.execute(
        'SELECT state::text, previous_payload FROM "LiteLLM_SpendLogDelivery"'
    ).fetchone()
    assert receipt is not None
    assert receipt[0] == ("parked" if state == "parked" else "pending")
    if state in {"applied", "missing"}:
        assert isinstance(receipt[1], dict)
        assert receipt[1]["spend"] == 0
        assert receipt[1]["accounting_path"] == "ledger"
    else:
        assert receipt[1] is None
    assert not (await repository.take_over_batch(_with_payload({**initial, "spend": 1.0}))).charged


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
    result: Final = await repository.take_over_batch(
        _with_payload({**initial, "spend": 0.5, "accounting_path": "ledger"})
    )
    assert not result.charged
    assert result.accounting_path == "legacy"
    assert delivery_database.execute('SELECT COUNT(*) FROM "LiteLLM_SpendLogDelivery"').fetchone() == (0,)
    assert delivery_database.execute('SELECT spend, accounting_path FROM "LiteLLM_SpendLogs"').fetchone() == (
        0.0,
        "legacy",
    )


@pytest.mark.asyncio
async def test_worker_applies_accounting_after_spend_logs_are_deleted(
    delivery_database: psycopg.Connection[tuple[object, ...]], delivery_client: Prisma
) -> None:
    _insert(delivery_database, (_row("removed"),))
    delivery_database.execute('DELETE FROM "LiteLLM_SpendLogs"')

    async def apply(transaction: LedgerTransaction, rows: tuple[DeliveryProjection, ...]) -> None:
        await transaction.execute_raw("UPDATE totals SET spend = spend + $1::float8", float(str(rows[0].row["spend"])))

    repository: Final = AccountingDeliveryRepository(delivery_client)
    assert (await repository.apply_pending(apply)).applied == 1
    assert (await repository.apply_pending(apply)).applied == 0
    assert delivery_database.execute("SELECT spend FROM totals").fetchone() == (0.5,)


@pytest.mark.asyncio
async def test_receipt_cleanup_keeps_pending_and_parked_rows_and_reports_lag(
    delivery_database: psycopg.Connection[tuple[object, ...]], delivery_client: Prisma
) -> None:
    _insert(delivery_database, tuple(_row(state) for state in ("pending", "parked", "applied")))
    delivery_database.execute(
        'UPDATE "LiteLLM_SpendLogDelivery" '
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
        'SELECT request_id FROM "LiteLLM_SpendLogDelivery" ORDER BY request_id'
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
            sum(
                float(str(row.row["spend"])) - float(str((row.previous_payload or {}).get("spend", 0))) for row in rows
            ),
        )

    worker = AccountingDeliveryRepository(delivery_client)
    repository = SpendLogsRepository(SimpleNamespace(db=delivery_client, writer_db=delivery_client))
    applying = asyncio.create_task(worker.apply_pending(apply))
    await asyncio.wait_for(loaded.wait(), 10)
    takeover = asyncio.create_task(repository.take_over_batch(_with_payload({**initial, "spend": 1.0})))
    try:
        await asyncio.sleep(0.1)
        assert not takeover.done()
    finally:
        release.set()
        result, taken = await asyncio.gather(applying, takeover)
    assert result.applied == 1 and taken.charged
    assert (await worker.apply_pending(apply)).applied == 1
    assert not (await repository.take_over_batch(_with_payload({**initial, "spend": 1.0}))).charged
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
    delivery_database.execute("UPDATE \"LiteLLM_SpendLogDelivery\" SET state='parked' WHERE request_id='parked'")
    blocked = subprocess.run(confirmed, env=environment, capture_output=True, text=True)
    assert blocked.returncode == 1 and "1 pending and 1 parked" in blocked.stdout, blocked.stderr
    delivery_database.execute("UPDATE \"LiteLLM_SpendLogDelivery\" SET state='applied'")
    passed = subprocess.run(confirmed, env=environment, capture_output=True, text=True)
    assert passed.returncode == 0 and "no pending or parked" in passed.stdout, passed.stderr


@pytest.mark.parametrize("delivery_database", ["public", "ledger_tenant"], indirect=True)
@pytest.mark.parametrize("state", [None, "pending", "parked", "applied"])
def test_delivery_migration_replay_preserves_ledger_receipts(delivery_database, state):
    if state is not None:
        _insert(delivery_database, (_row("replay"),))
        delivery_database.execute(
            'UPDATE "LiteLLM_SpendLogDelivery" SET state=%s, attempts=3, last_error=%s, previous_payload=%s',
            (state, "retained failure", '{"spend": 0}'),
        )
    before = delivery_database.execute('SELECT * FROM "LiteLLM_SpendLogDelivery"').fetchall()
    migration = (_MIGRATIONS / "20261103010000_spendlogs_accounting_delivery/migration.sql").read_text()
    delivery_database.execute(migration)
    delivery_database.execute(migration)
    assert delivery_database.execute('SELECT * FROM "LiteLLM_SpendLogDelivery"').fetchall() == before
    assert delivery_database.execute('SELECT count(*) FROM "LiteLLM_SpendLogs"').fetchone() == (int(state is not None),)
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert(delivery_database, (_row("invalid", "invalid"),))


@pytest.mark.parametrize("state", ["pending", "parked"])
def test_delivery_migration_preserves_unused_old_receipts(state):
    with scratch_database("spendlogs_old_delivery") as url, psycopg.connect(url, autocommit=True) as connection:
        connection.execute((_MIGRATIONS / "20261027000000_add_accounting_ledger_tables/migration.sql").read_text())
        connection.execute('CREATE TABLE "LiteLLM_SpendLogs" (request_id text PRIMARY KEY)')
        connection.execute(
            'INSERT INTO "LiteLLM_AccountingDelivery" (event_id, target, version, state) VALUES (%s, %s, 1, %s)',
            ("old-event", "LiteLLM_UserTable", state),
        )
        connection.execute((_MIGRATIONS / "20261103010000_spendlogs_accounting_delivery/migration.sql").read_text())
        assert connection.execute('SELECT event_id, state::text FROM "LiteLLM_AccountingDelivery"').fetchall() == [
            ("old-event", state)
        ]
        assert connection.execute('SELECT COUNT(*) FROM "LiteLLM_SpendLogDelivery"').fetchone() == (0,)


@pytest.mark.parametrize("column", ["accounting_payload", "previous_payload"])
@pytest.mark.asyncio
async def test_malformed_payload_is_parked_without_losing_other_deliveries(delivery_database, delivery_client, column):
    _insert(delivery_database, (_row("bad"), _row("good")))
    from psycopg import sql

    delivery_database.execute(
        sql.SQL('UPDATE "LiteLLM_SpendLogDelivery" SET {}=$$ [] $$::jsonb WHERE request_id=$$bad$$').format(
            sql.Identifier(column)
        )
    )

    async def apply(transaction, deliveries):
        await transaction.execute_raw(
            "UPDATE totals SET spend = spend + $1::float8", sum(float(row.row["spend"]) for row in deliveries)
        )

    repository = AccountingDeliveryRepository(delivery_client)
    assert (await repository.apply_pending(apply, max_attempts=2)).failed == 2
    assert delivery_database.execute("SELECT spend FROM totals").fetchone() == (0.0,)
    delivery_database.execute('UPDATE "LiteLLM_SpendLogDelivery" SET next_attempt_at=CURRENT_TIMESTAMP')
    assert (await repository.apply_pending(apply, max_attempts=2)).parked == ("bad",)
    assert (await repository.apply_pending(apply, max_attempts=2)).applied == 1
    assert delivery_database.execute("SELECT spend FROM totals").fetchone() == (0.5,)


@pytest.mark.asyncio
async def test_retention_keeps_batch_receipts_indefinitely(delivery_database, delivery_client):
    _insert(delivery_database, (_row("normal"), {**_row("batch"), "call_type": "aretrieve_batch"}))
    delivery_database.execute(
        "UPDATE \"LiteLLM_SpendLogDelivery\" SET state='applied', applied_at=CURRENT_TIMESTAMP - INTERVAL '400 days'"
    )
    repository = AccountingDeliveryRepository(delivery_client)
    assert await repository.cleanup_applied() == 1
    assert await repository.cleanup_applied() == 0
    assert delivery_database.execute('SELECT request_id FROM "LiteLLM_SpendLogDelivery"').fetchall() == [("batch",)]
    metrics = await repository.metrics()
    assert metrics.batch_count == 1
    assert metrics.table_bytes > 0


@pytest.mark.parametrize("call_type,status", [("acompletion", "success"), ("aretrieve_batch", "failure")])
@pytest.mark.asyncio
async def test_batch_takeover_preserves_another_request_receipt(delivery_database, delivery_client, call_type, status):
    from types import SimpleNamespace

    from litellm.repositories.table_repositories import SpendLogsRepository

    initial = {**_row("collision"), "call_type": call_type, "status": status, "spend": 0.0}
    _insert(delivery_database, (initial,))
    before = delivery_database.execute('SELECT accounting_payload FROM "LiteLLM_SpendLogDelivery"').fetchone()
    repository = SpendLogsRepository(SimpleNamespace(db=delivery_client, writer_db=delivery_client))
    replacement = {**initial, "call_type": "aretrieve_batch", "status": "success", "spend": 1.0}
    assert not (await repository.take_over_batch(_with_payload(replacement))).charged
    assert delivery_database.execute('SELECT accounting_payload FROM "LiteLLM_SpendLogDelivery"').fetchone() == before
    assert delivery_database.execute('SELECT spend FROM "LiteLLM_SpendLogs"').fetchone() == (0.0,)


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.asyncio
async def test_batch_takeover_respects_current_request_log_retention(
    delivery_database, delivery_client, monkeypatch, disabled
):
    from types import SimpleNamespace

    from litellm.proxy import proxy_server
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

    initial = {
        **_row("retained-batch"), "call_type": "aretrieve_batch", "status": "success", "spend": 0.0,
        "metadata": '{"status":"success","original":"retained"}',
    }
    _insert(delivery_database, (initial,))
    replacement = {
        **initial, "spend": 0.5, "metadata": '{"status":"success","private":"new request detail"}',
        "accounting_payload": {
            "request_id": "retained-batch", "call_type": "aretrieve_batch", "status": "success", "spend": 0.5,
            "metadata": {"accounting_savings": {"compression": 0.2}},
        },
    }
    monkeypatch.setattr(proxy_server, "general_settings", {
        "spend_logs_metadata_fields": {"include": ["status"]},
    })
    client = SimpleNamespace(db=delivery_client, writer_db=delivery_client)
    assert await DBSpendUpdateWriter()._claim_batch_cost_spend_log(
        payload=replacement, prisma_client=client, disable_spend_logs=disabled,
    )
    expected_metadata = {"status": "success", "original": "retained"} if disabled else {"status": "success"}
    assert delivery_database.execute('SELECT spend, metadata FROM "LiteLLM_SpendLogs"').fetchone() == (
        0.5, expected_metadata,
    )
    assert delivery_database.execute(
        'SELECT accounting_payload FROM "LiteLLM_SpendLogDelivery"'
    ).fetchone() == (replacement["accounting_payload"],)


@pytest.mark.parametrize("setting", ["retention_days", "batch_size"])
@pytest.mark.asyncio
async def test_invalid_cleanup_settings_preserve_receipts(delivery_database, delivery_client, setting):
    _insert(delivery_database, (_row("protected"),))
    with pytest.raises(ValueError, match="positive"):
        await AccountingDeliveryRepository(delivery_client).cleanup_applied(**{setting: 0})
    assert delivery_database.execute('SELECT request_id, state::text FROM "LiteLLM_SpendLogDelivery"').fetchall() == [
        ("protected", "pending")
    ]


@pytest.mark.parametrize("setting", ["max_attempts", "batch_size"])
@pytest.mark.asyncio
async def test_invalid_worker_settings_preserve_pending_work(delivery_database, delivery_client, setting):
    from unittest.mock import AsyncMock

    _insert(delivery_database, (_row("protected"),))
    apply = AsyncMock()
    with pytest.raises(ValueError, match="positive"):
        await AccountingDeliveryRepository(delivery_client).apply_pending(apply, **{setting: 0})
    apply.assert_not_awaited()
    assert delivery_database.execute(
        'SELECT request_id, state::text, attempts FROM "LiteLLM_SpendLogDelivery"'
    ).fetchall() == [("protected", "pending", 0)]


@pytest.mark.parametrize("flags,state,expected", [
    ((), "applied", 1),
    (("--all-writers-ledger-disabled",), "applied", 1),
    (("--ledger-buffers-drained",), "applied", 1),
    (("--all-writers-ledger-disabled", "--ledger-buffers-drained"), "pending", 1),
    (("--all-writers-ledger-disabled", "--ledger-buffers-drained"), "parked", 1),
    (("--all-writers-ledger-disabled", "--ledger-buffers-drained"), "applied", 0),
])
def test_downgrade_command_requires_operator_attestations_and_settled_deliveries(
    delivery_database, monkeypatch, capfd, flags, state, expected
):
    import runpy
    import sys

    _insert(delivery_database, (_row("downgrade"),))
    delivery_database.execute('UPDATE "LiteLLM_SpendLogDelivery" SET state=%s', (state,))
    url = urlunsplit(urlsplit(os.environ["DATABASE_URL"])._replace(path=f"/{delivery_database.info.dbname}"))
    monkeypatch.setenv("DATABASE_URL", url)
    script = Path(__file__).resolve().parents[3] / "scripts/check_spendlogs_ledger_downgrade.py"
    monkeypatch.setattr(sys, "argv", [str(script), *flags])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(script), run_name="__main__")
    assert result.value.code == expected
    prefix = "Database gate passed:" if expected == 0 else "Blocked:"
    assert any(line.startswith(prefix) for line in capfd.readouterr().out.splitlines())
    assert delivery_database.execute('SELECT request_id, state::text FROM "LiteLLM_SpendLogDelivery"').fetchall() == [
        ("downgrade", state)
    ]
