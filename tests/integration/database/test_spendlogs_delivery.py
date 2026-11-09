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


def _insert(
    connection: psycopg.Connection[tuple[object, ...]], rows: Sequence[Mapping[str, object]], *, ledger: bool = True
) -> None:
    from psycopg.types.json import Jsonb

    statement: Final = spend_log_insert_statement(rows)
    with connection.transaction():
        inserted: Final = {
            row[0] for row in connection.execute(statement[0].replace("$1", "%s"), (statement[1],)).fetchall()
        }
        for row in rows:
            if ledger and row["request_id"] in inserted:
                connection.execute(
                    'INSERT INTO "LiteLLM_SpendLogDelivery" (delivery_id, request_id, accounting_payload, is_batch) '
                    "VALUES (%s, %s, %s, %s)",
                    (row["request_id"], row["request_id"], Jsonb(dict(row)), row.get("call_type") == "aretrieve_batch"),
                )


def _with_payload(row: Mapping[str, object]) -> Mapping[str, object]:
    return {**row, "accounting_payload": dict(row)}


def _row(request_id: str) -> Mapping[str, object]:
    return {
        "request_id": request_id,
        "call_type": "acompletion",
        "spend": 0.5,
        "startTime": "2026-11-02T12:00:00+00:00",
        "metadata": '{"accounting_savings":{"compression":0.2}}',
    }


@pytest.mark.parametrize("call_type", ["acompletion", "aretrieve_batch"])
@pytest.mark.parametrize("mode", ["local", "url", "disabled"])
@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.asyncio
async def test_delivery_first_acceptance_deduplicates_all_storage_modes(
    delivery_database, delivery_client, mode, partitioned, call_type
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
    row = {**_with_payload({**_row("shared"), "call_type": call_type}), "accounting_log_disabled": mode == "disabled"}
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
    assert delivery_database.execute('SELECT accounting_payload FROM "LiteLLM_SpendLogDelivery"').fetchone() == (
        first["accounting_payload"],
    )
    assert delivery_database.execute('SELECT spend FROM "LiteLLM_SpendLogs"').fetchall() == (
        [(first["spend"],)] if mode == "local" else []
    )


@pytest.mark.parametrize(("legacy_spend", "accepted"), [(0.5, 0), (0.0, 1)])
@pytest.mark.asyncio
async def test_batch_charged_before_the_ledger_is_not_charged_again(
    delivery_database, delivery_client, legacy_spend, accepted
):
    from types import SimpleNamespace

    from litellm.repositories.table_repositories import SpendLogsRepository

    batch: Final = {**_row("batch_legacy"), "call_type": "aretrieve_batch", "status": "success"}
    _insert(delivery_database, ({**batch, "spend": legacy_spend},), ledger=False)
    repository = SpendLogsRepository(SimpleNamespace(db=delivery_client, writer_db=delivery_client))
    assert await repository.insert_with_deliveries((_with_payload({**batch, "spend": 1.0}),)) == accepted
    assert delivery_database.execute('SELECT COUNT(*) FROM "LiteLLM_SpendLogDelivery"').fetchone() == (accepted,)
    assert delivery_database.execute('SELECT spend FROM "LiteLLM_SpendLogs"').fetchall() == [(legacy_spend,)]


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


@pytest.mark.asyncio
async def test_a_locked_database_never_parks_a_valid_delivery(delivery_database, delivery_client):
    _insert(delivery_database, (_row("valid"),))

    async def apply(transaction, deliveries):
        await transaction.execute_raw("SET LOCAL lock_timeout = '50ms'")
        await transaction.execute_raw(
            "UPDATE totals SET spend = spend + $1::float8", sum(float(row.row["spend"]) for row in deliveries)
        )

    repository = AccountingDeliveryRepository(delivery_client)
    with psycopg.connect(
        urlunsplit(urlsplit(os.environ["DATABASE_URL"])._replace(path=f"/{delivery_database.info.dbname}"))
    ) as locker:
        locker.execute("LOCK TABLE totals IN ACCESS EXCLUSIVE MODE")
        for _ in range(4):
            result = await repository.apply_pending(apply, max_attempts=2)
            assert result.failed == 1 and result.parked == ()
            delivery_database.execute('UPDATE "LiteLLM_SpendLogDelivery" SET next_attempt_at = CURRENT_TIMESTAMP')
    assert delivery_database.execute('SELECT state::text, attempts FROM "LiteLLM_SpendLogDelivery"').fetchall() == [
        ("pending", 4)
    ]
    assert (await repository.apply_pending(apply, max_attempts=2)).applied == 1
    assert delivery_database.execute("SELECT spend FROM totals").fetchone() == (0.5,)


@pytest.mark.asyncio
async def test_an_unreachable_database_reported_as_a_data_error_never_parks_a_delivery(
    delivery_database, delivery_client
):
    from prisma.errors import DataError

    _insert(delivery_database, (_row("valid"),))

    async def unreachable(transaction, deliveries):
        raise DataError(data={"user_facing_error": {"message": "Can't reach database server at `db`:`5432`"}})

    repository = AccountingDeliveryRepository(delivery_client)
    for _ in range(3):
        assert (await repository.apply_pending(unreachable, max_attempts=2)).parked == ()
        delivery_database.execute('UPDATE "LiteLLM_SpendLogDelivery" SET next_attempt_at = CURRENT_TIMESTAMP')
    assert delivery_database.execute('SELECT state::text, attempts FROM "LiteLLM_SpendLogDelivery"').fetchall() == [
        ("pending", 3)
    ]


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
    backlog = await repository.backlog()
    assert backlog.pending_count == 1
    assert backlog.parked_count == 1
    assert backlog.oldest_pending_seconds >= 40 * 86400
    assert delivery_database.execute(
        'SELECT request_id FROM "LiteLLM_SpendLogDelivery" ORDER BY request_id'
    ).fetchall() == [("parked",), ("pending",)]


@pytest.mark.asyncio
async def test_backlog_reads_only_unapplied_deliveries_however_many_receipts_are_retained(
    delivery_database: psycopg.Connection[tuple[object, ...]], delivery_client: Prisma
) -> None:
    from litellm.repositories.accounting_delivery_repository import _BACKLOG

    delivery_database.execute(
        'INSERT INTO "LiteLLM_SpendLogDelivery" (delivery_id, request_id, accounting_payload, is_batch, state, applied_at) '
        "SELECT 'applied-' || n, 'applied-' || n, jsonb_build_object('payload', repeat(md5(n::text), 32)), "
        "n % 10 = 0, 'applied', CURRENT_TIMESTAMP FROM generate_series(1, 20000) AS n"
    )
    _insert(delivery_database, tuple(_row(state) for state in ("pending", "parked")))
    delivery_database.execute(
        'UPDATE "LiteLLM_SpendLogDelivery" SET state = request_id::"AccountingDeliveryState" '
        "WHERE request_id IN ('pending', 'parked')"
    )
    delivery_database.execute('ANALYZE "LiteLLM_SpendLogDelivery"')
    explained = delivery_database.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {_BACKLOG}").fetchone()
    assert explained is not None
    plan = explained[0][0]["Plan"]
    table_pages = delivery_database.execute(
        "SELECT relpages FROM pg_class WHERE oid = '\"LiteLLM_SpendLogDelivery\"'::regclass"
    ).fetchone()
    assert table_pages is not None and table_pages[0] > 1000
    assert plan["Shared Hit Blocks"] + plan["Shared Read Blocks"] < table_pages[0] / 20

    backlog = await AccountingDeliveryRepository(delivery_client).backlog()
    assert (backlog.pending_count, backlog.parked_count) == (1, 1)


@pytest.mark.parametrize("delivery_database", ["public", "ledger_tenant"], indirect=True)
@pytest.mark.parametrize("state", [None, "pending", "parked", "applied"])
def test_delivery_migration_replay_preserves_ledger_receipts(delivery_database, state):
    if state is not None:
        _insert(delivery_database, (_row("replay"),))
        delivery_database.execute(
            'UPDATE "LiteLLM_SpendLogDelivery" SET state=%s, attempts=3, last_error=%s',
            (state, "retained failure"),
        )
    before = delivery_database.execute('SELECT * FROM "LiteLLM_SpendLogDelivery"').fetchall()
    migration = (_MIGRATIONS / "20261103010000_spendlogs_accounting_delivery/migration.sql").read_text()
    delivery_database.execute(migration)
    delivery_database.execute(migration)
    assert delivery_database.execute('SELECT * FROM "LiteLLM_SpendLogDelivery"').fetchall() == before
    assert delivery_database.execute('SELECT count(*) FROM "LiteLLM_SpendLogs"').fetchone() == (int(state is not None),)


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


@pytest.mark.parametrize(
    ("payload", "huge_state"),
    [
        (
            "jsonb_set(accounting_payload, '{metadata}', ('{\"tokens_saved\": ' || repeat('9', 401) || '}')::jsonb)",
            "applied",
        ),
        (
            "jsonb_set(accounting_payload, '{metadata}', ('{\"tokens_saved\": ' || repeat('9', 5000) || '}')::jsonb)",
            "parked",
        ),
        ("'[]'::jsonb", "parked"),
    ],
    ids=["beyond_float", "beyond_python_int_parse", "not_an_object"],
)
@pytest.mark.asyncio
async def test_a_payload_the_engine_cannot_decode_never_blocks_other_deliveries(
    delivery_database, delivery_client, payload, huge_state
):
    _insert(delivery_database, (_row("huge"), _row("good")))
    delivery_database.execute(
        f'UPDATE "LiteLLM_SpendLogDelivery" SET accounting_payload = {payload} WHERE request_id = $$huge$$'
    )

    async def apply(transaction, deliveries):
        await transaction.execute_raw(
            "UPDATE totals SET spend = spend + $1::float8", sum(float(row.row["spend"]) for row in deliveries)
        )

    repository = AccountingDeliveryRepository(delivery_client)
    results = [await repository.apply_pending(apply, max_attempts=1) for _ in range(3)]
    _insert(delivery_database, (_row("later"),))
    results.append(await repository.apply_pending(apply, max_attempts=1))

    applied: Final = 3 if huge_state == "applied" else 2
    assert sum(result.applied for result in results) == applied
    assert delivery_database.execute("SELECT spend FROM totals").fetchone() == (0.5 * applied,)
    assert delivery_database.execute(
        'SELECT request_id, state::text FROM "LiteLLM_SpendLogDelivery" ORDER BY request_id'
    ).fetchall() == [("good", "applied"), ("huge", huge_state), ("later", "applied")]


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
    storage = await repository.storage()
    assert storage.batch_count == 1
    assert storage.table_bytes > 0


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.asyncio
async def test_ledger_batch_retrieves_charge_once_in_every_logging_mode(
    delivery_database, delivery_client, monkeypatch, disabled
):
    from types import SimpleNamespace

    from litellm.proxy import proxy_server
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
    from litellm.proxy.utils import jsonify_object

    completed = {
        **_row("batch_completed"),
        "call_type": "aretrieve_batch",
        "status": "success",
        "spend": 0.5,
        "metadata": '{"status":"success","private":"request detail"}',
        "accounting_payload": {
            "request_id": "batch_completed",
            "call_type": "aretrieve_batch",
            "status": "success",
            "spend": 0.5,
            "metadata": {"accounting_savings": {"compression": 0.2}},
        },
    }
    monkeypatch.setattr(proxy_server, "general_settings", {"spend_logs_metadata_fields": {"include": ["status"]}})
    client = SimpleNamespace(db=delivery_client, writer_db=delivery_client, jsonify_object=jsonify_object)
    writer = DBSpendUpdateWriter()
    first = await writer._claim_settle_once_cost_row(
        payload=completed, prisma_client=client, disable_spend_logs=disabled
    )
    again = await writer._claim_settle_once_cost_row(
        payload={**completed, "startTime": "2026-11-05T12:00:00+00:00"},
        prisma_client=client,
        disable_spend_logs=disabled,
    )
    assert (first, again) == (True, False)
    assert delivery_database.execute('SELECT spend, metadata FROM "LiteLLM_SpendLogs"').fetchall() == (
        [] if disabled else [(0.5, {"status": "success"})]
    )
    assert delivery_database.execute(
        'SELECT delivery_id, accounting_payload FROM "LiteLLM_SpendLogDelivery"'
    ).fetchall() == [("batch_completed", completed["accounting_payload"])]


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
