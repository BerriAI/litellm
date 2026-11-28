import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
from integration._support.database import read_rows
from psycopg import sql

from litellm.proxy.db.db_transaction_queue.spend_log_cleanup import SpendLogCleanup

if TYPE_CHECKING:
    from prisma import Prisma


@dataclass(frozen=True)
class PartitionConnection:
    db: "Prisma"


async def test_delete_batch_survives_witnessed_lock_past_transaction_default(caplog: pytest.LogCaptureFixture) -> None:
    from prisma import Prisma

    caplog.set_level(logging.ERROR)
    schema: Final = f"integration_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    parsed: Final = urlsplit(url)
    scoped_url: Final = urlunsplit(
        parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": schema}))
    )
    table: Final = sql.Identifier(schema, "LiteLLM_SpendLogs")
    now: Final = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff_date: Final = now - timedelta(days=30)
    with psycopg.connect(url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            setup.execute(sql.SQL('CREATE TABLE {} (request_id text, "startTime" timestamp NOT NULL)').format(table))
            setup.execute(
                sql.SQL("INSERT INTO {} VALUES (%s, %s), (%s, %s), (%s, %s), (%s, %s)").format(table),
                (
                    "expired-1",
                    now - timedelta(days=60),
                    "expired-2",
                    now - timedelta(days=60),
                    "expired-3",
                    now - timedelta(days=60),
                    "fresh",
                    now,
                ),
            )
            database: Final = Prisma(datasource={"url": scoped_url})
            await database.connect()
            try:
                cleanup: Final = SpendLogCleanup(general_settings={"maximum_spend_logs_retention_period": "30d"})
                with psycopg.connect(url) as blocker:
                    blocker.execute(sql.SQL("LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE").format(table))
                    blocker_pid: Final = blocker.info.backend_pid
                    operation: Final = asyncio.create_task(
                        cleanup._delete_old_rows_batched(
                            PartitionConnection(database),
                            cutoff_date=cutoff_date,
                            table_name="LiteLLM_SpendLogs",
                            key_columns=("request_id",),
                            time_column="startTime",
                            deadline=time.monotonic() + 30,
                        )
                    )
                    wait_deadline: Final = time.monotonic() + 3
                    try:
                        while True:
                            witnesses: Final = read_rows(
                                "SELECT a.pid, extract(epoch FROM "
                                "clock_timestamp()-a.query_start)::double precision AS age "
                                "FROM pg_stat_activity a WHERE %s = ANY(pg_blocking_pids(a.pid)) "
                                "AND a.wait_event_type = 'Lock' AND a.query LIKE '%%DELETE FROM%%'",
                                (blocker_pid,),
                            )
                            if witnesses:
                                break
                            assert time.monotonic() < wait_deadline, "Cleanup DELETE never reached the held lock"
                            await asyncio.sleep(0.02)
                        assert len(witnesses) == 1
                        held_at: Final = time.monotonic()
                        age: Final = float(witnesses[0]["age"])
                        await asyncio.sleep(max(0, 5.6 - age))
                        held_seconds: Final = age + time.monotonic() - held_at
                        assert held_seconds >= 5.5, f"Lock released before the transaction boundary: {held_seconds}"
                        assert not operation.done(), "Cleanup DELETE completed while its required lock was held"
                    except BaseException:
                        operation.cancel()
                        await asyncio.gather(operation, return_exceptions=True)
                        raise
                    finally:
                        blocker.rollback()
                    result: Final = await asyncio.wait_for(operation, timeout=5)
                assert result.rows_deleted == 3
                assert not any("cleanup batch failed" in record.message for record in caplog.records)
                assert read_rows(
                    f"SELECT request_id FROM {table.as_string(setup)} ORDER BY request_id",
                    (),
                ) == [{"request_id": "fresh"}]
            finally:
                await database.disconnect()
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.parametrize(
    ("imported", "partitioned"),
    [(False, False), (True, False), (True, True)],
    ids=["local-logs", "replicas-only", "older-replicas-with-local-partition"],
)
async def test_size_rule_deletes_the_oldest_rows_until_under_the_target_and_does_not_refire_after_vacuum(
    caplog: pytest.LogCaptureFixture,
    imported: bool,
    partitioned: bool,
) -> None:
    """
    Deleting rows never shrinks the table file, so the size pass must stop on live bytes, not on
    pg_total_relation_size, taking the oldest rows first and keeping the tool index on the same cutoff. Once
    VACUUM has turned the deleted rows into free space the catalog no longer shows them as dead, and the
    files still measure the same; the next check must read what the last run freed from its saved count and
    leave the table alone instead of calling the whole file live again.
    """
    from prisma import Prisma

    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    schema: Final = f"integration_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    parsed: Final = urlsplit(url)
    scoped_url: Final = urlunsplit(
        parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": schema}))
    )
    local_logs: Final = sql.Identifier(schema, "LiteLLM_SpendLogs")
    local_partition: Final = sql.Identifier(schema, "LiteLLM_SpendLogs_older")
    log_table: Final = "LiteLLM_ReplicatedSpendLogs" if imported else "LiteLLM_SpendLogs"
    logs: Final = sql.Identifier(schema, log_table)
    index: Final = sql.Identifier(schema, "LiteLLM_SpendLogToolIndex")
    config: Final = sql.Identifier(schema, "LiteLLM_Config")
    seeded: Final = 6000
    now: Final = datetime.now(timezone.utc).replace(tzinfo=None)
    local_day: Final = now.date() - timedelta(days=4)
    stats_sql: Final = (
        "SELECT sum(n_live_tup)::int AS live, sum(n_dead_tup)::int AS dead, sum(n_tup_ins)::int AS inserted "
        "FROM pg_stat_user_tables "
        "WHERE schemaname = %s AND relname IN (%s, 'LiteLLM_SpendLogToolIndex')"
    )
    with psycopg.connect(url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            setup.execute(
                sql.SQL(
                    'CREATE TABLE {} (request_id text, "startTime" timestamp NOT NULL, payload text, '
                    'PRIMARY KEY (request_id, "startTime")) '
                    "{}"
                ).format(
                    local_logs,
                    sql.SQL('PARTITION BY RANGE ("startTime")')
                    if partitioned
                    else sql.SQL("WITH (autovacuum_enabled = false, toast.autovacuum_enabled = false)"),
                )
            )
            if partitioned:
                setup.execute(
                    sql.SQL("CREATE TABLE {} PARTITION OF {} FOR VALUES FROM ({}) TO ({})").format(
                        local_partition, local_logs, sql.Literal(local_day), sql.Literal(local_day + timedelta(days=1))
                    )
                )
                setup.execute(
                    sql.SQL("INSERT INTO {} VALUES ('newer-local', %s, 'local')").format(local_logs),
                    (local_day,),
                )
            if imported:
                setup.execute(
                    sql.SQL(
                        'CREATE TABLE {} (request_id text, "startTime" timestamp NOT NULL, payload text, '
                        "source_region text DEFAULT 'us', "
                        'PRIMARY KEY (request_id, "startTime", source_region)) '
                        "WITH (autovacuum_enabled = false, toast.autovacuum_enabled = false)"
                    ).format(logs)
                )
            setup.execute(
                sql.SQL(
                    "CREATE TABLE {} (request_id text, tool_name text, start_time timestamp NOT NULL, "
                    "PRIMARY KEY (request_id, tool_name)) "
                    "WITH (autovacuum_enabled = false, toast.autovacuum_enabled = false)"
                ).format(index)
            )
            setup.execute(sql.SQL("CREATE TABLE {} (param_name text PRIMARY KEY, param_value jsonb)").format(config))
            setup.execute(
                sql.SQL(
                    'INSERT INTO {} (request_id, "startTime", payload) '
                    "SELECT 'req-' || lpad(i::text, 6, '0'), %s - (%s - i) * interval '1 minute', "
                    "repeat('x', 1000) FROM generate_series(1, %s) AS i"
                ).format(logs),
                (local_day - timedelta(days=1) if partitioned else now, seeded, seeded),
            )
            setup.execute(
                sql.SQL(
                    "INSERT INTO {} SELECT request_id, 'tool', \"startTime\" FROM {} WHERE right(request_id, 1) = '0'"
                ).format(index, local_logs)
            )
            index_rows: Final = 0 if imported else seeded // 10
            inserts_deadline: Final = time.monotonic() + 10
            while True:
                # a backend only ships its pending row counters when it runs another command, so give the
                # seeding connection one; the size rule rolls its count forward from n_tup_ins, which must
                # show the seed inserts before ANALYZE sets the live estimate or the two add up to double
                setup.execute("SELECT 1")
                (inserts,) = read_rows(stats_sql, (schema, log_table))
                if inserts["inserted"] == seeded + index_rows:
                    break
                assert time.monotonic() < inserts_deadline, f"row statistics never saw the seed inserts: {inserts}"
                await asyncio.sleep(0.1)
            setup.execute(sql.SQL("ANALYZE {}").format(logs))
            setup.execute(sql.SQL("ANALYZE {}").format(index))
            stats_deadline: Final = time.monotonic() + 10
            while True:
                setup.execute("SELECT 1")
                (stats,) = read_rows(stats_sql, (schema, log_table))
                if stats["live"] == seeded + index_rows:
                    break
                assert time.monotonic() < stats_deadline, f"row statistics never caught up with ANALYZE: {stats}"
                await asyncio.sleep(0.1)
            (allocated,) = read_rows(
                "SELECT (pg_total_relation_size(%s::regclass) + pg_total_relation_size(%s::regclass) "
                "+ CASE WHEN %s THEN pg_total_relation_size(%s::regclass) ELSE 0 END)::bigint "
                "AS allocated",
                (
                    logs.as_string(setup),
                    index.as_string(setup),
                    imported,
                    (local_partition if partitioned else local_logs).as_string(setup),
                ),
            )
            allocated_before: Final = int(str(allocated["allocated"]))
            assert allocated_before > 0

            cleanup: Final = SpendLogCleanup(
                general_settings={
                    # trigger below what is there, clean down to half of it
                    "maximum_spend_logs_size": str(allocated_before * 3 // 4),
                    "maximum_spend_logs_cleanup_target_size": str(allocated_before // 2),
                    "maximum_spend_logs_cleanup_batch_size": 100,
                    "use_spend_logs_partitioning": partitioned,
                }
            )

            async def size_check() -> None:
                database: Final = Prisma(datasource={"url": scoped_url})
                await database.connect()
                try:
                    await cleanup.cleanup_spend_logs_by_size(PartitionConnection(database))
                finally:
                    await database.disconnect()

            await size_check()

            local_rows: Final = (
                read_rows(f"SELECT request_id FROM {local_logs.as_string(setup)}", ()) if partitioned else []
            )
            assert local_rows == ([{"request_id": "newer-local"}] if partitioned else [])
            remaining: Final = read_rows(
                f'SELECT request_id FROM {logs.as_string(setup)} ORDER BY "startTime"',
                (),
            )
            # half the bytes is half the rows: the oldest 3000 spend logs and their 300 index rows go, counted
            # from the deletes themselves, so the stats collector's lag cannot make the run overshoot. A page the
            # files grew by between this measurement and the run's probe costs at most one more batch
            assert 2900 <= len(remaining) <= 3000, f"{len(remaining)} spend logs remain, expected 3000 less one batch"
            assert remaining[0]["request_id"] == f"req-{seeded - len(remaining) + 1:06d}"
            assert remaining[-1]["request_id"] == f"req-{seeded:06d}"
            (orphans,) = read_rows(
                f"SELECT count(*)::int AS n FROM {index.as_string(setup)} i "
                f"WHERE NOT EXISTS (SELECT 1 FROM {logs.as_string(setup)} l WHERE l.request_id = i.request_id)",
                (),
            )
            assert orphans["n"] == 0
            (kept_index,) = read_rows(f"SELECT count(*)::int AS n FROM {index.as_string(setup)}", ())
            assert kept_index["n"] == (0 if imported else len(remaining) // 10)
            summary: Final = next(
                record.getMessage()
                for record in caplog.records
                if "Spend log cleanup run finished" in record.getMessage()
            )
            assert "target_reached=true" in summary
            assert f"allocated_before={allocated_before}" in summary
            if not partitioned:
                # The partitioned case also precreates empty partitions, which adds allocated bytes.
                assert f"allocated_after={allocated_before}" in summary
            assert "VACUUM FULL or pg_repack" in summary

            deletes_deadline: Final = time.monotonic() + 10
            while True:
                (after_deletes,) = read_rows(stats_sql, (schema, log_table))
                if after_deletes["dead"] == seeded - len(remaining) + index_rows - kept_index["n"]:
                    break
                assert time.monotonic() < deletes_deadline, f"row statistics never saw the deletes: {after_deletes}"
                await asyncio.sleep(0.1)
            setup.execute(sql.SQL("VACUUM (ANALYZE) {}").format(logs))
            setup.execute(sql.SQL("VACUUM (ANALYZE) {}").format(index))
            vacuum_deadline: Final = time.monotonic() + 10
            while True:
                (after_vacuum,) = read_rows(stats_sql, (schema, log_table))
                if after_vacuum["dead"] == 0:
                    break
                assert time.monotonic() < vacuum_deadline, f"row statistics never caught up with VACUUM: {after_vacuum}"
                await asyncio.sleep(0.1)
            caplog.clear()

            await size_check()

            (still_there,) = read_rows(f"SELECT count(*)::int AS n FROM {logs.as_string(setup)}", ())
            assert still_there["n"] == len(remaining), (
                "the second check deleted rows the first one had already brought under the limit"
            )
            assert any("is under maximum_spend_logs_size" in record.getMessage() for record in caplog.records)
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
