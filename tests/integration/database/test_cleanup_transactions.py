import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
from integration._support.database import read_rows
from prisma import Prisma
from psycopg import sql

from litellm.proxy.db.db_transaction_queue.spend_log_cleanup import SpendLogCleanup


@dataclass(frozen=True)
class PartitionConnection:
    db: Prisma


async def test_delete_batch_survives_witnessed_lock_past_transaction_default(caplog: pytest.LogCaptureFixture) -> None:
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


async def test_size_rule_deletes_the_oldest_rows_in_steps_until_the_live_estimate_is_under_the_target(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Deleting rows never shrinks the table file, so the size pass must stop on the live-bytes estimate from
    pg_stat_user_tables, not on pg_total_relation_size, and it must take the oldest rows first and keep the
    tool index on the same cutoff.
    """
    caplog.set_level(logging.INFO, logger="LiteLLM Proxy")
    schema: Final = f"integration_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    parsed: Final = urlsplit(url)
    scoped_url: Final = urlunsplit(
        parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": schema}))
    )
    logs: Final = sql.Identifier(schema, "LiteLLM_SpendLogs")
    index: Final = sql.Identifier(schema, "LiteLLM_SpendLogToolIndex")
    seeded: Final = 6000
    now: Final = datetime.now(timezone.utc).replace(tzinfo=None)
    stats_sql: Final = (
        "SELECT sum(n_live_tup)::int AS live, sum(n_dead_tup)::int AS dead FROM pg_stat_user_tables "
        "WHERE schemaname = %s AND relname IN ('LiteLLM_SpendLogs', 'LiteLLM_SpendLogToolIndex')"
    )
    with psycopg.connect(url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            setup.execute(
                sql.SQL(
                    'CREATE TABLE {} (request_id text, "startTime" timestamp NOT NULL, payload text, '
                    'PRIMARY KEY (request_id, "startTime"))'
                ).format(logs)
            )
            setup.execute(
                sql.SQL(
                    "CREATE TABLE {} (request_id text, tool_name text, start_time timestamp NOT NULL, "
                    "PRIMARY KEY (request_id, tool_name))"
                ).format(index)
            )
            setup.execute(
                sql.SQL(
                    "INSERT INTO {} SELECT 'req-' || lpad(i::text, 6, '0'), %s - (%s - i) * interval '1 minute', "
                    "repeat('x', 1000) FROM generate_series(1, %s) AS i"
                ).format(logs),
                (now, seeded, seeded),
            )
            setup.execute(
                sql.SQL(
                    "INSERT INTO {} SELECT request_id, 'tool', \"startTime\" FROM {} WHERE right(request_id, 1) = '0'"
                ).format(index, logs)
            )
            setup.execute(sql.SQL("ANALYZE {}").format(logs))
            setup.execute(sql.SQL("ANALYZE {}").format(index))
            index_rows: Final = seeded // 10
            stats_deadline: Final = time.monotonic() + 10
            while True:
                (stats,) = read_rows(stats_sql, (schema,))
                if stats["live"] == seeded + index_rows:
                    break
                assert time.monotonic() < stats_deadline, f"row statistics never caught up with ANALYZE: {stats}"
                await asyncio.sleep(0.1)
            (allocated,) = read_rows(
                "SELECT (pg_total_relation_size(%s::regclass) + pg_total_relation_size(%s::regclass))::bigint "
                "AS allocated",
                (logs.as_string(setup), index.as_string(setup)),
            )
            allocated_before: Final = int(str(allocated["allocated"]))
            assert allocated_before > 0

            database: Final = Prisma(datasource={"url": scoped_url})
            await database.connect()
            try:
                cleanup: Final = SpendLogCleanup(
                    general_settings={
                        # trigger below what is there, clean down to half of it: three steps of
                        # 10 batches x 100 rows take out the oldest 3000 of the 6000 rows
                        "maximum_spend_logs_size": str(allocated_before * 3 // 4),
                        "maximum_spend_logs_cleanup_target_size": str(allocated_before // 2),
                        "maximum_spend_logs_cleanup_batch_size": 100,
                    }
                )
                cleanup.pod_lock_manager = None
                await cleanup.cleanup_spend_logs_by_size(PartitionConnection(database))
            finally:
                await database.disconnect()

            remaining: Final = read_rows(
                f'SELECT request_id FROM {logs.as_string(setup)} ORDER BY "startTime"',
                (),
            )
            # oldest first, in whole steps of 1000 rows; the row counters can lag one step behind the
            # deletes before the stats collector flushes, which only ever costs one extra step
            assert len(remaining) in (3000, 2000), len(remaining)
            assert remaining[0]["request_id"] == f"req-{seeded - len(remaining) + 1:06d}"
            assert remaining[-1]["request_id"] == f"req-{seeded:06d}"
            (orphans,) = read_rows(
                f"SELECT count(*)::int AS n FROM {index.as_string(setup)} i "
                f"WHERE NOT EXISTS (SELECT 1 FROM {logs.as_string(setup)} l WHERE l.request_id = i.request_id)",
                (),
            )
            assert orphans["n"] == 0
            (kept_index,) = read_rows(f"SELECT count(*)::int AS n FROM {index.as_string(setup)}", ())
            assert kept_index["n"] == len(remaining) // 10
            summary: Final = next(
                record.getMessage()
                for record in caplog.records
                if "Spend log cleanup run finished" in record.getMessage()
            )
            assert "target_reached=true" in summary
            assert f"allocated_before={allocated_before}" in summary
            # the file did not shrink: that is the whole reason the estimate exists
            assert f"allocated_after={allocated_before}" in summary
            assert "VACUUM FULL or pg_repack" in summary
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
