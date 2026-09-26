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
            setup.execute(
                sql.SQL('CREATE TABLE {} (request_id text, "startTime" timestamp NOT NULL)').format(table)
            )
            setup.execute(
                sql.SQL('INSERT INTO {} VALUES (%s, %s), (%s, %s), (%s, %s), (%s, %s)').format(table),
                ("expired-1", now - timedelta(days=60), "expired-2", now - timedelta(days=60),
                 "expired-3", now - timedelta(days=60), "fresh", now),
            )
            database: Final = Prisma(datasource={"url": scoped_url})
            await database.connect()
            try:
                cleanup: Final = SpendLogCleanup(
                    general_settings={"maximum_spend_logs_retention_period": "30d"}
                )
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
