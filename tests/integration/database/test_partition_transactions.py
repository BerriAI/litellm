import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
from psycopg import sql
from prisma import Prisma

from integration._support.database import read_rows
from litellm.proxy.db.db_transaction_queue.spend_logs_partition_manager import SpendLogsPartitionManager


@dataclass(frozen=True)
class PartitionConnection:
    db: Prisma


@pytest.mark.covers("other.database.partitions.lock_wait_outlives_transaction_default", "other.database.partitions.repeat_preserves_rows")
async def test_real_partition_ddl_survives_witnessed_lock_and_is_idempotent() -> None:
    schema: Final = f"integration_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    parsed: Final = urlsplit(url)
    scoped_url: Final = urlunsplit(parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": schema})))
    parent: Final = sql.Identifier(schema, "LiteLLM_SpendLogs")
    with psycopg.connect(url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            setup.execute(sql.SQL('CREATE TABLE {} (request_id text, "startTime" timestamp NOT NULL) PARTITION BY RANGE ("startTime")').format(parent))
            database: Final = Prisma(datasource={"url": scoped_url})
            await database.connect()
            try:
                manager: Final = SpendLogsPartitionManager(interval="day", precreate_ahead=0)
                with psycopg.connect(url) as blocker:
                    blocker.execute(sql.SQL("LOCK TABLE {} IN ACCESS SHARE MODE").format(parent))
                    blocker_pid: Final = blocker.info.backend_pid
                    operation: Final = asyncio.create_task(manager.ensure_partitions(PartitionConnection(database), lambda: 7000))
                    wait_deadline: Final = time.monotonic() + 3
                    try:
                        while True:
                            witnesses: Final = read_rows(
                                "SELECT a.pid, extract(epoch FROM clock_timestamp()-a.query_start)::double precision AS age "
                                "FROM pg_stat_activity a WHERE %s = ANY(pg_blocking_pids(a.pid)) "
                                "AND a.wait_event_type = 'Lock' AND a.query LIKE 'CREATE TABLE IF NOT EXISTS%%'",
                                (blocker_pid,),
                            )
                            if witnesses:
                                break
                            assert time.monotonic() < wait_deadline, "Partition DDL never reached the held lock"
                            await asyncio.sleep(0.02)
                        assert len(witnesses) == 1
                        held_at: Final = time.monotonic()
                        age: Final = float(witnesses[0]["age"])
                        assert age < 1, "DDL lock witness arrived too late for the qualification window"
                        await asyncio.sleep(5.6 - age)
                        held_seconds: Final = age + time.monotonic() - held_at
                        assert 5.5 <= held_seconds < 6.5, f"Lock qualification timing outside window: {held_seconds}"
                        assert not operation.done(), "DDL completed while its required lock was held"
                    except BaseException:
                        operation.cancel()
                        await asyncio.gather(operation, return_exceptions=True)
                        raise
                    finally:
                        blocker.rollback()
                    ensured: Final = await asyncio.wait_for(operation, timeout=5)
                assert len(ensured) == 1, "Partition DDL failed after the permitted lock wait"
                catalog: Final = read_rows(
                    "SELECT child.relname FROM pg_inherits i JOIN pg_class child ON child.oid=i.inhrelid "
                    "JOIN pg_class parent ON parent.oid=i.inhparent JOIN pg_namespace n ON n.oid=parent.relnamespace "
                    "WHERE n.nspname=%s AND parent.relname='LiteLLM_SpendLogs'", (schema,),
                )
                assert catalog == [{"relname": ensured[0]}]
                now: Final = datetime.now(timezone.utc).replace(tzinfo=None)
                setup.execute(sql.SQL('INSERT INTO {} VALUES (%s, %s)').format(parent), ("retained", now))
                assert await manager.ensure_partitions(PartitionConnection(database), lambda: 7000) == ensured
                assert setup.execute(sql.SQL("SELECT request_id FROM {}").format(parent)).fetchall() == [("retained",)]
            finally:
                await database.disconnect()
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
    assert read_rows("SELECT nspname FROM pg_namespace WHERE nspname=%s", (schema,)) == []
