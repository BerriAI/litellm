import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
from psycopg import sql
from prisma import Prisma

from integration._support.database import read_rows
from litellm.proxy.db.db_transaction_queue.spend_logs_partition_manager import SpendLogsPartitionManager, partition_name


@dataclass(frozen=True)
class PartitionConnection:
    db: Prisma


async def _wait_for_partition_ddl_lock(blocker_pid: int, relation: sql.Identifier | None = None) -> float:
    deadline: Final = time.monotonic() + 10
    relation_name: Final = relation.as_string() if relation is not None else ""
    while time.monotonic() < deadline:
        witnesses: Final = await asyncio.to_thread(
            read_rows,
            "SELECT extract(epoch FROM clock_timestamp()-a.query_start)::double precision AS age "
            "FROM pg_stat_activity a "
            "WHERE %s::int = ANY(pg_blocking_pids(a.pid)) AND a.wait_event_type='Lock' "
            "AND (%s = '' OR EXISTS (SELECT 1 FROM pg_locks l WHERE l.pid=a.pid "
            "AND NOT l.granted AND l.relation=NULLIF(%s, '')::regclass))",
            (str(blocker_pid), relation_name, relation_name),
        )
        if witnesses:
            assert len(witnesses) == 1
            age: Final = witnesses[0]["age"]
            assert isinstance(age, (float, int))
            return float(age)
        await asyncio.sleep(0.02)
    pytest.fail("Partition DDL never reached the held lock")


@pytest.mark.covers(
    "other.database.partitions.lock_wait_outlives_transaction_default",
    "other.database.partitions.repeat_preserves_rows",
)
async def test_real_partition_ddl_survives_witnessed_lock_and_is_idempotent() -> None:
    schema: Final = f"integration_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    parsed: Final = urlsplit(url)
    scoped_url: Final = urlunsplit(
        parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": schema}))
    )
    parent: Final = sql.Identifier(schema, "LiteLLM_SpendLogs")
    with psycopg.connect(url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            setup.execute(
                sql.SQL(
                    'CREATE TABLE {} (request_id text, "startTime" timestamp NOT NULL) PARTITION BY RANGE ("startTime")'
                ).format(parent)
            )
            database: Final = Prisma(datasource={"url": scoped_url})
            await database.connect()
            try:
                manager: Final = SpendLogsPartitionManager(interval="day", precreate_ahead=0)
                client: Final = PartitionConnection(database)
                assert await manager.is_partitioned(client, lambda: 7000)
                with psycopg.connect(url) as blocker:
                    blocker.execute(sql.SQL("LOCK TABLE ONLY {} IN SHARE UPDATE EXCLUSIVE MODE").format(parent))
                    blocker_pid: Final = blocker.info.backend_pid
                    operation: Final = asyncio.create_task(manager.ensure_partitions(client, lambda: 7000))
                    try:
                        age: Final = await _wait_for_partition_ddl_lock(blocker_pid, parent)
                        held_at: Final = time.monotonic()
                        await asyncio.sleep(max(0, 5.6 - age))
                        held_seconds: Final = age + time.monotonic() - held_at
                        assert held_seconds >= 5.5, f"Lock released before the transaction boundary: {held_seconds}"
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
                    "WHERE n.nspname=%s AND parent.relname='LiteLLM_SpendLogs'",
                    (schema,),
                )
                assert catalog == [{"relname": ensured[0]}]
                now: Final = datetime.now(timezone.utc).replace(tzinfo=None)
                setup.execute(sql.SQL("INSERT INTO {} VALUES (%s, %s)").format(parent), ("retained", now))
                assert await manager.ensure_partitions(client, lambda: 7000) == ensured
                assert setup.execute(sql.SQL("SELECT request_id FROM {}").format(parent)).fetchall() == [("retained",)]
            finally:
                await database.disconnect()
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
    assert read_rows("SELECT nspname FROM pg_namespace WHERE nspname=%s", (schema,)) == []


async def test_real_partition_attachment_allows_writes_to_existing_partition() -> None:
    schema: Final = f"integration_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    parsed: Final = urlsplit(url)
    scoped_url: Final = urlunsplit(
        parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": schema}))
    )
    parent: Final = sql.Identifier(schema, "LiteLLM_SpendLogs")
    previous: Final = sql.Identifier(schema, "spend_previous")
    default: Final = sql.Identifier(schema, "spend_default")
    today: Final = datetime.now(timezone.utc).date()
    with psycopg.connect(url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            setup.execute(
                sql.SQL(
                    'CREATE TABLE {} (request_id text, "startTime" timestamp NOT NULL) PARTITION BY RANGE ("startTime")'
                ).format(parent)
            )
            setup.execute(
                sql.SQL("CREATE TABLE {} PARTITION OF {} FOR VALUES FROM (MINVALUE) TO ({})").format(
                    previous, parent, sql.Literal(today)
                )
            )
            setup.execute(sql.SQL("CREATE TABLE {} PARTITION OF {} DEFAULT").format(default, parent))
            database: Final = Prisma(datasource={"url": scoped_url})
            await database.connect()
            try:
                manager: Final = SpendLogsPartitionManager(interval="day", precreate_ahead=0)
                with psycopg.connect(url) as blocker:
                    blocker.execute(sql.SQL("LOCK TABLE ONLY {} IN ACCESS SHARE MODE").format(default))
                    operation: Final = asyncio.create_task(
                        manager.ensure_partitions(PartitionConnection(database), lambda: 15000)
                    )
                    try:
                        await _wait_for_partition_ddl_lock(blocker.info.backend_pid, default)
                        async with await psycopg.AsyncConnection.connect(url, autocommit=True) as writer:
                            await writer.execute("SET statement_timeout = '2s'")
                            inserted: Final = await writer.execute(
                                sql.SQL("INSERT INTO {} VALUES (%s, %s) RETURNING tableoid = %s::regclass").format(
                                    parent
                                ),
                                ("during_attachment", today - timedelta(days=1), previous.as_string()),
                            )
                            assert await inserted.fetchone() == (True,)
                        assert not operation.done(), "Partition DDL finished before DEFAULT was unlocked"
                    finally:
                        blocker.rollback()
                        ensured: Final = await asyncio.wait_for(operation, timeout=5)
                assert len(ensured) == 1, "Partition DDL failed after DEFAULT was unlocked"
                catalog: Final = read_rows(
                    "SELECT inhrelid::regclass::text AS partition FROM pg_inherits "
                    "WHERE inhparent=%s::regclass AND inhrelid=%s::regclass",
                    (parent.as_string(), sql.Identifier(schema, ensured[0]).as_string()),
                )
                assert len(catalog) == 1, "New partition was not attached to SpendLogs"
            finally:
                await database.disconnect()
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


async def test_real_partition_creation_blocks_overlapping_writes_to_default() -> None:
    schema: Final = f"integration_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    parsed: Final = urlsplit(url)
    scoped_url: Final = urlunsplit(
        parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": schema}))
    )
    parent: Final = sql.Identifier(schema, "LiteLLM_SpendLogs")
    default: Final = sql.Identifier(schema, "spend_default")
    today: Final = datetime.now(timezone.utc).date()
    child_name: Final = partition_name(today)
    child: Final = sql.Identifier(schema, child_name)
    with psycopg.connect(url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            setup.execute(
                sql.SQL(
                    'CREATE TABLE {} (request_id text, "startTime" timestamp NOT NULL) PARTITION BY RANGE ("startTime")'
                ).format(parent)
            )
            setup.execute(sql.SQL("CREATE TABLE {} PARTITION OF {} DEFAULT").format(default, parent))
            database: Final = Prisma(datasource={"url": scoped_url})
            await database.connect()
            try:
                manager: Final = SpendLogsPartitionManager(interval="day", precreate_ahead=0)
                with psycopg.connect(url) as blocker:
                    blocker.execute(sql.SQL("CREATE TABLE {} (placeholder int)").format(child))
                    operation: Final = asyncio.create_task(
                        manager.ensure_partitions(PartitionConnection(database), lambda: 15000)
                    )
                    try:
                        await _wait_for_partition_ddl_lock(blocker.info.backend_pid)
                        async with await psycopg.AsyncConnection.connect(url, autocommit=True) as writer:
                            await writer.execute("SET statement_timeout = '2s'")
                            with pytest.raises(psycopg.errors.QueryCanceled, match="statement timeout"):
                                await writer.execute(
                                    sql.SQL("INSERT INTO {} VALUES (%s, %s)").format(parent), ("overlapping", today)
                                )
                    finally:
                        blocker.rollback()
                        ensured: Final = await asyncio.wait_for(operation, timeout=5)
                assert ensured == [child_name], "Partition creation failed after the conflicting CREATE rolled back"
            finally:
                await database.disconnect()
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
