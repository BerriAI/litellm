import asyncio
from datetime import timedelta
from typing import Final, cast

import pytest
from integration._support.actors import ManagementDatabase

from litellm.repositories.analytics_budget import (
    AnalyticsCapacityError,
    AnalyticsDatabase,
    BoundedAnalyticsDatabase,
    analytics_deadline,
)

_SETTINGS: Final = """SELECT current_setting('statement_timeout') AS statement_timeout,
    current_setting('lock_timeout') AS lock_timeout,current_setting('work_mem') AS work_mem,
    current_setting('max_parallel_workers_per_gather') AS parallelism"""


@pytest.mark.asyncio(loop_scope="session")
async def test_analytics_limits_are_transaction_local_and_admission_is_shared(db: ManagementDatabase) -> None:
    database: Final = cast(AnalyticsDatabase, db.prisma.writer_db)
    bounded: Final = BoundedAnalyticsDatabase(database)
    before: Final = await db.prisma.writer_db.query_raw(_SETTINGS)
    limits: Final = await bounded.query_raw(
        """SELECT EXTRACT(epoch FROM current_setting('statement_timeout')::interval) AS seconds,
        EXTRACT(epoch FROM current_setting('lock_timeout')::interval) AS lock_seconds,
        pg_size_bytes(current_setting('work_mem')) AS bytes,
        current_setting('max_parallel_workers_per_gather')::int AS parallelism"""
    )
    assert 0 < float(str(limits[0]["seconds"])) <= 5
    assert 0 < float(str(limits[0]["lock_seconds"])) <= 0.1
    assert int(str(limits[0]["bytes"])) <= 1048576
    assert limits[0]["parallelism"] == 0
    assert await db.prisma.writer_db.query_raw(_SETTINGS) == before
    async with database.tx(max_wait=timedelta(seconds=1), timeout=timedelta(seconds=5)) as transaction:
        await transaction.query_raw("SELECT true AS locked FROM pg_advisory_xact_lock(1876539401,0)")
        await transaction.query_raw("SELECT true AS locked FROM pg_advisory_xact_lock(1876539401,1)")
        with pytest.raises(AnalyticsCapacityError):
            await bounded.query_raw("SELECT 1 AS result")
        assert (await BoundedAnalyticsDatabase(database, background=True).query_raw("SELECT 1 AS result"))[0][
            "result"
        ] == 1
    assert (await bounded.query_raw("SELECT 1 AS result"))[0]["result"] == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_expensive_analytics_are_cancelled_and_release_capacity(db: ManagementDatabase) -> None:
    database: Final = cast(AnalyticsDatabase, db.prisma.writer_db)
    bounded: Final = BoundedAnalyticsDatabase(database)
    before: Final = await db.prisma.writer_db.query_raw(_SETTINGS)
    with pytest.raises(AnalyticsCapacityError):
        await bounded.query_raw("SELECT 1 AS result FROM pg_sleep(30) /* bounded_analytics_timeout */")
    active: Final = await db.prisma.writer_db.query_raw(
        """SELECT count(*)::int AS running FROM pg_stat_activity
        WHERE pid<>pg_backend_pid() AND state='active' AND query LIKE $1""",
        "%bounded_analytics_timeout%",
    )
    assert active[0]["running"] == 0
    assert (await bounded.query_raw("SELECT 1 AS result"))[0]["result"] == 1
    assert await db.prisma.writer_db.query_raw(_SETTINGS) == before


@pytest.mark.asyncio(loop_scope="session")
async def test_work_deadline_cancels_inflight_sql_and_restores_capacity(db: ManagementDatabase) -> None:
    database: Final = cast(AnalyticsDatabase, db.prisma.writer_db)
    bounded: Final = BoundedAnalyticsDatabase(database, background=True)
    loop: Final = asyncio.get_running_loop()
    start: Final = loop.time()
    with analytics_deadline(start + 0.25), analytics_deadline(start + 10):
        with pytest.raises(AnalyticsCapacityError):
            await bounded.query_raw("SELECT 1 FROM pg_sleep(30) /* bounded_analytics_deadline */")
    assert loop.time() - start < 2
    active: Final = await db.prisma.writer_db.query_raw(
        """SELECT count(*)::int AS running FROM pg_stat_activity
        WHERE pid<>pg_backend_pid() AND state='active' AND query LIKE $1""",
        "%bounded_analytics_deadline%",
    )
    assert active[0]["running"] == 0
    with analytics_deadline(loop.time() - 1), pytest.raises(AnalyticsCapacityError):
        await bounded.query_raw("SELECT 1 AS result")
    assert (await bounded.query_raw("SELECT 1 AS result"))[0]["result"] == 1
