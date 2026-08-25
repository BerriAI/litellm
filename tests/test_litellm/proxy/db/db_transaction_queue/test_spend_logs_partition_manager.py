"""
Tests for SpendLogsPartitionManager: partition naming/bounds math, retention
selection, the non-partitioned no-op safety path, and the drop/ensure SQL flow.
"""

from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.db.db_transaction_queue.spend_logs_partition_manager import (
    SpendLogsPartitionManager,
    next_period_start,
    parse_partition_upper_bound,
    partition_name,
    period_start,
    select_partitions_to_drop,
    upcoming_partitions,
)


DDL_TIMEOUT_MS = 30000


def _budget(ms: "int | None" = DDL_TIMEOUT_MS):
    """The injected per-statement bound: a callable re-read before each statement."""
    return lambda: ms


def _wire_tx(db) -> list[str]:
    """
    Model the prisma seam the partition DDL uses.

    Every statement this manager issues, DDL and catalog query alike, runs inside
    db.tx() so it can carry SET LOCAL timeouts. Those SET LOCAL statements are
    collected in the returned list rather than forwarded, so assertions on
    db.execute_raw and db.query_raw still see only the real statements.
    """
    session_settings: list[str] = []

    @asynccontextmanager
    async def _tx():
        tx = MagicMock()

        async def _execute_raw(sql, *args):
            if sql.lstrip().upper().startswith("SET LOCAL"):
                session_settings.append(sql.strip())
                return 0
            return await db.execute_raw(sql, *args)

        async def _query_raw(sql, *args):
            return await db.query_raw(sql, *args)

        tx.execute_raw = _execute_raw
        tx.query_raw = _query_raw
        yield tx

    db.tx = _tx
    return session_settings


def test_period_start_per_interval():
    d = date(2026, 6, 3)  # a Wednesday
    assert period_start(d, "day") == date(2026, 6, 3)
    assert period_start(d, "week") == date(2026, 6, 1)  # Monday
    assert period_start(d, "month") == date(2026, 6, 1)


def test_next_period_start_crosses_year_and_month_boundaries():
    assert next_period_start(date(2026, 6, 3), "day") == date(2026, 6, 4)
    assert next_period_start(date(2026, 6, 1), "week") == date(2026, 6, 8)
    assert next_period_start(date(2026, 12, 1), "month") == date(2027, 1, 1)


def test_partition_name_uses_period_start_date():
    assert partition_name(date(2026, 6, 1)) == "LiteLLM_SpendLogs_p20260601"


def test_upcoming_partitions_count_and_contiguous_ranges():
    specs = upcoming_partitions(date(2026, 6, 1), "day", ahead=3)
    assert len(specs) == 4  # current + 3 ahead
    names = [s[0] for s in specs]
    assert names == [
        "LiteLLM_SpendLogs_p20260601",
        "LiteLLM_SpendLogs_p20260602",
        "LiteLLM_SpendLogs_p20260603",
        "LiteLLM_SpendLogs_p20260604",
    ]
    # ranges must be contiguous and half-open: each upper is the next lower
    for (_, _, upper), (_, next_lower, _) in zip(specs, specs[1:]):
        assert upper == next_lower


def test_parse_partition_upper_bound_extracts_to_value():
    bound = "FOR VALUES FROM ('2026-06-01 00:00:00') TO ('2026-06-02 00:00:00')"
    assert parse_partition_upper_bound(bound) == datetime(2026, 6, 2, 0, 0, 0)


def test_parse_partition_upper_bound_default_is_none():
    assert parse_partition_upper_bound("DEFAULT") is None
    assert parse_partition_upper_bound("garbage") is None


def test_select_partitions_to_drop_only_fully_expired():
    cutoff = datetime(2026, 6, 10, 0, 0, 0)
    partitions = [
        ("p_old", datetime(2026, 6, 9, 0, 0, 0)),  # upper < cutoff -> drop
        ("p_boundary", datetime(2026, 6, 10, 0, 0, 0)),  # upper == cutoff -> drop
        ("p_partial", datetime(2026, 6, 11, 0, 0, 0)),  # straddles cutoff -> keep
        ("p_default", None),  # DEFAULT -> keep
    ]
    assert select_partitions_to_drop(partitions, cutoff) == ["p_old", "p_boundary"]


@pytest.mark.asyncio
async def test_is_partitioned_true_and_false():
    mgr = SpendLogsPartitionManager()

    client_true = MagicMock()
    client_true.db.query_raw = AsyncMock(return_value=[{"partitioned": True}])
    _wire_tx(client_true.db)
    assert await mgr.is_partitioned(client_true, _budget()) is True

    client_false = MagicMock()
    client_false.db.query_raw = AsyncMock(return_value=[{"partitioned": False}])
    _wire_tx(client_false.db)
    assert await mgr.is_partitioned(client_false, _budget()) is False


@pytest.mark.asyncio
async def test_catalog_queries_are_scoped_to_current_schema():
    """
    Both catalog lookups must filter by current_schema(); otherwise a same-named
    table in another schema can flip is_partitioned or return foreign partitions.
    """
    mgr = SpendLogsPartitionManager()
    client = MagicMock()
    client.db.query_raw = AsyncMock(return_value=[])
    _wire_tx(client.db)

    await mgr.is_partitioned(client, _budget())
    is_partitioned_sql = client.db.query_raw.call_args.args[0]
    assert "pg_namespace" in is_partitioned_sql
    assert "current_schema()" in is_partitioned_sql

    await mgr._list_partitions(client, DDL_TIMEOUT_MS)
    list_sql = client.db.query_raw.call_args.args[0]
    assert "pg_namespace" in list_sql
    assert "current_schema()" in list_sql


@pytest.mark.asyncio
async def test_is_partitioned_swallows_errors_and_returns_false():
    """A catalog query failure must not crash cleanup; fall back to non-partitioned."""
    mgr = SpendLogsPartitionManager()
    client = MagicMock()
    client.db.query_raw = AsyncMock(side_effect=Exception("db down"))
    # Wire the real seam: without it the async with itself raises, and the test
    # would pass on the wrong exception.
    _wire_tx(client.db)
    assert await mgr.is_partitioned(client, _budget()) is False


@pytest.mark.asyncio
async def test_drop_partitions_older_than_drops_expired_only():
    mgr = SpendLogsPartitionManager()
    client = MagicMock()
    client.db.query_raw = AsyncMock(
        return_value=[
            {
                "name": "LiteLLM_SpendLogs_p20260601",
                "bound": "FOR VALUES FROM ('2026-06-01 00:00:00') TO ('2026-06-02 00:00:00')",
            },
            {
                "name": "LiteLLM_SpendLogs_p20260609",
                "bound": "FOR VALUES FROM ('2026-06-09 00:00:00') TO ('2026-06-10 00:00:00')",
            },
            {"name": "LiteLLM_SpendLogs_pdefault", "bound": "DEFAULT"},
        ]
    )
    client.db.execute_raw = AsyncMock(return_value=0)
    _wire_tx(client.db)

    cutoff = datetime(2026, 6, 5, 0, 0, 0, tzinfo=timezone.utc)
    dropped = await mgr.drop_partitions_older_than(client, cutoff, _budget())

    assert dropped == ["LiteLLM_SpendLogs_p20260601"]
    executed = " ".join(call.args[0] for call in client.db.execute_raw.call_args_list)
    assert 'DROP TABLE IF EXISTS "LiteLLM_SpendLogs_p20260601"' in executed
    assert "p20260609" not in executed
    assert "pdefault" not in executed


@pytest.mark.asyncio
async def test_ensure_partitions_issues_create_for_each_period():
    mgr = SpendLogsPartitionManager(interval="day", precreate_ahead=2)
    client = MagicMock()
    client.db.execute_raw = AsyncMock(return_value=0)
    _wire_tx(client.db)

    created = await mgr.ensure_partitions(client, _budget())

    assert len(created) == 3  # current + 2 ahead
    assert client.db.execute_raw.await_count == 3
    first_sql = client.db.execute_raw.call_args_list[0].args[0]
    assert 'PARTITION OF "LiteLLM_SpendLogs"' in first_sql
    assert "CREATE TABLE IF NOT EXISTS" in first_sql


@pytest.mark.asyncio
async def test_partition_ddl_carries_a_statement_and_lock_timeout():
    """
    Partition DDL takes an ACCESS EXCLUSIVE lock, so an unbounded DROP queues
    behind any long-running reader for as long as that reader lives. That is the
    one path by which cleanup could outlast its run budget without bound, and
    lock_timeout is what bounds the wait rather than only the work.
    """
    mgr = SpendLogsPartitionManager(interval="day", precreate_ahead=0)
    client = MagicMock()
    client.db.execute_raw = AsyncMock(return_value=0)
    client.db.query_raw = AsyncMock(
        return_value=[
            {
                "name": "LiteLLM_SpendLogs_p20260601",
                "bound": "FOR VALUES FROM ('2026-06-01 00:00:00') TO ('2026-06-02 00:00:00')",
            }
        ]
    )
    session_settings = _wire_tx(client.db)

    await mgr.ensure_partitions(client, _budget(7000))
    await mgr.drop_partitions_older_than(client, datetime(2026, 6, 5, tzinfo=timezone.utc), _budget(7000))

    # Three statements were issued: the CREATE, the catalog list the drop needs,
    # and the DROP. All three carry a statement timeout; only the two that take
    # a lock also carry a lock timeout, since the catalog read takes none.
    assert session_settings.count("SET LOCAL statement_timeout = 7000") == 3
    assert session_settings.count("SET LOCAL lock_timeout = 7000") == 2


@pytest.mark.asyncio
async def test_catalog_queries_carry_a_statement_timeout():
    """
    Bounding only the DDL leaves the two catalog lookups as statements this job
    issues with no bound at all, so a run could still outlast its budget waiting
    on one. Every statement the manager issues carries the caller's timeout.
    """
    mgr = SpendLogsPartitionManager()
    client = MagicMock()
    client.db.query_raw = AsyncMock(return_value=[])
    session_settings = _wire_tx(client.db)

    await mgr.is_partitioned(client, _budget(4000))
    assert session_settings == ["SET LOCAL statement_timeout = 4000"], (
        f"is_partitioned issued no statement timeout: {session_settings}"
    )

    session_settings.clear()
    await mgr._list_partitions(client, 4000)
    assert session_settings == ["SET LOCAL statement_timeout = 4000"], (
        f"_list_partitions issued no statement timeout: {session_settings}"
    )


@pytest.mark.asyncio
async def test_partition_loops_stop_when_the_budget_runs_out_mid_way():
    """
    Each loop issues one statement per partition, so a bound read once at entry
    would let N statements each run for the budget that was left before the
    first of them. The bound is re-read per statement and the loop stops.
    """
    mgr = SpendLogsPartitionManager(interval="day", precreate_ahead=4)
    client = MagicMock()
    client.db.execute_raw = AsyncMock(return_value=0)
    _wire_tx(client.db)

    # Budget for two statements, then spent.
    calls = {"n": 0}

    def budget() -> "int | None":
        calls["n"] += 1
        return 5000 if calls["n"] <= 2 else None

    created = await mgr.ensure_partitions(client, budget)

    assert len(created) == 2, f"the loop ran past its budget and created {len(created)}"
    assert client.db.execute_raw.await_count == 2


@pytest.mark.asyncio
async def test_partition_maintenance_issues_nothing_when_the_budget_is_already_spent():
    """A run with no budget left must not issue even the catalog lookups."""
    mgr = SpendLogsPartitionManager(interval="day", precreate_ahead=2)
    client = MagicMock()
    client.db.execute_raw = AsyncMock(return_value=0)
    client.db.query_raw = AsyncMock(return_value=[])
    _wire_tx(client.db)

    spent = _budget(None)

    assert await mgr.is_partitioned(client, spent) is False
    assert await mgr.ensure_partitions(client, spent) == []
    assert await mgr.drop_partitions_older_than(client, datetime(2026, 6, 5, tzinfo=timezone.utc), spent) == []

    client.db.execute_raw.assert_not_awaited()
    client.db.query_raw.assert_not_awaited()


def test_unsupported_interval_raises():
    with pytest.raises(ValueError, match='Unsupported partition interval: year'):
        period_start(date(2026, 6, 1), "year")
    with pytest.raises(ValueError, match='Unsupported partition interval: year'):
        next_period_start(date(2026, 6, 1), "year")


def test_parse_partition_upper_bound_unparseable_to_value_is_none():
    """A TO(...) value that is not a valid timestamp must not raise; return None."""
    assert (
        parse_partition_upper_bound("FOR VALUES FROM ('x') TO ('not-a-date')") is None
    )


@pytest.mark.asyncio
async def test_ensure_partitions_continues_when_one_create_fails():
    mgr = SpendLogsPartitionManager(interval="day", precreate_ahead=2)
    client = MagicMock()
    client.db.execute_raw = AsyncMock(side_effect=[0, Exception("overlap"), 0])
    _wire_tx(client.db)

    created = await mgr.ensure_partitions(client, _budget())

    # the failed partition is skipped, the others still created
    assert len(created) == 2
    assert client.db.execute_raw.await_count == 3


def test_invalid_interval_falls_back_to_day():
    """
    An invalid interval must not be stored as-is. Otherwise ensure_partitions
    raises (via period_start) and aborts the cleanup run before retention drops
    old partitions, silently skipping retention.
    """
    mgr = SpendLogsPartitionManager(interval="year")
    assert mgr.interval == "day"


@pytest.mark.asyncio
async def test_invalid_interval_does_not_abort_ensure_partitions():
    """With the fallback, ensure_partitions completes instead of raising ValueError."""
    mgr = SpendLogsPartitionManager(interval="fortnight", precreate_ahead=1)
    client = MagicMock()
    client.db.execute_raw = AsyncMock(return_value=0)
    _wire_tx(client.db)

    created = await mgr.ensure_partitions(client, _budget())

    assert len(created) == 2  # current + 1 ahead, day-based fallback


@pytest.mark.asyncio
async def test_drop_partitions_continues_when_one_drop_fails():
    mgr = SpendLogsPartitionManager()
    client = MagicMock()
    client.db.query_raw = AsyncMock(
        return_value=[
            {
                "name": "LiteLLM_SpendLogs_p20260601",
                "bound": "FOR VALUES FROM ('2026-06-01 00:00:00') TO ('2026-06-02 00:00:00')",
            },
            {
                "name": "LiteLLM_SpendLogs_p20260602",
                "bound": "FOR VALUES FROM ('2026-06-02 00:00:00') TO ('2026-06-03 00:00:00')",
            },
        ]
    )
    client.db.execute_raw = AsyncMock(side_effect=[Exception("locked"), 0])
    _wire_tx(client.db)

    cutoff = datetime(2026, 6, 10, 0, 0, 0, tzinfo=timezone.utc)
    dropped = await mgr.drop_partitions_older_than(client, cutoff, _budget())

    # both were eligible; the first drop failed so only the second is reported
    assert dropped == ["LiteLLM_SpendLogs_p20260602"]
