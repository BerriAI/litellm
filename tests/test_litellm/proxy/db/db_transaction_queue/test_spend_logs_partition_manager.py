"""
Tests for SpendLogsPartitionManager: partition naming/bounds math, retention
selection, the non-partitioned no-op safety path, and the drop/ensure SQL flow.
"""

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
    assert parse_partition_upper_bound(bound) == datetime(2026, 6, 2, 0, 0, 0, tzinfo=timezone.utc)


def test_parse_partition_upper_bound_default_is_none():
    assert parse_partition_upper_bound("DEFAULT") is None
    assert parse_partition_upper_bound("garbage") is None


def test_select_partitions_to_drop_only_fully_expired():
    cutoff = datetime(2026, 6, 10, 0, 0, 0, tzinfo=timezone.utc)
    partitions = [
        ("p_old", datetime(2026, 6, 9, 0, 0, 0, tzinfo=timezone.utc)),  # upper < cutoff -> drop
        ("p_boundary", datetime(2026, 6, 10, 0, 0, 0, tzinfo=timezone.utc)),  # upper == cutoff -> drop
        ("p_partial", datetime(2026, 6, 11, 0, 0, 0, tzinfo=timezone.utc)),  # straddles cutoff -> keep
        ("p_default", None),  # DEFAULT -> keep
    ]
    assert select_partitions_to_drop(partitions, cutoff) == ["p_old", "p_boundary"]


@pytest.mark.asyncio
async def test_is_partitioned_true_and_false():
    mgr = SpendLogsPartitionManager()

    client_true = MagicMock()
    client_true.db.query_raw = AsyncMock(return_value=[{"partitioned": True}])
    assert await mgr.is_partitioned(client_true) is True

    client_false = MagicMock()
    client_false.db.query_raw = AsyncMock(return_value=[{"partitioned": False}])
    assert await mgr.is_partitioned(client_false) is False


@pytest.mark.asyncio
async def test_catalog_queries_are_scoped_to_current_schema():
    """
    Both catalog lookups must filter by current_schema(); otherwise a same-named
    table in another schema can flip is_partitioned or return foreign partitions.
    """
    mgr = SpendLogsPartitionManager()
    client = MagicMock()
    client.db.query_raw = AsyncMock(return_value=[])

    await mgr.is_partitioned(client)
    is_partitioned_sql = client.db.query_raw.call_args.args[0]
    assert "pg_namespace" in is_partitioned_sql
    assert "current_schema()" in is_partitioned_sql

    await mgr._list_partitions(client)
    list_sql = client.db.query_raw.call_args.args[0]
    assert "pg_namespace" in list_sql
    assert "current_schema()" in list_sql


@pytest.mark.asyncio
async def test_is_partitioned_swallows_errors_and_returns_false():
    """A catalog query failure must not crash cleanup; fall back to non-partitioned."""
    mgr = SpendLogsPartitionManager()
    client = MagicMock()
    client.db.query_raw = AsyncMock(side_effect=Exception("db down"))
    assert await mgr.is_partitioned(client) is False


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

    cutoff = datetime(2026, 6, 5, 0, 0, 0, tzinfo=timezone.utc)
    dropped = await mgr.drop_partitions_older_than(client, cutoff)

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

    created = await mgr.ensure_partitions(client)

    assert len(created) == 3  # current + 2 ahead
    assert client.db.execute_raw.await_count == 3
    first_sql = client.db.execute_raw.call_args_list[0].args[0]
    assert 'PARTITION OF "LiteLLM_SpendLogs"' in first_sql
    assert "CREATE TABLE IF NOT EXISTS" in first_sql


def test_unsupported_interval_raises():
    with pytest.raises(ValueError):
        period_start(date(2026, 6, 1), "year")
    with pytest.raises(ValueError):
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

    created = await mgr.ensure_partitions(client)

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

    created = await mgr.ensure_partitions(client)

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

    cutoff = datetime(2026, 6, 10, 0, 0, 0, tzinfo=timezone.utc)
    dropped = await mgr.drop_partitions_older_than(client, cutoff)

    # both were eligible; the first drop failed so only the second is reported
    assert dropped == ["LiteLLM_SpendLogs_p20260602"]
