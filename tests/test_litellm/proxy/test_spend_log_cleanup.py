"""
Test cases for spend log cleanup functionality
"""

import asyncio
import math
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.constants import (
    SPEND_LOG_CLEANUP_BATCH_SIZE,
    SPEND_LOG_CLEANUP_REMAINING_COUNT_CAP,
    SPEND_LOG_CLEANUP_RUN_BUDGET_SECONDS,
)
from litellm.proxy.db.db_transaction_queue.spend_log_cleanup import (
    SPEND_LOG_CLEANUP_BOUND_SETTINGS,
    SpendLogCleanup,
    TableCleanupResult,
)
from litellm.proxy.db.db_transaction_queue.spend_log_cleanup_metrics import (
    SpendLogCleanupMetrics,
)


def _far_deadline() -> float:
    """A run deadline far enough out that only the other bounds can stop a batch loop."""
    return time.monotonic() + 3600


def _wire_tx(db):
    """
    Model the prisma seam the cleanup job actually uses.

    Every statement the job issues runs inside db.tx() so it can carry a SET
    LOCAL statement_timeout. Batch and probe statements are forwarded to
    db.execute_raw and db.query_raw, which is what tests configure and assert
    on, while the SET LOCAL statements are answered here so they neither consume
    a side_effect entry nor show up in the recorded call list. Lookup is
    deferred to call time so this can be wired before a test assigns its own
    execute_raw.
    """

    @asynccontextmanager
    async def _tx():
        tx = MagicMock()

        async def _execute_raw(sql, *args):
            if sql.lstrip().upper().startswith("SET LOCAL"):
                return 0
            return await db.execute_raw(sql, *args)

        async def _query_raw(sql, *args):
            return await db.query_raw(sql, *args)

        tx.execute_raw = _execute_raw
        tx.query_raw = _query_raw
        yield tx

    db.tx = _tx
    db.query_raw = AsyncMock(return_value=[{"remaining": 0}])


def test_spend_log_cleanup_cron_scheduling():
    """Test that cron expressions are correctly parsed for spend log cleanup scheduling"""
    from apscheduler.triggers.cron import CronTrigger

    # Valid cron expressions
    cron_expr = "0 4 * * *"  # 4:00 AM daily
    trigger = CronTrigger.from_crontab(cron_expr)
    assert trigger is not None

    # Every minute (useful for testing)
    trigger_minute = CronTrigger.from_crontab("*/1 * * * *")
    assert trigger_minute is not None

    # Specific day and hour
    trigger_weekly = CronTrigger.from_crontab("0 3 * * 0")  # 3 AM every Sunday
    assert trigger_weekly is not None

    # Invalid cron expression should raise ValueError
    with pytest.raises(ValueError, match='Wrong number of fields; got'):
        CronTrigger.from_crontab("invalid cron")

    with pytest.raises(ValueError, match='is higher than the maximum value'):
        CronTrigger.from_crontab("60 25 * * *")  # Invalid minute and hour


def test_spend_log_cleanup_cron_scheduler_integration():
    """
    Integration test: Verify the proxy_server scheduler logic correctly adds
    cron-based cleanup job when maximum_spend_logs_cleanup_cron is configured.

    This tests the logic in proxy_server.py lines 4671-4717 without requiring
    a real database connection.
    """
    from unittest.mock import MagicMock
    from apscheduler.triggers.cron import CronTrigger

    # Mock scheduler
    mock_scheduler = MagicMock()
    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_cleanup_instance = MagicMock()

    # Test Case 1: Cron-based scheduling
    general_settings_cron = {
        "maximum_spend_logs_retention_period": "7d",
        "maximum_spend_logs_cleanup_cron": "0 4 * * *",  # 4 AM daily
    }

    cleanup_cron = general_settings_cron.get("maximum_spend_logs_cleanup_cron")
    assert cleanup_cron is not None

    # Simulate the scheduler logic from proxy_server.py
    cron_trigger = CronTrigger.from_crontab(cleanup_cron)
    mock_scheduler.add_job(
        mock_cleanup_instance.cleanup_old_spend_logs,
        cron_trigger,
        args=[mock_prisma_client],
        id="spend_log_cleanup_job",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    # Verify scheduler was called correctly
    mock_scheduler.add_job.assert_called_once()
    call_args = mock_scheduler.add_job.call_args

    # Verify the trigger is a CronTrigger
    assert isinstance(call_args[0][1], CronTrigger)

    # Verify job ID
    assert call_args[1]["id"] == "spend_log_cleanup_job"
    assert call_args[1]["replace_existing"] is True

    # Test Case 2: Interval-based scheduling (fallback)
    mock_scheduler.reset_mock()
    general_settings_interval = {
        "maximum_spend_logs_retention_period": "7d",
        # No cron, so it should fall back to interval
    }

    cleanup_cron_fallback = general_settings_interval.get(
        "maximum_spend_logs_cleanup_cron"
    )
    assert cleanup_cron_fallback is None  # No cron configured

    # Simulate interval-based scheduling fallback
    retention_interval = general_settings_interval.get(
        "maximum_spend_logs_retention_interval", "1d"
    )
    from litellm.litellm_core_utils.duration_parser import duration_in_seconds

    interval_seconds = duration_in_seconds(retention_interval)

    mock_scheduler.add_job(
        mock_cleanup_instance.cleanup_old_spend_logs,
        "interval",
        seconds=interval_seconds,
        args=[mock_prisma_client],
        id="spend_log_cleanup_job",
        replace_existing=True,
    )

    # Verify interval scheduling was called
    mock_scheduler.add_job.assert_called_once()
    interval_call_args = mock_scheduler.add_job.call_args
    assert interval_call_args[0][1] == "interval"
    assert interval_call_args[1]["seconds"] == 86400  # 1 day in seconds


@pytest.mark.asyncio
async def test_should_delete_spend_logs():
    # Test case 1: No retention set
    cleaner = SpendLogCleanup(general_settings={})
    assert cleaner._should_delete_spend_logs() is False

    # Test case 2: Valid seconds string
    cleaner = SpendLogCleanup(
        general_settings={"maximum_spend_logs_retention_period": "3600s"}
    )
    assert cleaner._should_delete_spend_logs() is True

    # Test case 3: Valid days string
    cleaner = SpendLogCleanup(
        general_settings={"maximum_spend_logs_retention_period": "30d"}
    )
    assert cleaner._should_delete_spend_logs() is True

    # Test case 4: Valid hours string
    cleaner = SpendLogCleanup(
        general_settings={"maximum_spend_logs_retention_period": "24h"}
    )
    assert cleaner._should_delete_spend_logs() is True

    # Test case 5: Invalid format
    cleaner = SpendLogCleanup(
        general_settings={"maximum_spend_logs_retention_period": "invalid"}
    )
    assert cleaner._should_delete_spend_logs() is False


@pytest.mark.asyncio
async def test_cleanup_old_spend_logs_batch_deletion():
    from unittest.mock import AsyncMock, MagicMock

    # Setup Prisma client
    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_db = MagicMock()
    _wire_tx(mock_db)

    # Mock execute_raw to return deleted counts (3 spend-log batches, then the
    # tool-index cleanup's first batch returning 0)
    mock_db.execute_raw = AsyncMock(side_effect=[1000, 500, 0, 0])

    # Wire up mocks
    mock_prisma_client.db = mock_db

    # Mock Redis cache and pod_lock_manager
    mock_redis_cache = MagicMock()
    mock_pod_lock_manager = MagicMock()
    mock_pod_lock_manager.redis_cache = mock_redis_cache
    mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
    mock_pod_lock_manager.release_lock = AsyncMock()

    # Run cleanup with mocked pod_lock_manager
    test_settings = {"maximum_spend_logs_retention_period": "7d"}
    cleaner = SpendLogCleanup(general_settings=test_settings)
    cleaner.pod_lock_manager = mock_pod_lock_manager
    assert cleaner._should_delete_spend_logs() is True
    await cleaner.cleanup_old_spend_logs(mock_prisma_client)

    # Validate batching and deletion via raw SQL
    assert mock_db.execute_raw.call_count == 4

    # Check the first call argument
    call_args_sql = mock_db.execute_raw.call_args_list[0][0][0]
    assert 'DELETE FROM "LiteLLM_SpendLogs"' in call_args_sql
    # must match on the full composite identity: on a partitioned table
    # request_id alone is not unique, and deleting by it would let a client
    # reusing x-litellm-call-id take out a fresh row alongside the expired one
    assert 'WHERE ("request_id", "startTime") IN' in call_args_sql

    # After spend logs, the derived tool index rows expire on the same cutoff
    tool_index_sql = mock_db.execute_raw.call_args_list[3][0][0]
    assert 'DELETE FROM "LiteLLM_SpendLogToolIndex"' in tool_index_sql

    # The LiteLLM_DailyToolSpend rollup must outlive spend-log retention: it is
    # the only copy of tool spend history once its per-request sources expire,
    # so spend-log cleanup must never touch it.
    for call in mock_db.execute_raw.call_args_list:
        assert "LiteLLM_DailyToolSpend" not in call[0][0]


@pytest.mark.asyncio
async def test_cleanup_old_spend_logs_retention_period_cutoff():
    """
    Test that logs are filtered using correct cutoff based on retention
    """
    # Setup Prisma client
    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_db = MagicMock()
    _wire_tx(mock_db)
    mock_db.execute_raw = AsyncMock(return_value=0)
    mock_prisma_client.db = mock_db

    # Mock Redis cache and pod_lock_manager
    mock_redis_cache = MagicMock()
    mock_pod_lock_manager = MagicMock()
    mock_pod_lock_manager.redis_cache = mock_redis_cache
    mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
    mock_pod_lock_manager.release_lock = AsyncMock()

    # Run cleanup with mocked pod_lock_manager
    test_settings = {"maximum_spend_logs_retention_period": "24h"}
    cleaner = SpendLogCleanup(general_settings=test_settings)
    cleaner.pod_lock_manager = mock_pod_lock_manager
    assert cleaner._should_delete_spend_logs() is True
    await cleaner.cleanup_old_spend_logs(mock_prisma_client)

    # Verify the cutoff date is correct
    cutoff_date = mock_db.execute_raw.call_args[0][1]
    expected_cutoff = datetime.now(timezone.utc) - timedelta(seconds=86400)
    assert (
        abs((cutoff_date - expected_cutoff).total_seconds()) < 1
    )  # Allow 1 second difference for test execution time


@pytest.mark.asyncio
async def test_cleanup_drops_partitions_when_enabled_and_partitioned():
    """
    With use_spend_logs_partitioning enabled and a partitioned table, cleanup
    must reclaim disk by dropping partitions AND still delete expired rows the
    drops cannot reach (DEFAULT partition, cutoff-spanning partitions), so
    retention is never bypassed.
    """
    from unittest.mock import AsyncMock, MagicMock

    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_prisma_client.db.execute_raw = AsyncMock(return_value=0)

    partition_manager = MagicMock()
    partition_manager.is_partitioned = AsyncMock(return_value=True)
    partition_manager.ensure_partitions = AsyncMock(return_value=["p1"])
    partition_manager.drop_partitions_older_than = AsyncMock(
        return_value=["LiteLLM_SpendLogs_p20260601"]
    )

    cleaner = SpendLogCleanup(
        general_settings={
            "maximum_spend_logs_retention_period": "7d",
            "use_spend_logs_partitioning": True,
        },
        partition_manager=partition_manager,
    )
    cleaner.pod_lock_manager = MagicMock()
    cleaner.pod_lock_manager.redis_cache = None

    await cleaner.cleanup_old_spend_logs(mock_prisma_client)

    partition_manager.ensure_partitions.assert_awaited_once()
    partition_manager.drop_partitions_older_than.assert_awaited_once()
    delete_sql = mock_prisma_client.db.execute_raw.call_args_list[0][0][0]
    assert 'DELETE FROM "LiteLLM_SpendLogs"' in delete_sql
    # Partition drops only reclaim spend logs; the tool index must still be
    # cleaned row-wise on the same run
    all_sql = [c[0][0] for c in mock_prisma_client.db.execute_raw.call_args_list]
    assert any('DELETE FROM "LiteLLM_SpendLogToolIndex"' in s for s in all_sql)


@pytest.mark.asyncio
async def test_cleanup_uses_delete_when_partitioning_not_enabled():
    """
    Even against a partitioned table, the partition path must stay off until
    use_spend_logs_partitioning is explicitly enabled, so existing deployments
    see zero behavior change. The catalog must not even be queried.
    """
    from unittest.mock import AsyncMock, MagicMock

    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_prisma_client.db.execute_raw = AsyncMock(side_effect=[10, 0, 0])

    partition_manager = MagicMock()
    partition_manager.is_partitioned = AsyncMock(return_value=True)
    partition_manager.ensure_partitions = AsyncMock()
    partition_manager.drop_partitions_older_than = AsyncMock()

    cleaner = SpendLogCleanup(
        general_settings={"maximum_spend_logs_retention_period": "7d"},
        partition_manager=partition_manager,
    )
    cleaner.pod_lock_manager = MagicMock()
    cleaner.pod_lock_manager.redis_cache = None

    await cleaner.cleanup_old_spend_logs(mock_prisma_client)

    partition_manager.is_partitioned.assert_not_awaited()
    partition_manager.drop_partitions_older_than.assert_not_awaited()
    delete_sql = mock_prisma_client.db.execute_raw.call_args_list[0][0][0]
    assert 'DELETE FROM "LiteLLM_SpendLogs"' in delete_sql


@pytest.mark.asyncio
async def test_cleanup_uses_delete_when_not_partitioned():
    """
    With the feature enabled but the table not actually partitioned (script not
    run yet), cleanup must keep using the batched DELETE path.
    """
    from unittest.mock import AsyncMock, MagicMock

    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_prisma_client.db.execute_raw = AsyncMock(side_effect=[10, 0, 0])

    partition_manager = MagicMock()
    partition_manager.is_partitioned = AsyncMock(return_value=False)
    partition_manager.drop_partitions_older_than = AsyncMock()

    cleaner = SpendLogCleanup(
        general_settings={
            "maximum_spend_logs_retention_period": "7d",
            "use_spend_logs_partitioning": True,
        },
        partition_manager=partition_manager,
    )
    cleaner.pod_lock_manager = MagicMock()
    cleaner.pod_lock_manager.redis_cache = None

    await cleaner.cleanup_old_spend_logs(mock_prisma_client)

    partition_manager.drop_partitions_older_than.assert_not_awaited()
    assert mock_prisma_client.db.execute_raw.await_count == 3
    delete_sql = mock_prisma_client.db.execute_raw.call_args_list[0][0][0]
    assert 'DELETE FROM "LiteLLM_SpendLogs"' in delete_sql


@pytest.mark.asyncio
async def test_cleanup_old_spend_logs_no_retention_period():
    """
    Test that no logs are deleted when no retention period is set
    """
    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_prisma_client.db.execute_raw = AsyncMock()

    cleaner = SpendLogCleanup(general_settings={})  # no retention
    await cleaner.cleanup_old_spend_logs(mock_prisma_client)

    mock_prisma_client.db.execute_raw.assert_not_called()


@pytest.mark.asyncio
async def test_lock_not_released_when_not_acquired():
    """
    Lock release should be skipped when _should_delete_spend_logs returns False
    before the lock is ever acquired.
    """
    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_prisma_client.db.execute_raw = AsyncMock()

    mock_redis_cache = MagicMock()
    mock_pod_lock_manager = MagicMock()
    mock_pod_lock_manager.redis_cache = mock_redis_cache
    mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
    mock_pod_lock_manager.release_lock = AsyncMock()

    # No retention setting → _should_delete_spend_logs() returns False before lock is acquired
    cleaner = SpendLogCleanup(general_settings={})
    cleaner.pod_lock_manager = mock_pod_lock_manager

    await cleaner.cleanup_old_spend_logs(mock_prisma_client)

    mock_pod_lock_manager.acquire_lock.assert_not_called()
    mock_pod_lock_manager.release_lock.assert_not_called()


@pytest.mark.asyncio
async def test_integer_retention_treated_as_days():
    """
    An integer value for maximum_spend_logs_retention_period should be treated
    as days (e.g., 3 → '3d' → 259200 seconds).
    """
    cleaner = SpendLogCleanup(
        general_settings={"maximum_spend_logs_retention_period": 3}
    )
    result = cleaner._should_delete_spend_logs()
    assert result is True
    assert cleaner.retention_seconds == 3 * 86400  # 3 days in seconds


def test_string_retention_still_works():
    """
    String values like '3d', '24h', '3600s' should continue to parse correctly.
    """
    cases = [
        ("3d", 3 * 86400),
        ("24h", 24 * 3600),
        ("3600s", 3600),
        ("2w", 2 * 604800),
    ]
    for setting, expected_seconds in cases:
        cleaner = SpendLogCleanup(
            general_settings={"maximum_spend_logs_retention_period": setting}
        )
        assert cleaner._should_delete_spend_logs() is True, f"Failed for {setting}"
        assert (
            cleaner.retention_seconds == expected_seconds
        ), f"Expected {expected_seconds} for {setting}, got {cleaner.retention_seconds}"


@pytest.mark.asyncio
async def test_delete_old_logs_aborts_on_non_int_execute_raw_return():
    """should abort deletion loop immediately when execute_raw returns a non-int
    (e.g. None or dict), preventing an infinite loop."""
    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_db = MagicMock()
    _wire_tx(mock_db)
    mock_db.execute_raw = AsyncMock(return_value=None)
    mock_prisma_client.db = mock_db

    cleaner = SpendLogCleanup(
        general_settings={"maximum_spend_logs_retention_period": "7d"}
    )

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)
    result = await cleaner._delete_old_logs(mock_prisma_client, cutoff_date, _far_deadline())

    assert mock_db.execute_raw.call_count == 1
    assert result.rows_deleted == 0


@pytest.mark.asyncio
async def test_delete_old_logs_continues_on_valid_int_return():
    """should continue deletion loop across batches when execute_raw returns valid int counts."""
    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_db = MagicMock()
    _wire_tx(mock_db)
    mock_db.execute_raw = AsyncMock(side_effect=[500, 300, 0])
    mock_prisma_client.db = mock_db

    cleaner = SpendLogCleanup(
        general_settings={"maximum_spend_logs_retention_period": "7d"}
    )

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)
    result = await cleaner._delete_old_logs(mock_prisma_client, cutoff_date, _far_deadline())

    assert mock_db.execute_raw.call_count == 3
    assert result.rows_deleted == 800


@pytest.mark.asyncio
async def test_delete_old_rows_stops_at_max_batches():
    """The batch cap must halt a cleanup that keeps finding rows, so a huge
    backlog is spread across scheduled runs instead of one unbounded loop, and
    the operator-facing knob must mean exactly the number of statements it names."""
    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_db = MagicMock()
    _wire_tx(mock_db)
    mock_db.execute_raw = AsyncMock(return_value=1000)
    mock_prisma_client.db = mock_db

    cleaner = SpendLogCleanup(
        general_settings={
            "maximum_spend_logs_retention_period": "7d",
            "maximum_spend_logs_cleanup_max_batches": 2,
        }
    )

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)
    result = await cleaner._delete_old_logs(mock_prisma_client, cutoff_date, _far_deadline())

    assert mock_db.execute_raw.call_count == 2
    assert result.rows_deleted == 2000
    assert result.stop_reason == "batch_cap_reached"


@pytest.mark.asyncio
async def test_delete_old_tool_index_rows_deletes_on_composite_key():
    """Tool index rows are derived from spend logs and expire on the same cutoff;
    the delete must match on the table's composite primary key."""
    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_db = MagicMock()
    _wire_tx(mock_db)
    mock_db.execute_raw = AsyncMock(side_effect=[5, 0])
    mock_prisma_client.db = mock_db

    cleaner = SpendLogCleanup(
        general_settings={"maximum_spend_logs_retention_period": "7d"}
    )

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)
    result = await cleaner._delete_old_tool_index_rows(mock_prisma_client, cutoff_date, _far_deadline())

    assert result.rows_deleted == 5
    delete_sql = mock_db.execute_raw.call_args_list[0][0][0]
    assert 'DELETE FROM "LiteLLM_SpendLogToolIndex"' in delete_sql
    assert 'WHERE ("request_id", "tool_name") IN' in delete_sql
    assert '"start_time" <' in delete_sql
    assert mock_db.execute_raw.call_args_list[0][0][1] == cutoff_date


@pytest.mark.asyncio
async def test_delete_old_logs_continues_after_single_batch_failure(monkeypatch):
    """A single batch failure (e.g. DB timeout) must not abort the whole run —
    subsequent batches should still execute and their counts accumulate."""
    import litellm.proxy.db.db_transaction_queue.spend_log_cleanup as cleanup_module

    # Zero out the failure backoff so the test doesn't take ~0.5s of real sleep.
    monkeypatch.setattr(
        cleanup_module, "SPEND_LOG_CLEANUP_BATCH_FAILURE_BACKOFF_SECONDS", 0.0
    )

    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_db = MagicMock()
    _wire_tx(mock_db)
    # batch 1 succeeds, batch 2 raises (one-off DB timeout), batches 3-4 succeed,
    # batch 5 returns 0 → loop exits naturally.
    mock_db.execute_raw = AsyncMock(
        side_effect=[100, TimeoutError("simulated DB timeout"), 200, 50, 0]
    )
    mock_prisma_client.db = mock_db

    cleaner = cleanup_module.SpendLogCleanup(
        general_settings={"maximum_spend_logs_retention_period": "7d"}
    )

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)
    result = await cleaner._delete_old_logs(mock_prisma_client, cutoff_date, _far_deadline())

    # All 5 batches should have been attempted; 100 + 200 + 50 = 350 deleted.
    assert mock_db.execute_raw.call_count == 5
    assert result.rows_deleted == 350


@pytest.mark.asyncio
async def test_delete_old_logs_aborts_after_consecutive_failures(monkeypatch):
    """If batch failures persist for SPEND_LOG_CLEANUP_MAX_CONSECUTIVE_BATCH_FAILURES
    in a row (e.g. DB is down), the loop must abort instead of hot-looping."""
    import litellm.proxy.db.db_transaction_queue.spend_log_cleanup as cleanup_module

    # Lower the threshold so the test is fast and deterministic.
    monkeypatch.setattr(
        cleanup_module, "SPEND_LOG_CLEANUP_MAX_CONSECUTIVE_BATCH_FAILURES", 3
    )
    monkeypatch.setattr(
        cleanup_module, "SPEND_LOG_CLEANUP_BATCH_FAILURE_BACKOFF_SECONDS", 0.0
    )

    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_db = MagicMock()
    _wire_tx(mock_db)
    # Every batch raises — must abort after exactly 3 attempts, not loop forever.
    mock_db.execute_raw = AsyncMock(
        side_effect=ConnectionError("simulated persistent DB outage")
    )
    mock_prisma_client.db = mock_db

    cleaner = cleanup_module.SpendLogCleanup(
        general_settings={"maximum_spend_logs_retention_period": "7d"}
    )

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)
    result = await cleaner._delete_old_logs(mock_prisma_client, cutoff_date, _far_deadline())

    assert mock_db.execute_raw.call_count == 3
    assert result.rows_deleted == 0


@pytest.mark.asyncio
async def test_delete_old_logs_resets_consecutive_failures_on_success(monkeypatch):
    """A success between failures must reset the consecutive-failure counter so
    intermittent timeouts don't trip the abort threshold."""
    import litellm.proxy.db.db_transaction_queue.spend_log_cleanup as cleanup_module

    monkeypatch.setattr(
        cleanup_module, "SPEND_LOG_CLEANUP_MAX_CONSECUTIVE_BATCH_FAILURES", 3
    )
    monkeypatch.setattr(
        cleanup_module, "SPEND_LOG_CLEANUP_BATCH_FAILURE_BACKOFF_SECONDS", 0.0
    )

    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_db = MagicMock()
    _wire_tx(mock_db)
    # Pattern: fail, fail, success (resets counter), fail, fail, success, done.
    # Without reset, three of these would trip abort; with reset, they don't.
    mock_db.execute_raw = AsyncMock(
        side_effect=[
            TimeoutError("t1"),
            TimeoutError("t2"),
            100,
            TimeoutError("t3"),
            TimeoutError("t4"),
            50,
            0,
        ]
    )
    mock_prisma_client.db = mock_db

    cleaner = cleanup_module.SpendLogCleanup(
        general_settings={"maximum_spend_logs_retention_period": "7d"}
    )

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)
    result = await cleaner._delete_old_logs(mock_prisma_client, cutoff_date, _far_deadline())

    assert mock_db.execute_raw.call_count == 7
    assert result.rows_deleted == 150


@pytest.mark.asyncio
async def test_cleanup_uses_logger_exception_for_full_traceback(monkeypatch):
    """The outer error handler must call logger.exception() (not .error(str(e)))
    so Prisma/DB timeouts surface a full traceback and exception type."""
    import litellm.proxy.db.db_transaction_queue.spend_log_cleanup as cleanup_module

    mock_logger = MagicMock()
    monkeypatch.setattr(cleanup_module, "verbose_proxy_logger", mock_logger)

    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    # Force the outer try/except to fire by making _should_delete_spend_logs raise.
    cleaner = cleanup_module.SpendLogCleanup(
        general_settings={"maximum_spend_logs_retention_period": "7d"}
    )
    cleaner.pod_lock_manager = None

    def boom():
        raise RuntimeError("simulated prisma timeout")

    cleaner._should_delete_spend_logs = boom  # type: ignore[assignment]

    await cleaner.cleanup_old_spend_logs(mock_prisma_client)

    assert mock_logger.exception.called, "expected logger.exception() to be called"
    # The exception type name must appear in the formatted args so operators can
    # tell *what* failed, not just "Error during cleanup:".
    call_args = mock_logger.exception.call_args
    formatted = call_args[0][0] % call_args[0][1:]
    assert "RuntimeError" in formatted
    assert "simulated prisma timeout" in formatted


@pytest.mark.asyncio
async def test_cleanup_releases_lock_after_persistent_batch_failures(monkeypatch):
    """Even when batch deletion aborts due to consecutive failures, the pod lock
    must still be released so the next scheduled run isn't permanently blocked."""
    import litellm.proxy.db.db_transaction_queue.spend_log_cleanup as cleanup_module

    monkeypatch.setattr(
        cleanup_module, "SPEND_LOG_CLEANUP_MAX_CONSECUTIVE_BATCH_FAILURES", 2
    )
    monkeypatch.setattr(
        cleanup_module, "SPEND_LOG_CLEANUP_BATCH_FAILURE_BACKOFF_SECONDS", 0.0
    )

    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_db = MagicMock()
    _wire_tx(mock_db)
    mock_db.execute_raw = AsyncMock(side_effect=TimeoutError("DB down"))
    mock_prisma_client.db = mock_db

    mock_pod_lock_manager = MagicMock()
    mock_pod_lock_manager.redis_cache = MagicMock()
    mock_pod_lock_manager.acquire_lock = AsyncMock(return_value=True)
    mock_pod_lock_manager.release_lock = AsyncMock()

    cleaner = cleanup_module.SpendLogCleanup(
        general_settings={"maximum_spend_logs_retention_period": "7d"}
    )
    cleaner.pod_lock_manager = mock_pod_lock_manager

    await cleaner.cleanup_old_spend_logs(mock_prisma_client)

    # Cleanup didn't crash; the abort-after-failures path returned cleanly.
    mock_pod_lock_manager.release_lock.assert_awaited_once()


def test_cleanup_batch_size_env_var(monkeypatch):
    """Ensure batch size is configurable via environment variable"""
    import importlib

    import litellm.constants as constants_module
    import litellm.proxy.db.db_transaction_queue.spend_log_cleanup as cleanup_module

    # Set env var and reload modules to pick up new value
    monkeypatch.setenv("SPEND_LOG_CLEANUP_BATCH_SIZE", "25")
    importlib.reload(constants_module)
    importlib.reload(cleanup_module)

    cleaner = cleanup_module.SpendLogCleanup(general_settings={})
    assert cleaner.batch_size == 25

    # Remove env var and reload to restore default for other tests
    monkeypatch.delenv("SPEND_LOG_CLEANUP_BATCH_SIZE", raising=False)
    importlib.reload(constants_module)
    importlib.reload(cleanup_module)


def _mock_prisma_for_retention(side_effect: list) -> "MagicMock":
    from unittest.mock import AsyncMock, MagicMock

    client = MagicMock()
    _wire_tx(client.db)
    client.db.execute_raw = AsyncMock(side_effect=side_effect)
    return client


@pytest.mark.asyncio
async def test_spend_logs_retention_alone_does_not_touch_the_session_rollup():
    client = _mock_prisma_for_retention([0, 0])
    cleaner = SpendLogCleanup(general_settings={"maximum_spend_logs_retention_period": "7d"})
    cleaner.pod_lock_manager = None
    await cleaner.cleanup_old_spend_logs(client)
    tables = [call[0][0] for call in client.db.execute_raw.call_args_list]
    assert any('"LiteLLM_SpendLogs"' in sql for sql in tables)
    assert not any('"LiteLLM_AutoRouterSession"' in sql for sql in tables)
    assert not any('"LiteLLM_HealthCheckTable"' in sql for sql in tables)


@pytest.mark.asyncio
async def test_session_retention_alone_cleans_only_the_session_rollup():
    client = _mock_prisma_for_retention([0])
    cleaner = SpendLogCleanup(general_settings={"maximum_autorouter_session_retention_period": "365d"})
    cleaner.pod_lock_manager = None
    await cleaner.cleanup_old_spend_logs(client)
    tables = [call[0][0] for call in client.db.execute_raw.call_args_list]
    assert len(tables) == 1
    assert '"LiteLLM_AutoRouterSession"' in tables[0]


@pytest.mark.asyncio
async def test_health_check_retention_alone_cleans_only_the_health_check_table():
    client = _mock_prisma_for_retention([0])
    cleaner = SpendLogCleanup(general_settings={"maximum_health_check_retention_period": "30d"})
    cleaner.pod_lock_manager = None
    await cleaner.cleanup_old_spend_logs(client)
    tables = [call[0][0] for call in client.db.execute_raw.call_args_list]
    assert len(tables) == 1
    assert '"LiteLLM_HealthCheckTable"' in tables[0]
    assert '"health_check_id"' in tables[0]
    assert '"checked_at"' in tables[0]
    cutoff_date = client.db.execute_raw.call_args[0][1]
    expected_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    assert abs((cutoff_date - expected_cutoff).total_seconds()) < 1


@pytest.mark.asyncio
async def test_each_retention_key_cuts_off_at_its_own_horizon():
    client = _mock_prisma_for_retention([0, 0, 0, 0])
    cleaner = SpendLogCleanup(
        general_settings={
            "maximum_spend_logs_retention_period": "7d",
            "maximum_autorouter_session_retention_period": "365d",
            "maximum_health_check_retention_period": "30d",
        }
    )
    cleaner.pod_lock_manager = None
    await cleaner.cleanup_old_spend_logs(client)
    cutoffs = {
        (
            "LiteLLM_AutoRouterSession"
            if '"LiteLLM_AutoRouterSession"' in call[0][0]
            else "LiteLLM_HealthCheckTable"
            if '"LiteLLM_HealthCheckTable"' in call[0][0]
            else "logs"
        ): call[0][1]
        for call in client.db.execute_raw.call_args_list
    }
    now = datetime.now(timezone.utc)
    assert (now - cutoffs["logs"]).days == 7
    assert (now - cutoffs["LiteLLM_AutoRouterSession"]).days == 365
    assert (now - cutoffs["LiteLLM_HealthCheckTable"]).days == 30


@pytest.mark.asyncio
async def test_no_retention_keys_means_no_cleanup_at_all():
    client = _mock_prisma_for_retention([])
    cleaner = SpendLogCleanup(general_settings={})
    cleaner.pod_lock_manager = None
    await cleaner.cleanup_old_spend_logs(client)
    assert client.db.execute_raw.await_count == 0


@pytest.mark.asyncio
async def test_run_budget_stops_the_loop_and_leaves_the_backlog_for_the_next_run():
    """
    The wall-clock budget is the bound that keeps a large backlog from turning
    into one multi-hour run. With rows always available, the loop must stop on
    the deadline rather than on the batch cap, and must report that reason so
    operators can tell a budgeted stop from a drained table.
    """
    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_db = MagicMock()
    _wire_tx(mock_db)
    mock_db.execute_raw = AsyncMock(return_value=1000)
    mock_prisma_client.db = mock_db

    cleaner = SpendLogCleanup(
        general_settings={
            "maximum_spend_logs_retention_period": "7d",
            # Comfortably more batches than a sub-second budget can reach (each
            # batch sleeps 0.1s), but small enough that a broken deadline fails
            # this test in seconds instead of hanging it
            "maximum_spend_logs_cleanup_max_batches": 50,
        }
    )

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)
    started_at = time.monotonic()
    result = await cleaner._delete_old_logs(mock_prisma_client, cutoff_date, time.monotonic() + 0.25)
    elapsed = time.monotonic() - started_at

    assert result.stop_reason == "budget_exhausted"
    assert elapsed < 3, f"budgeted run overran its deadline: {elapsed}s"
    assert mock_db.execute_raw.call_count < 50
    assert result.rows_deleted > 0


@pytest.mark.asyncio
async def test_run_budget_is_shared_across_tables_not_granted_per_table():
    """
    A per-table budget would let a run take N times the configured bound. The
    deadline is computed once per run, so once it is spent on the first table
    the later tables must stop immediately rather than each getting a fresh one.
    """
    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_db = MagicMock()
    _wire_tx(mock_db)
    mock_db.execute_raw = AsyncMock(return_value=1000)
    mock_prisma_client.db = mock_db

    cleaner = SpendLogCleanup(
        general_settings={
            "maximum_spend_logs_retention_period": "7d",
            "maximum_autorouter_session_retention_period": "365d",
            # Comfortably more batches than a sub-second budget can reach (each
            # batch sleeps 0.1s), but small enough that a broken deadline fails
            # this test in seconds instead of hanging it
            "maximum_spend_logs_cleanup_max_batches": 50,
            "maximum_spend_logs_cleanup_run_budget": "1s",
        }
    )
    cleaner.pod_lock_manager = None

    started_at = time.monotonic()
    await cleaner.cleanup_old_spend_logs(mock_prisma_client)
    elapsed = time.monotonic() - started_at

    # three tables are eligible; a per-table budget would push this past 3s
    assert elapsed < 2.5, f"budget was granted per table, not per run: {elapsed}s"
    tables_touched = {call[0][0].split('"')[1] for call in mock_db.execute_raw.call_args_list}
    assert "LiteLLM_SpendLogs" in tables_touched


@pytest.mark.asyncio
async def test_cleanup_groups_share_budget_so_health_checks_still_get_a_delete():
    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_db = MagicMock()
    _wire_tx(mock_db)
    mock_db.execute_raw = AsyncMock(return_value=1000)
    mock_prisma_client.db = mock_db

    cleaner = SpendLogCleanup(
        general_settings={
            "maximum_spend_logs_retention_period": "7d",
            "maximum_health_check_retention_period": "30d",
            "maximum_spend_logs_cleanup_max_batches": 50,
            "maximum_spend_logs_cleanup_run_budget": "1s",
        }
    )
    cleaner.pod_lock_manager = None

    await cleaner.cleanup_old_spend_logs(mock_prisma_client)

    tables_touched = {call[0][0].split('"')[1] for call in mock_db.execute_raw.call_args_list}
    assert "LiteLLM_SpendLogs" in tables_touched
    assert "LiteLLM_HealthCheckTable" in tables_touched


@pytest.mark.asyncio
async def test_each_batch_carries_a_statement_and_lock_timeout():
    """
    A Prisma transaction timeout cannot interrupt a statement already running,
    so the Postgres statement_timeout and lock_timeout are the only things
    stopping one batch from holding row locks and a pooled connection
    indefinitely. Both must be set, inside the batch's own transaction, and
    scoped with SET LOCAL so the pooled connection is left unchanged.
    """
    recorded: list[str] = []

    mock_prisma_client = MagicMock()
    mock_db = MagicMock()

    @asynccontextmanager
    async def _tx():
        tx = MagicMock()

        async def _execute_raw(sql, *args):
            recorded.append(sql.strip())
            return 0

        tx.execute_raw = _execute_raw
        yield tx

    mock_db.tx = _tx
    mock_db.query_raw = AsyncMock(return_value=[{"remaining": 0}])
    mock_prisma_client.db = mock_db

    cleaner = SpendLogCleanup(
        general_settings={
            "maximum_spend_logs_retention_period": "7d",
            "maximum_spend_logs_cleanup_batch_timeout": "12s",
        }
    )

    await cleaner._delete_old_logs(
        mock_prisma_client, datetime.now(timezone.utc) - timedelta(days=7), _far_deadline()
    )

    assert "SET LOCAL statement_timeout = 12000" in recorded
    assert "SET LOCAL lock_timeout = 12000" in recorded
    # the timeouts must precede the delete they are meant to bound
    assert recorded.index("SET LOCAL statement_timeout = 12000") < next(
        i for i, sql in enumerate(recorded) if sql.startswith("DELETE")
    )


@pytest.mark.parametrize(
    "setting_value",
    ["inf", "-inf", "nan", "1e400", "0s", "-5m", "not-a-duration"],
)
def test_a_non_finite_or_non_positive_budget_falls_back_to_the_default(setting_value):
    """
    The knob must not be able to remove the bound it exists to enforce.

    'inf', 'nan' and '1e400' are the spellings that would turn the deadline
    into no deadline at all, and '0s' and '-5m' would make every run stop before
    deleting anything. All of them must land on the default rather than being
    honoured, and the resulting budget must be usable arithmetic.
    """
    cleaner = SpendLogCleanup(
        general_settings={
            "maximum_spend_logs_retention_period": "7d",
            "maximum_spend_logs_cleanup_run_budget": setting_value,
        }
    )

    assert cleaner.run_budget_seconds == SPEND_LOG_CLEANUP_RUN_BUDGET_SECONDS
    assert math.isfinite(cleaner.run_budget_seconds)
    assert cleaner.run_budget_seconds > 0


@pytest.mark.parametrize("setting_value", [0, -1, "abc", "", 2.9])
def test_a_bad_batch_size_falls_back_to_the_default(setting_value):
    """A zero or negative batch size would make every DELETE a no-op and the
    loop spin, so unusable values must fall back rather than be honoured."""
    cleaner = SpendLogCleanup(
        general_settings={
            "maximum_spend_logs_retention_period": "7d",
            "maximum_spend_logs_cleanup_batch_size": setting_value,
        }
    )

    assert cleaner.batch_size >= 1


def test_operator_knobs_override_the_env_defaults():
    """The knobs are meant to be reachable from general_settings (and therefore
    from the admin UI), not only from environment variables."""
    cleaner = SpendLogCleanup(
        general_settings={
            "maximum_spend_logs_retention_period": "7d",
            "maximum_spend_logs_cleanup_batch_size": 250,
            "maximum_spend_logs_cleanup_max_batches": 7,
            "maximum_spend_logs_cleanup_run_budget": "90s",
            "maximum_spend_logs_cleanup_batch_timeout": "2m",
        }
    )

    assert cleaner.batch_size == 250
    assert cleaner.max_batches == 7
    assert cleaner.run_budget_seconds == 90
    assert cleaner.batch_timeout_seconds == 120


_BOUND_SETTING_CASES = (
    ("maximum_spend_logs_cleanup_batch_size", 137, "batch_size", 137),
    ("maximum_spend_logs_cleanup_max_batches", 9, "max_batches", 9),
    ("maximum_spend_logs_cleanup_run_budget", "45s", "run_budget_seconds", 45.0),
    ("maximum_spend_logs_cleanup_batch_timeout", "8s", "batch_timeout_seconds", 8.0),
)


@pytest.mark.parametrize("setting_name, setting_value, attribute, expected", _BOUND_SETTING_CASES)
@pytest.mark.asyncio
async def test_a_bound_changed_after_construction_reaches_the_next_run(
    setting_name, setting_value, attribute, expected
):
    """The scheduler holds one long-lived instance and the config reload mutates
    general_settings in place, so a bound captured at construction would leave
    every dashboard change inert until the process restarts."""
    settings = {"maximum_spend_logs_retention_period": "7d"}
    cleaner = SpendLogCleanup(general_settings=settings)
    cleaner.pod_lock_manager = None
    assert getattr(cleaner, attribute) != expected

    settings[setting_name] = setting_value

    await cleaner.cleanup_old_spend_logs(_mock_prisma_for_retention([0, 0]))

    assert getattr(cleaner, attribute) == expected


@pytest.mark.parametrize("cleared_to_none", [True, False])
@pytest.mark.asyncio
async def test_a_bound_cleared_after_construction_falls_back_to_its_default(cleared_to_none):
    """Blanking the field in the dashboard has to restore the shipped default
    rather than leave the operator's old bound in force, whether the reload
    spells the clear as an explicit None or as an absent key."""
    settings = {"maximum_spend_logs_retention_period": "7d", "maximum_spend_logs_cleanup_batch_size": 137}
    cleaner = SpendLogCleanup(general_settings=settings)
    cleaner.pod_lock_manager = None
    assert cleaner.batch_size == 137

    if cleared_to_none:
        settings["maximum_spend_logs_cleanup_batch_size"] = None
    else:
        del settings["maximum_spend_logs_cleanup_batch_size"]

    await cleaner.cleanup_old_spend_logs(_mock_prisma_for_retention([0, 0]))

    assert cleaner.batch_size == SPEND_LOG_CLEANUP_BATCH_SIZE


def test_every_declared_bound_setting_is_covered_by_a_live_reread_case():
    """A bound added to the declared set without a live-reread case would be
    propagated by the proxy and then ignored by the running job."""
    assert {case[0] for case in _BOUND_SETTING_CASES} == set(SPEND_LOG_CLEANUP_BOUND_SETTINGS)


@pytest.mark.asyncio
async def test_remaining_rows_probe_is_capped_so_it_cannot_scan_the_table():
    """The remaining-eligible-rows metric must never itself become the long
    scan this job exists to avoid, so its probe carries a LIMIT."""
    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_db = MagicMock()
    _wire_tx(mock_db)
    mock_db.execute_raw = AsyncMock(return_value=0)
    mock_prisma_client.db = mock_db

    cleaner = SpendLogCleanup(general_settings={"maximum_spend_logs_retention_period": "7d"})

    await cleaner._delete_old_logs(
        mock_prisma_client, datetime.now(timezone.utc) - timedelta(days=7), _far_deadline()
    )

    count_sql = mock_db.query_raw.call_args[0][0]
    assert "count(*)" in count_sql
    assert "LIMIT $2" in count_sql
    assert mock_db.query_raw.call_args[0][2] == SPEND_LOG_CLEANUP_REMAINING_COUNT_CAP


@pytest.mark.asyncio
async def test_a_run_skipped_because_another_pod_holds_the_lock_is_reported():
    """Operators need to tell "nothing to do" apart from "someone else is doing
    it", so a lock-skipped run is recorded under its own outcome."""
    recorded: list[str] = []
    original_record_run = SpendLogCleanupMetrics.record_run

    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)

    cleaner = SpendLogCleanup(general_settings={"maximum_spend_logs_retention_period": "7d"})
    cleaner.pod_lock_manager = MagicMock()
    cleaner.pod_lock_manager.redis_cache = MagicMock()
    cleaner.pod_lock_manager.acquire_lock = AsyncMock(return_value=False)
    cleaner.pod_lock_manager.release_lock = AsyncMock()

    SpendLogCleanupMetrics.record_run = classmethod(lambda cls, outcome: recorded.append(outcome))
    try:
        await cleaner.cleanup_old_spend_logs(mock_prisma_client)
    finally:
        SpendLogCleanupMetrics.record_run = original_record_run

    assert recorded == ["skipped_locked"]
    cleaner.pod_lock_manager.release_lock.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_outstanding_rows_probe_carries_a_statement_timeout():
    """
    The probe is a statement like any other, so if it were issued bare a slow one
    would hold a connection past the budget the job advertises, which is exactly
    what the bounds exist to prevent. With budget to spare it carries the same
    per-statement timeout the delete batches do.
    """
    recorded: list[str] = []

    mock_prisma_client = MagicMock()
    mock_db = MagicMock()

    @asynccontextmanager
    async def _tx():
        tx = MagicMock()

        async def _execute_raw(sql, *args):
            recorded.append(sql.strip())
            return 0

        async def _query_raw(sql, *args):
            recorded.append(sql.strip())
            return [{"remaining": 7}]

        tx.execute_raw = _execute_raw
        tx.query_raw = _query_raw
        yield tx

    mock_db.tx = _tx
    mock_prisma_client.db = mock_db

    cleaner = SpendLogCleanup(
        general_settings={
            "maximum_spend_logs_retention_period": "7d",
            "maximum_spend_logs_cleanup_batch_timeout": "8s",
        }
    )

    remaining = await cleaner._count_remaining(
        mock_prisma_client,
        datetime.now(timezone.utc) - timedelta(days=7),
        "LiteLLM_SpendLogs",
        "startTime",
        _far_deadline(),
    )

    assert remaining == 7
    count_index = next(i for i, sql in enumerate(recorded) if sql.startswith("SELECT count(*)"))
    assert "SET LOCAL statement_timeout = 8000" in recorded[:count_index], (
        f"the probe ran without a statement timeout: {recorded}"
    )


@pytest.mark.asyncio
async def test_a_statement_timeout_is_clamped_to_the_budget_that_is_left():
    """
    Postgres has no 'stop at time T', only a per-statement duration, so a batch
    issued just under the deadline would run a whole batch timeout past it and
    the run budget would be advisory. Clamping the timeout to the remaining
    budget is what makes the budget a real wall clock.
    """
    recorded: list[str] = []
    client = MagicMock()

    @asynccontextmanager
    async def _tx():
        tx = MagicMock()

        async def _execute_raw(sql, *args):
            recorded.append(sql.strip())
            return 0

        tx.execute_raw = _execute_raw
        tx.query_raw = AsyncMock(return_value=[{"remaining": 0}])
        yield tx

    client.db.tx = _tx

    cleaner = SpendLogCleanup(
        general_settings={
            "maximum_spend_logs_retention_period": "7d",
            "maximum_spend_logs_cleanup_batch_timeout": "30s",
        }
    )

    # Only 2s of budget left against a 30s batch timeout.
    await cleaner._execute_delete_batch(client, "DELETE FROM x", datetime.now(timezone.utc), time.monotonic() + 2)

    timeouts = [sql for sql in recorded if "statement_timeout" in sql]
    assert timeouts, f"no statement timeout was issued: {recorded}"
    issued_ms = int(timeouts[0].split("=")[1].strip())
    assert issued_ms <= 2000, f"the batch was given {issued_ms}ms with only 2000ms of budget left"


@pytest.mark.asyncio
async def test_no_statement_is_issued_once_the_budget_is_spent():
    """
    Every table exits through _finish_table, including the ones a spent run never
    started, so an unconditional probe there would put one more statement per
    table past the bound.
    """
    client = _mock_prisma_for_retention([0, 0])
    cleaner = SpendLogCleanup(general_settings={"maximum_spend_logs_retention_period": "7d"})

    result = await cleaner._finish_table(
        client,
        datetime.now(timezone.utc) - timedelta(days=7),
        "LiteLLM_SpendLogs",
        "startTime",
        123,
        "budget_exhausted",
        time.monotonic() - 1,
    )

    assert result.rows_deleted == 123
    assert result.stop_reason == "budget_exhausted"
    client.db.query_raw.assert_not_called()


@pytest.mark.asyncio
async def test_a_batch_cancelled_by_the_deadline_is_budget_exhaustion_not_a_failure(monkeypatch):
    """
    Clamping the timeout means the last batch of a budget-exhausted run is
    cancelled by the deadline itself. Counting that as a batch failure would
    inflate the failure metric on every such run and walk it toward the abort
    threshold, so it has to be classified as the bound working.
    """
    failures: list[str] = []
    client = MagicMock()
    _wire_tx(client.db)

    # The deadline has to pass DURING the batch, not before it: a deadline
    # already spent is caught by the loop's own check and no batch is ever
    # issued, which would exercise none of the classification under test.
    async def _cancelled_after_the_deadline(sql, *args):
        await asyncio.sleep(0.05)
        raise Exception("canceling statement due to statement timeout")

    client.db.execute_raw = _cancelled_after_the_deadline

    cleaner = SpendLogCleanup(general_settings={"maximum_spend_logs_retention_period": "7d"})
    monkeypatch.setattr(SpendLogCleanupMetrics, "record_batch_failure", lambda table: failures.append(table))

    result = await cleaner._delete_old_logs(
        client, datetime.now(timezone.utc) - timedelta(days=7), time.monotonic() + 0.02
    )

    assert result.stop_reason == "budget_exhausted"
    assert failures == [], f"a deadline cancellation was recorded as a batch failure: {failures}"


@pytest.mark.asyncio
async def test_partition_maintenance_is_skipped_once_the_run_budget_is_spent():
    """
    Dropping a partition is DDL holding an ACCESS EXCLUSIVE lock, and unlike a
    delete batch it cannot be cut short once it has started. A run whose budget is
    already gone must therefore not start it at all; the next tick picks it up.
    """
    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_db = MagicMock()
    _wire_tx(mock_db)
    mock_db.execute_raw = AsyncMock(return_value=0)
    mock_prisma_client.db = mock_db

    partition_manager = MagicMock()
    partition_manager.is_partitioned = AsyncMock(return_value=True)
    partition_manager.ensure_partitions = AsyncMock()
    partition_manager.drop_partitions_older_than = AsyncMock(return_value=[])

    cleaner = SpendLogCleanup(
        general_settings={
            "maximum_spend_logs_retention_period": "7d",
            "use_spend_logs_partitioning": True,
        },
        partition_manager=partition_manager,
    )
    cleaner._should_delete_spend_logs()

    # a deadline already in the past is what a run that spent its budget on an
    # earlier table looks like
    await cleaner._clean_spend_log_tables(mock_prisma_client, time.monotonic() - 1)

    partition_manager.ensure_partitions.assert_not_awaited()
    partition_manager.drop_partitions_older_than.assert_not_awaited()


@pytest.mark.asyncio
async def test_partition_maintenance_still_runs_while_the_run_has_budget():
    """The skip above must be caused by the spent budget, not by breaking the
    partition path outright."""
    mock_prisma_client = MagicMock()
    _wire_tx(mock_prisma_client.db)
    mock_db = MagicMock()
    _wire_tx(mock_db)
    mock_db.execute_raw = AsyncMock(return_value=0)
    mock_prisma_client.db = mock_db

    partition_manager = MagicMock()
    partition_manager.is_partitioned = AsyncMock(return_value=True)
    partition_manager.ensure_partitions = AsyncMock()
    partition_manager.drop_partitions_older_than = AsyncMock(return_value=["LiteLLM_SpendLogs_p20260601"])

    cleaner = SpendLogCleanup(
        general_settings={
            "maximum_spend_logs_retention_period": "7d",
            "use_spend_logs_partitioning": True,
        },
        partition_manager=partition_manager,
    )
    cleaner._should_delete_spend_logs()

    await cleaner._clean_spend_log_tables(mock_prisma_client, _far_deadline())

    partition_manager.ensure_partitions.assert_awaited_once()
    partition_manager.drop_partitions_older_than.assert_awaited_once()


@pytest.mark.parametrize(
    "stop_reasons, expected",
    [
        (("exhausted",), "completed"),
        (("exhausted", "exhausted"), "completed"),
        (("exhausted", "batch_cap_reached"), "batch_cap_reached"),
        (("batch_cap_reached", "exhausted"), "batch_cap_reached"),
        (("exhausted", "budget_exhausted"), "budget_exhausted"),
        (("budget_exhausted", "exhausted"), "budget_exhausted"),
        (("batch_cap_reached", "budget_exhausted"), "budget_exhausted"),
        (("budget_exhausted", "batch_cap_reached"), "budget_exhausted"),
        (("exhausted", "aborted"), "aborted"),
        (("aborted", "exhausted"), "aborted"),
        (("budget_exhausted", "aborted"), "aborted"),
        (("aborted", "budget_exhausted"), "aborted"),
        (("aborted", "budget_exhausted", "batch_cap_reached"), "aborted"),
    ],
)
def test_the_reported_run_outcome_is_the_most_significant_reason_in_any_order(stop_reasons, expected):
    """
    The run outcome answers "why did this run stop", so a table that merely ran
    dry must never mask one that hit a bound, and an abort must outrank both.

    Both orders of every pair are covered because this folds several per-table
    results into one answer: a first-match-wins implementation would pass on
    whichever order happened to be written and fail on its mirror.
    """
    results = tuple(TableCleanupResult(rows_deleted=0, stop_reason=reason) for reason in stop_reasons)
    assert SpendLogCleanup._run_outcome(results) == expected
