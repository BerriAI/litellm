"""
Test cases for spend log cleanup functionality
"""

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.db.db_transaction_queue.spend_log_cleanup import SpendLogCleanup


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
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    # Setup Prisma client
    mock_prisma_client = MagicMock()
    mock_db = MagicMock()

    # Mock spendlogs table
    mock_spendlogs = MagicMock()
    mock_spendlogs.find_many = AsyncMock()
    mock_spendlogs.delete_many = AsyncMock()

    # Create 1500 mocked logs with .request_id
    mock_logs = [SimpleNamespace(request_id=f"req_{i}") for i in range(1500)]
    mock_spendlogs.find_many.side_effect = [
        mock_logs[:1000],  # Batch 1
        mock_logs[1000:],  # Batch 2
        [],  # Done
    ]

    # Wire up mocks
    mock_db.litellm_spendlogs = mock_spendlogs
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

    # Validate batching and deletion
    assert mock_spendlogs.find_many.call_count == 3
    assert mock_spendlogs.delete_many.call_count == 2
    mock_spendlogs.delete_many.assert_any_call(
        where={"request_id": {"in": [f"req_{i}" for i in range(1000)]}}
    )
    mock_spendlogs.delete_many.assert_any_call(
        where={"request_id": {"in": [f"req_{i}" for i in range(1000, 1500)]}}
    )


@pytest.mark.asyncio
async def test_cleanup_old_spend_logs_retention_period_cutoff():
    """
    Test that logs are filtered using correct cutoff based on retention
    """
    # Setup Prisma client
    mock_prisma_client = MagicMock()
    mock_db = MagicMock()
    mock_spendlogs = MagicMock()
    mock_spendlogs.find_many = AsyncMock(return_value=[])
    mock_spendlogs.delete_many = AsyncMock()
    mock_db.litellm_spendlogs = mock_spendlogs
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
    cutoff_date = mock_spendlogs.find_many.call_args[1]["where"]["startTime"]["lt"]
    expected_cutoff = datetime.now(timezone.utc) - timedelta(seconds=86400)
    assert (
        abs((cutoff_date - expected_cutoff).total_seconds()) < 1
    )  # Allow 1 second difference for test execution time


@pytest.mark.asyncio
async def test_cleanup_old_spend_logs_no_retention_period():
    """
    Test that no logs are deleted when no retention period is set
    """
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_spendlogs.find_many = AsyncMock()
    mock_prisma_client.db.litellm_spendlogs.delete = AsyncMock()

    cleaner = SpendLogCleanup(general_settings={})  # no retention
    await cleaner.cleanup_old_spend_logs(mock_prisma_client)

    mock_prisma_client.db.litellm_spendlogs.find_many.assert_not_called()
    mock_prisma_client.db.litellm_spendlogs.delete.assert_not_called()


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
