"""
Test cases for spend log cleanup functionality
"""

import pytest
from datetime import datetime, timedelta, UTC
from litellm.proxy.db.db_transaction_queue.spend_log_cleanup import SpendLogCleanup
from unittest.mock import MagicMock, AsyncMock

@pytest.mark.asyncio
async def test_should_delete_spend_logs():
    # Test case 1: No retention set
    cleaner = SpendLogCleanup(general_settings={})
    assert cleaner._should_delete_spend_logs() is False

    # Test case 2: Valid seconds string
    cleaner = SpendLogCleanup(general_settings={"maximum_spend_logs_retention_period": "3600s"})
    assert cleaner._should_delete_spend_logs() is True

    # Test case 3: Valid days string
    cleaner = SpendLogCleanup(general_settings={"maximum_spend_logs_retention_period": "30d"})
    assert cleaner._should_delete_spend_logs() is True

    # Test case 4: Valid hours string
    cleaner = SpendLogCleanup(general_settings={"maximum_spend_logs_retention_period": "24h"})
    assert cleaner._should_delete_spend_logs() is True

    # Test case 5: Invalid format
    cleaner = SpendLogCleanup(general_settings={"maximum_spend_logs_retention_period": "invalid"})
    assert cleaner._should_delete_spend_logs() is False


@pytest.mark.asyncio
async def test_cleanup_old_spend_logs_batch_deletion():
    from types import SimpleNamespace

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
        []  # Done
    ]

    # Wire up mocks
    mock_db.litellm_spendlogs = mock_spendlogs
    mock_prisma_client.db = mock_db

    # Run cleanup
    test_settings = {"maximum_spend_logs_retention_period": "7d"}
    cleaner = SpendLogCleanup(general_settings=test_settings)
    assert cleaner._should_delete_spend_logs() is True
    await cleaner.cleanup_old_spend_logs(mock_prisma_client)

    # Validate batching and deletion
    assert mock_spendlogs.find_many.call_count == 3
    assert mock_spendlogs.delete_many.await_count == 2


@pytest.mark.asyncio
async def test_cleanup_old_spend_logs_retention_period_cutoff():
    """
    Test that logs are filtered using correct cutoff based on retention
    """
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_spendlogs.find_many = AsyncMock(return_value=[])
    mock_prisma_client.db.litellm_spendlogs.delete = AsyncMock()

    test_settings = {"maximum_spend_logs_retention_period": "24h"}
    cleaner = SpendLogCleanup(general_settings=test_settings)
    assert cleaner._should_delete_spend_logs() is True
    await cleaner.cleanup_old_spend_logs(mock_prisma_client)

    cutoff_date = mock_prisma_client.db.litellm_spendlogs.find_many.call_args[1]["where"]["startTime"]["lt"]
    expected_cutoff = datetime.now(UTC) - timedelta(hours=24)
    assert abs((cutoff_date - expected_cutoff).total_seconds()) < 5


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