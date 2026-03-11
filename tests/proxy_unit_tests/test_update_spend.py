import asyncio
import os
import sys
from unittest.mock import Mock
from litellm.proxy.utils import _get_redoc_url, _get_docs_url

import pytest
from fastapi import Request

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from unittest.mock import MagicMock, patch, AsyncMock


import httpx
from litellm.proxy.utils import update_spend, DB_CONNECTION_ERROR_TYPES


class MockPrismaClient:
    def __init__(self):
        # Create AsyncMock for db operations
        self.db = AsyncMock()
        self.db.litellm_spendlogs = AsyncMock()
        self.db.litellm_spendlogs.create_many = AsyncMock()
        
        # Initialize transaction lists
        self.spend_log_transactions = []
        self.daily_user_spend_transactions = {}
        
        # Add lock for spend_log_transactions (matches real PrismaClient)
        import asyncio
        self._spend_log_transactions_lock = asyncio.Lock()

    def jsonify_object(self, obj):
        return obj

    def add_spend_log_transaction_to_daily_user_transaction(self, payload):
        # Mock implementation
        pass


def create_mock_proxy_logging():
    print("creating mock proxy logging")
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.failure_handler = AsyncMock()
    proxy_logging_obj.db_spend_update_writer = AsyncMock()
    proxy_logging_obj.db_spend_update_writer.db_update_spend_transaction_handler = AsyncMock()
    print("returning proxy logging obj")
    return proxy_logging_obj


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type",
    [
        httpx.ConnectError("Failed to connect"),
        httpx.ReadError("Failed to read response"),
        httpx.ReadTimeout("Request timed out"),
    ],
)
async def test_update_spend_logs_connection_errors(error_type):
    """Test retry mechanism for different connection error types"""
    # Setup
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()
    
    # Create AsyncMock for db_spend_update_writer
    proxy_logging_obj.db_spend_update_writer = AsyncMock()
    proxy_logging_obj.db_spend_update_writer.db_update_spend_transaction_handler = AsyncMock()

    # Add test spend logs
    prisma_client.spend_log_transactions = [
        {"id": "1", "spend": 10},
        {"id": "2", "spend": 20},
    ]

    # Mock the database to fail with connection error twice then succeed
    create_many_mock = AsyncMock()
    create_many_mock.side_effect = [
        error_type,  # First attempt fails
        error_type,  # Second attempt fails
        error_type,  # Third attempt fails
        None,  # Fourth attempt succeeds
    ]

    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    # Execute
    await update_spend(prisma_client, None, proxy_logging_obj)

    # Verify
    assert create_many_mock.call_count == 4  # Should have tried 3 times
    assert (
        len(prisma_client.spend_log_transactions) == 0
    )  # Should have cleared after success


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type",
    [
        httpx.ConnectError("Failed to connect"),
        httpx.ReadError("Failed to read response"),
        httpx.ReadTimeout("Request timed out"),
    ],
)
async def test_update_spend_logs_max_retries_exceeded(error_type):
    """Test that each connection error type properly fails after max retries"""
    # Setup
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()

    # Add test spend logs
    prisma_client.spend_log_transactions = [
        {"id": "1", "spend": 10},
        {"id": "2", "spend": 20},
    ]

    # Mock the database to always fail
    create_many_mock = AsyncMock(side_effect=error_type)

    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    # Execute and verify it raises after max retries
    with pytest.raises(type(error_type)) as exc_info:
        await update_spend(prisma_client, None, proxy_logging_obj)

    # Verify error message matches
    assert str(exc_info.value) == str(error_type)
    # Verify retry attempts (initial try + 4 retries)
    assert create_many_mock.call_count == 4

    await asyncio.sleep(2)
    # Verify failure handler was called
    assert proxy_logging_obj.failure_handler.call_count == 1


@pytest.mark.asyncio
async def test_update_spend_logs_non_connection_error():
    """Test handling of non-connection related errors"""
    # Setup
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()

    # Add test spend logs
    prisma_client.spend_log_transactions = [
        {"id": "1", "spend": 10},
        {"id": "2", "spend": 20},
    ]

    # Mock a different type of error (not connection-related)
    unexpected_error = ValueError("Unexpected database error")
    create_many_mock = AsyncMock(side_effect=unexpected_error)

    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    # Execute and verify it raises immediately without retrying
    with pytest.raises(ValueError) as exc_info:
        await update_spend(prisma_client, None, proxy_logging_obj)

    # Verify error message
    assert str(exc_info.value) == "Unexpected database error"
    # Verify only tried once (no retries for non-connection errors)
    assert create_many_mock.call_count == 1
    # Verify failure handler was called
    assert proxy_logging_obj.failure_handler.called


@pytest.mark.asyncio
async def test_update_spend_logs_exponential_backoff():
    """Test that exponential backoff is working correctly"""
    # Setup
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()

    # Add test spend logs
    prisma_client.spend_log_transactions = [{"id": "1", "spend": 10}]

    # Track sleep times
    sleep_times = []

    # Mock asyncio.sleep to track delay times
    async def mock_sleep(seconds):
        sleep_times.append(seconds)

    # Mock the database to fail with connection errors
    create_many_mock = AsyncMock(
        side_effect=[
            httpx.ConnectError("Failed to connect"),  # First attempt
            httpx.ConnectError("Failed to connect"),  # Second attempt
            None,  # Third attempt succeeds
        ]
    )

    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    # Apply mocks
    with patch("asyncio.sleep", mock_sleep):
        await update_spend(prisma_client, None, proxy_logging_obj)

    # Verify exponential backoff
    assert len(sleep_times) == 2  # Should have slept twice
    assert sleep_times[0] >= 1 and sleep_times[0] <= 2  # First retry after 2^0~2^1 seconds
    assert sleep_times[1] >= 2 and sleep_times[1] <= 4 # Second retry after 2^1~2^2 seconds


@pytest.mark.asyncio
async def test_update_spend_logs_multiple_batches_success():
    """
    Test successful processing of multiple batches of spend logs

    Code sets batch size to 1000. This test creates 1500 logs, so it should make 2 batches.
    """
    # Setup
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()

    # Create 1500 test spend logs (1.5x BATCH_SIZE)
    prisma_client.spend_log_transactions = [
        {"id": str(i), "spend": 10} for i in range(1500)
    ]

    create_many_mock = AsyncMock(return_value=None)
    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    # Execute
    await update_spend(prisma_client, None, proxy_logging_obj)

    # Verify
    assert create_many_mock.call_count == 2  # Should have made 2 batch calls

    # Get the actual data from each batch call
    first_batch = create_many_mock.call_args_list[0][1]["data"]
    second_batch = create_many_mock.call_args_list[1][1]["data"]

    # Verify batch sizes
    assert len(first_batch) == 1000
    assert len(second_batch) == 500

    # Verify exact IDs in each batch
    expected_first_batch_ids = {str(i) for i in range(1000)}
    expected_second_batch_ids = {str(i) for i in range(1000, 1500)}

    actual_first_batch_ids = {item["id"] for item in first_batch}
    actual_second_batch_ids = {item["id"] for item in second_batch}

    assert actual_first_batch_ids == expected_first_batch_ids
    assert actual_second_batch_ids == expected_second_batch_ids

    # Verify all logs were processed
    assert len(prisma_client.spend_log_transactions) == 0


@pytest.mark.asyncio
async def test_update_spend_logs_multiple_batches_with_failure():
    """
    Test processing of multiple batches where one batch fails.
    Creates 4000 logs (4 batches) with one batch failing but eventually succeeding after retry.
    """
    # Setup
    prisma_client = MockPrismaClient()
    proxy_logging_obj = create_mock_proxy_logging()

    # Create 4000 test spend logs (4x BATCH_SIZE)
    prisma_client.spend_log_transactions = [
        {"id": str(i), "spend": 10} for i in range(4000)
    ]

    # Mock to fail on second batch first attempt, then succeed
    call_count = 0

    async def create_many_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        # Fail on the second batch's first attempt
        if call_count == 2:
            raise httpx.ConnectError("Failed to connect")
        return None

    create_many_mock = AsyncMock(side_effect=create_many_side_effect)
    prisma_client.db.litellm_spendlogs.create_many = create_many_mock

    # Execute
    await update_spend(prisma_client, None, proxy_logging_obj)

    # Verify
    assert create_many_mock.call_count == 6  # 4 batches + 2 retries for failed batch

    # Verify all batches were processed
    all_processed_logs = []
    for call in create_many_mock.call_args_list:
        all_processed_logs.extend(call[1]["data"])

    # Verify all IDs were processed
    processed_ids = {item["id"] for item in all_processed_logs}

    # these should have ids 0-3999
    print("all processed ids", sorted(processed_ids, key=int))
    expected_ids = {str(i) for i in range(4000)}
    assert processed_ids == expected_ids

    # Verify all logs were cleared from transactions
    assert len(prisma_client.spend_log_transactions) == 0
