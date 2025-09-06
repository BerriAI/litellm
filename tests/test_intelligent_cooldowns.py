import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import backoff
import httpx
import pytest
from litellm.proxy import utils as proxy_utils
from litellm.proxy.utils import ProxyUpdateSpend


@pytest.mark.asyncio
async def test_update_spend_logs_success_first_try():
    """
    Test that update_spend_logs succeeds on the first try.
    """
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_spendlogs = MagicMock()
    mock_prisma.db.litellm_spendlogs.create_many = AsyncMock()

    # Simulate spend logs in the queue
    mock_prisma.spend_log_transactions = [{"data": "test"}]

    spend_updater = proxy_utils.ProxyUpdateSpend()
    await spend_updater.update_spend_logs(
        prisma_client=mock_prisma,
        proxy_logging_obj=MagicMock(),
        db_writer_client=None
    )

    mock_prisma.db.litellm_spendlogs.create_many.assert_called_once()
    assert len(mock_prisma.spend_log_transactions) == 0

@pytest.mark.asyncio
async def test_update_spend_logs_success_after_retry_5xx():
    """
    Test that update_spend_logs succeeds after a retry on a 5xx error.
    """
    mock_prisma = MagicMock()
    mock_db_writer = MagicMock()
    # Simulate a 500 error then a success
    mock_db_writer.post = AsyncMock(side_effect=[
        httpx.HTTPStatusError("Server Error", request=MagicMock(), response=httpx.Response(500)),
        MagicMock() # Represents a successful call
    ])

    mock_prisma.spend_log_transactions = [{"data": "test"}]

    spend_updater = proxy_utils.ProxyUpdateSpend()
    await spend_updater.update_spend_logs(
        prisma_client=mock_prisma,
        proxy_logging_obj=MagicMock(),
        db_writer_client=mock_db_writer
    )

    assert mock_db_writer.post.call_count == 2
    assert len(mock_prisma.spend_log_transactions) == 0

@pytest.mark.asyncio
async def test_update_spend_logs_fail_after_max_retries_5xx():
    """
    Test that update_spend_logs fails after the maximum number of retries on 5xx errors.
    """
    mock_prisma = MagicMock()
    mock_db_writer = MagicMock()
    mock_db_writer.post = AsyncMock(side_effect=httpx.HTTPStatusError(
        "Server Error", request=MagicMock(), response=httpx.Response(500)
    ))

    mock_prisma.spend_log_transactions = [{"data": "test"}]

    # Mock the failure handler to be an async function
    mock_failure_handler = AsyncMock()

    with pytest.raises(Exception), \
         patch('litellm.proxy.utils.os.getenv', return_value="http://localhost:8000"), \
         patch('litellm.proxy.utils.proxy_utils.failure_handler', mock_failure_handler):
        spend_updater = proxy_utils.ProxyUpdateSpend()
        await spend_updater.update_spend_logs(
            prisma_client=mock_prisma,
            proxy_logging_obj=MagicMock(),
            db_writer_client=mock_db_writer
        )

    # The backoff library will call it 3 times by default (1 initial + 2 retries)
    assert mock_db_writer.post.call_count == 3
    assert len(mock_prisma.spend_log_transactions) > 0


@pytest.mark.asyncio
async def test_update_spend_logs_success_after_retry_429_with_header():
    """
    Test that update_spend_logs succeeds after a retry on a 429 error with a Retry-After header.
    """
    mock_prisma = MagicMock()
    mock_db_writer = MagicMock()
    headers = {"Retry-After": "1"}
    mock_db_writer.post = AsyncMock(side_effect=[
        httpx.HTTPStatusError(
            "Rate Limit Exceeded",
            request=MagicMock(),
            response=httpx.Response(429, headers=headers)
        ),
        MagicMock()
    ])

    mock_prisma.spend_log_transactions = [{"data": "test"}]

    spend_updater = proxy_utils.ProxyUpdateSpend()
    await spend_updater.update_spend_logs(
        prisma_client=mock_prisma,
        proxy_logging_obj=MagicMock(),
        db_writer_client=mock_db_writer
    )

    assert mock_db_writer.post.call_count == 2
    assert len(mock_prisma.spend_log_transactions) == 0

@pytest.mark.asyncio
async def test_update_spend_logs_fail_first_try_4xx():
    """
    Test that update_spend_logs fails on the first try for a 4xx error (except 429).
    """
    mock_prisma = MagicMock()
    mock_db_writer = MagicMock()
    mock_db_writer.post = AsyncMock(side_effect=httpx.HTTPStatusError(
        "Bad Request", request=MagicMock(), response=httpx.Response(400)
    ))

    mock_prisma.spend_log_transactions = [{"data": "test"}]
    
    # Mock the failure handler to be an async function
    mock_failure_handler = AsyncMock()


    with pytest.raises(Exception), \
         patch('litellm.proxy.utils.os.getenv', return_value="http://localhost:8000"), \
         patch('litellm.proxy.utils.proxy_utils.failure_handler', mock_failure_handler):
        spend_updater = proxy_utils.ProxyUpdateSpend()
        await spend_updater.update_spend_logs(
            prisma_client=mock_prisma,
            proxy_logging_obj=MagicMock(),
            db_writer_client=mock_db_writer
        )

    mock_db_writer.post.assert_called_once()
    assert len(mock_prisma.spend_log_transactions) > 0