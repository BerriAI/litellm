from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.utils import update_daily_tag_spend
from litellm.proxy._types import DailyTagSpendTransaction
import httpx
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter


@pytest.mark.asyncio
async def test_update_daily_tag_spend_delegates_to_tag_commit_writer():
    prisma_client = MagicMock()
    proxy_logging_obj = MagicMock()
    redis_update_buffer = MagicMock()
    redis_update_buffer._should_commit_spend_updates_to_redis.return_value = False
    proxy_logging_obj.db_spend_update_writer = MagicMock()
    proxy_logging_obj.db_spend_update_writer.redis_update_buffer = redis_update_buffer
    proxy_logging_obj.db_spend_update_writer._commit_daily_tag_spend_to_db = AsyncMock()
    proxy_logging_obj.db_spend_update_writer._commit_daily_tag_spend_to_db_with_redis = (
        AsyncMock()
    )

    await update_daily_tag_spend(
        prisma_client,
        proxy_logging_obj,
    )

    proxy_logging_obj.db_spend_update_writer._commit_daily_tag_spend_to_db.assert_awaited_once_with(
        prisma_client=prisma_client,
        n_retry_times=3,
        proxy_logging_obj=proxy_logging_obj,
    )
    proxy_logging_obj.db_spend_update_writer._commit_daily_tag_spend_to_db_with_redis.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_daily_tag_spend_logs_error_and_does_not_raise():
    prisma_client = MagicMock()
    proxy_logging_obj = MagicMock()
    redis_update_buffer = MagicMock()
    redis_update_buffer._should_commit_spend_updates_to_redis.return_value = False
    proxy_logging_obj.db_spend_update_writer = MagicMock()
    proxy_logging_obj.db_spend_update_writer.redis_update_buffer = redis_update_buffer
    proxy_logging_obj.db_spend_update_writer._commit_daily_tag_spend_to_db = AsyncMock(
        side_effect=ValueError("boom")
    )
    proxy_logging_obj.db_spend_update_writer._commit_daily_tag_spend_to_db_with_redis = (
        AsyncMock()
    )

    with patch("litellm.proxy.utils.verbose_proxy_logger.error") as error_logger:
        await update_daily_tag_spend(
            prisma_client,
            proxy_logging_obj,
        )

    proxy_logging_obj.db_spend_update_writer._commit_daily_tag_spend_to_db.assert_awaited_once()
    error_logger.assert_called_once()


@pytest.mark.asyncio
async def test_update_daily_tag_spend_uses_redis_writer_when_enabled():
    prisma_client = MagicMock()
    proxy_logging_obj = MagicMock()
    redis_update_buffer = MagicMock()
    redis_update_buffer._should_commit_spend_updates_to_redis.return_value = True
    proxy_logging_obj.db_spend_update_writer = MagicMock()
    proxy_logging_obj.db_spend_update_writer._commit_daily_tag_spend_to_db = AsyncMock()
    proxy_logging_obj.db_spend_update_writer.redis_update_buffer = redis_update_buffer
    proxy_logging_obj.db_spend_update_writer._commit_daily_tag_spend_to_db_with_redis = (
        AsyncMock()
    )

    await update_daily_tag_spend(
        prisma_client,
        proxy_logging_obj,
    )

    proxy_logging_obj.db_spend_update_writer._commit_daily_tag_spend_to_db_with_redis.assert_awaited_once_with(
        prisma_client=prisma_client,
        n_retry_times=3,
        proxy_logging_obj=proxy_logging_obj,
    )
    proxy_logging_obj.db_spend_update_writer._commit_daily_tag_spend_to_db.assert_not_awaited()


@pytest.mark.asyncio
async def test_daily_tag_spend_retries_then_succeeds():
    prisma_client = MagicMock()
    proxy_logging_obj = MagicMock()

    mock_batcher = MagicMock()
    mock_table = MagicMock()
    mock_batcher.litellm_dailytagspend = mock_table

    # Fail entering batch context 3 times with retryable DB errors, then succeed.
    prisma_client.db.batch_.return_value.__aenter__ = AsyncMock(
        side_effect=[
            httpx.ConnectError("x"),
            httpx.ConnectError("x"),
            httpx.ConnectError("x"),
            mock_batcher,
        ]
    )

    daily_spend_transactions: Dict[str, DailyTagSpendTransaction] = {
        "k": {
            "tag": "prod-tag",
            "date": "2026-04-03",
            "api_key": "key-1",
            "model": "gpt-4o",
            "model_group": None,
            "custom_llm_provider": "openai",
            "mcp_namespaced_tool_name": "",
            "endpoint": "",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "spend": 0.01,
            "api_requests": 1,
            "successful_requests": 1,
            "failed_requests": 0,
            "request_id": None,
        }
    }

    with (
        patch("asyncio.sleep", new_callable=AsyncMock) as sleep_mock,
        patch("random.uniform", return_value=0),
    ):
        await DBSpendUpdateWriter.update_daily_tag_spend(
            n_retry_times=3,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
            daily_spend_transactions=daily_spend_transactions,
        )

    assert prisma_client.db.batch_.return_value.__aenter__.await_count == 4
    assert sleep_mock.await_count == 3
    mock_table.upsert.assert_called_once()
