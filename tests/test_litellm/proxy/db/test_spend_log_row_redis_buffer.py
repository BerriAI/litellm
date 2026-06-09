from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

EXPECTED_REDIS_SPEND_LOG_BUFFER_KEY = "litellm_spend_log_buffer"


def _redis_spend_log_buffer_key() -> str:
    from litellm.constants import REDIS_SPEND_LOG_BUFFER_KEY

    return REDIS_SPEND_LOG_BUFFER_KEY


def test_db_spend_update_writer_exposes_spend_log_redis_buffer() -> None:
    writer = DBSpendUpdateWriter(redis_cache=MagicMock())
    assert hasattr(writer, "spend_log_redis_buffer")


@pytest.mark.asyncio
async def test_insert_spend_log_to_db_buffers_memory_and_redis_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy_server_mod

    monkeypatch.setattr(
        proxy_server_mod,
        "general_settings",
        {"use_redis_transaction_buffer": True},
    )

    mock_redis_cache = MagicMock()
    mock_redis_cache.async_rpush = AsyncMock()
    mock_prisma_client = MagicMock()
    mock_prisma_client.spend_log_transactions = []
    mock_prisma_client._spend_log_transactions_lock = asyncio.Lock()

    writer = DBSpendUpdateWriter(redis_cache=mock_redis_cache)
    payload = {
        "request_id": "req-buffer-test",
        "spend": 0.42,
        "model": "gpt-4o-mini",
    }

    await writer._insert_spend_log_to_db(
        payload=payload,
        prisma_client=mock_prisma_client,
    )

    assert mock_prisma_client.spend_log_transactions == [payload]
    mock_redis_cache.async_rpush.assert_awaited_once()
    assert (
        mock_redis_cache.async_rpush.await_args.args[0]
        == _redis_spend_log_buffer_key()
    )


@pytest.mark.asyncio
async def test_update_spend_logs_job_flushes_rows_buffered_only_in_redis(
    mock_prisma_client: object,
    make_spend_log_row: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.utils import update_spend_logs_job

    import litellm.proxy.guardrails.usage_tracking as guard_mod
    import litellm.proxy.db.spend_log_tool_index as tool_mod

    monkeypatch.setattr(
        guard_mod, "process_spend_logs_guardrail_usage", AsyncMock(), raising=False
    )
    monkeypatch.setattr(
        tool_mod, "process_spend_logs_tool_usage", AsyncMock(), raising=False
    )

    mock_prisma_client.spend_log_transactions = []  # type: ignore[attr-defined]
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock()  # type: ignore[attr-defined]

    redis_row = make_spend_log_row(request_id="redis-only")  # type: ignore[operator]
    redis_buffer = MagicMock()
    redis_buffer.is_enabled = MagicMock(return_value=True)
    redis_buffer.get_buffered_row_count = AsyncMock(return_value=1)
    redis_buffer.pop_buffered_spend_log_rows = AsyncMock(return_value=[redis_row])
    redis_buffer.requeue_spend_log_rows = AsyncMock()

    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    proxy_logging.db_spend_update_writer = MagicMock()
    proxy_logging.db_spend_update_writer.spend_log_redis_buffer = redis_buffer

    await update_spend_logs_job(
        prisma_client=mock_prisma_client,  # type: ignore[arg-type]
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
    )

    assert mock_prisma_client.db.litellm_spendlogs.create_many.await_count == 1  # type: ignore[attr-defined]
    flushed = mock_prisma_client.db.litellm_spendlogs.create_many.await_args.kwargs["data"][0]  # type: ignore[attr-defined]
    assert flushed["request_id"] == "redis-only"
