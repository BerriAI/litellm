from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.constants import REDIS_SPEND_LOG_BUFFER_KEY
from litellm.proxy.db.db_transaction_queue.spend_log_redis_buffer import (
    SpendLogRedisBuffer,
)


@pytest.fixture
def mock_redis_cache() -> MagicMock:
    cache = MagicMock()
    cache.redis_batch_writing_buffer_key = "test-buffer-key"
    cache.async_rpush = AsyncMock()
    cache.async_lpop = AsyncMock(return_value=None)
    cache.async_llen = AsyncMock(return_value=0)
    return cache


@pytest.mark.asyncio
async def test_buffer_spend_log_row_pushes_to_redis_when_enabled(
    mock_redis_cache: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy_server_mod

    monkeypatch.setattr(
        proxy_server_mod,
        "general_settings",
        {"use_redis_transaction_buffer": True},
    )

    buffer = SpendLogRedisBuffer(redis_cache=mock_redis_cache)
    payload = {"request_id": "req-1", "spend": 12.0}
    await buffer.buffer_spend_log_row(payload)  # type: ignore[arg-type]

    mock_redis_cache.async_rpush.assert_awaited_once()
    assert mock_redis_cache.async_rpush.await_args.args[0] == REDIS_SPEND_LOG_BUFFER_KEY


@pytest.mark.asyncio
async def test_pop_buffered_spend_log_rows_returns_deserialized_rows(
    mock_redis_cache: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy_server_mod
    from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

    monkeypatch.setattr(
        proxy_server_mod,
        "general_settings",
        {"use_redis_transaction_buffer": True},
    )

    payload = {"request_id": "req-2", "spend": 3.5}
    serialized = safe_dumps(payload)
    mock_redis_cache.async_lpop = AsyncMock(side_effect=[serialized, None])

    buffer = SpendLogRedisBuffer(redis_cache=mock_redis_cache)
    rows = await buffer.pop_buffered_spend_log_rows(max_rows=5)

    assert rows == [payload]


@pytest.mark.asyncio
async def test_requeue_spend_log_rows_pushes_each_row(
    mock_redis_cache: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy_server_mod

    monkeypatch.setattr(
        proxy_server_mod,
        "general_settings",
        {"use_redis_transaction_buffer": True},
    )

    buffer = SpendLogRedisBuffer(redis_cache=mock_redis_cache)
    rows = [
        {"request_id": "req-a", "spend": 1.0},
        {"request_id": "req-b", "spend": 2.0},
    ]
    await buffer.requeue_spend_log_rows(rows)  # type: ignore[arg-type]

    assert mock_redis_cache.async_rpush.await_count == 2


@pytest.mark.asyncio
async def test_buffer_spend_log_row_noop_when_disabled(
    mock_redis_cache: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy_server_mod

    monkeypatch.setattr(
        proxy_server_mod,
        "general_settings",
        {"use_redis_transaction_buffer": False},
    )

    buffer = SpendLogRedisBuffer(redis_cache=mock_redis_cache)
    await buffer.buffer_spend_log_row({"request_id": "req-x", "spend": 1.0})  # type: ignore[arg-type]

    mock_redis_cache.async_rpush.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_buffered_row_count_returns_redis_length(
    mock_redis_cache: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy_server_mod

    monkeypatch.setattr(
        proxy_server_mod,
        "general_settings",
        {"use_redis_transaction_buffer": True},
    )
    mock_redis_cache.async_llen = AsyncMock(return_value=7)

    buffer = SpendLogRedisBuffer(redis_cache=mock_redis_cache)
    assert await buffer.get_buffered_row_count() == 7

