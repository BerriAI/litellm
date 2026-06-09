"""Regression tests for durable spend-log flushing and re-queue behavior."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import httpx
import prisma.errors
import pytest

from litellm.proxy.utils import (
    ProxyUpdateSpend,
    _collect_spend_logs_for_flush,
    _dedupe_spend_logs_by_request_id,
    _requeue_failed_spend_logs,
    update_spend_logs_job,
)


def test_dedupe_spend_logs_by_request_id_prefers_first_occurrence() -> None:
    logs = [
        {"request_id": "a", "spend": 1.0},
        {"request_id": "b", "spend": 2.0},
        {"request_id": "a", "spend": 3.0},
        {"spend": 4.0},
    ]
    assert _dedupe_spend_logs_by_request_id(logs) == [
        {"request_id": "a", "spend": 1.0},
        {"request_id": "b", "spend": 2.0},
        {"spend": 4.0},
    ]


@pytest.mark.asyncio
async def test_collect_spend_logs_for_flush_merges_redis_and_memory(
    mock_prisma_client: Any,
    make_spend_log_row: Any,
) -> None:
    mock_prisma_client.spend_log_transactions = [
        make_spend_log_row(request_id="mem-1"),
        make_spend_log_row(request_id="mem-2"),
    ]
    redis_buffer = MagicMock()
    redis_buffer.is_enabled = MagicMock(return_value=True)
    redis_buffer.pop_buffered_spend_log_rows = AsyncMock(
        return_value=[make_spend_log_row(request_id="redis-1")]
    )

    proxy_logging = MagicMock()
    proxy_logging.db_spend_update_writer = MagicMock()
    proxy_logging.db_spend_update_writer.spend_log_redis_buffer = redis_buffer

    collected = await _collect_spend_logs_for_flush(
        prisma_client=mock_prisma_client,
        proxy_logging_obj=proxy_logging,
        max_logs=10,
    )
    assert [row["request_id"] for row in collected] == ["redis-1", "mem-1", "mem-2"]
    assert mock_prisma_client.spend_log_transactions == []
    redis_buffer.pop_buffered_spend_log_rows.assert_awaited_once_with(max_rows=10)


@pytest.mark.asyncio
async def test_collect_spend_logs_for_flush_dedupes_across_sources(
    mock_prisma_client: Any,
    make_spend_log_row: Any,
) -> None:
    duplicate = make_spend_log_row(request_id="dup", spend=1.0)
    mock_prisma_client.spend_log_transactions = [
        make_spend_log_row(request_id="mem-only"),
        make_spend_log_row(request_id="dup", spend=9.0),
    ]
    redis_buffer = MagicMock()
    redis_buffer.is_enabled = MagicMock(return_value=True)
    redis_buffer.pop_buffered_spend_log_rows = AsyncMock(return_value=[duplicate])

    proxy_logging = MagicMock()
    proxy_logging.db_spend_update_writer = MagicMock()
    proxy_logging.db_spend_update_writer.spend_log_redis_buffer = redis_buffer

    collected = await _collect_spend_logs_for_flush(
        prisma_client=mock_prisma_client,
        proxy_logging_obj=proxy_logging,
        max_logs=10,
    )
    assert [row["request_id"] for row in collected] == ["dup", "mem-only"]
    assert collected[0]["spend"] == 1.0


@pytest.mark.asyncio
async def test_requeue_failed_spend_logs_restores_memory_and_redis(
    mock_prisma_client: Any,
    make_spend_log_row: Any,
) -> None:
    failed_rows = [make_spend_log_row(request_id="failed-1")]
    mock_prisma_client.spend_log_transactions = [make_spend_log_row(request_id="queued")]
    redis_buffer = MagicMock()
    redis_buffer.is_enabled = MagicMock(return_value=True)
    redis_buffer.requeue_spend_log_rows = AsyncMock()

    await _requeue_failed_spend_logs(
        prisma_client=mock_prisma_client,
        logs_to_process=failed_rows,
        spend_log_redis_buffer=redis_buffer,
    )
    assert [row["request_id"] for row in mock_prisma_client.spend_log_transactions] == [
        "failed-1",
        "queued",
    ]
    redis_buffer.requeue_spend_log_rows.assert_awaited_once_with(failed_rows)


@pytest.mark.asyncio
async def test_update_spend_logs_job_processes_redis_only_queue(
    mock_prisma_client: Any,
    make_spend_log_row: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.guardrails.usage_tracking as guard_mod
    import litellm.proxy.db.spend_log_tool_index as tool_mod
    import litellm.proxy.utils as utils_mod

    mock_prisma_client.spend_log_transactions = []
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock()

    redis_row = make_spend_log_row(request_id="redis-only")
    redis_buffer = MagicMock()
    redis_buffer.is_enabled = MagicMock(return_value=True)
    redis_buffer.get_buffered_row_count = AsyncMock(return_value=1)
    redis_buffer.pop_buffered_spend_log_rows = AsyncMock(return_value=[redis_row])
    redis_buffer.requeue_spend_log_rows = AsyncMock()

    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    proxy_logging.db_spend_update_writer = MagicMock()
    proxy_logging.db_spend_update_writer.spend_log_redis_buffer = redis_buffer

    monkeypatch.setattr(
        guard_mod, "process_spend_logs_guardrail_usage", AsyncMock(), raising=False
    )
    monkeypatch.setattr(
        tool_mod, "process_spend_logs_tool_usage", AsyncMock(), raising=False
    )

    await update_spend_logs_job(
        prisma_client=mock_prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
    )
    assert mock_prisma_client.db.litellm_spendlogs.create_many.await_count == 1
    assert (
        mock_prisma_client.db.litellm_spendlogs.create_many.await_args.kwargs["data"][0][
            "request_id"
        ]
        == "redis-only"
    )
    redis_buffer.requeue_spend_log_rows.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_spend_logs_job_requeues_after_retryable_flush_failure(
    mock_prisma_client: Any,
    make_spend_log_row: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.utils as utils_mod

    async def _fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(utils_mod.asyncio, "sleep", _fake_sleep)

    failed_row = make_spend_log_row(request_id="lost-if-not-requeued")
    mock_prisma_client.spend_log_transactions = [failed_row]
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock(
        side_effect=httpx.ReadError("connection reset")
    )

    redis_buffer = MagicMock()
    redis_buffer.is_enabled = MagicMock(return_value=True)
    redis_buffer.get_buffered_row_count = AsyncMock(return_value=0)
    redis_buffer.pop_buffered_spend_log_rows = AsyncMock(return_value=[])
    redis_buffer.requeue_spend_log_rows = AsyncMock()

    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    proxy_logging.db_spend_update_writer = MagicMock()
    proxy_logging.db_spend_update_writer.spend_log_redis_buffer = redis_buffer

    with pytest.raises(httpx.ReadError):
        await update_spend_logs_job(
            prisma_client=mock_prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging,
        )

    assert mock_prisma_client.spend_log_transactions == [failed_row]
    redis_buffer.requeue_spend_log_rows.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_spend_logs_does_not_requeue_non_retryable_errors(
    mock_prisma_client: Any,
    make_spend_log_row: Any,
) -> None:
    failed_logs = [make_spend_log_row(request_id="bad-row")]
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock(
        side_effect=ValueError("invalid spend log row")
    )
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    proxy_logging.db_spend_update_writer = MagicMock()
    proxy_logging.db_spend_update_writer.spend_log_redis_buffer = MagicMock()

    with pytest.raises(ValueError, match="invalid spend log row"):
        await ProxyUpdateSpend.update_spend_logs(
            n_retry_times=0,
            prisma_client=mock_prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging,
            logs_to_process=failed_logs,
        )
    assert mock_prisma_client.spend_log_transactions == []


@pytest.mark.asyncio
async def test_update_spend_logs_retries_deadlock_then_succeeds_without_requeue(
    mock_prisma_client: Any,
    make_spend_log_row: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.utils as utils_mod

    async def _fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(utils_mod.asyncio, "sleep", _fake_sleep)

    create_many = AsyncMock(
        side_effect=[
            prisma.errors.PrismaError("deadlock detected"),
            None,
        ]
    )
    mock_prisma_client.db.litellm_spendlogs.create_many = create_many
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    proxy_logging.db_spend_update_writer = MagicMock()
    proxy_logging.db_spend_update_writer.spend_log_redis_buffer = MagicMock()
    proxy_logging.db_spend_update_writer.spend_log_redis_buffer.requeue_spend_log_rows = AsyncMock()

    logs = [make_spend_log_row(request_id="r1")]
    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=1,
        prisma_client=mock_prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
        logs_to_process=logs,
    )
    assert create_many.await_count == 2
    assert mock_prisma_client.spend_log_transactions == []
    proxy_logging.db_spend_update_writer.spend_log_redis_buffer.requeue_spend_log_rows.assert_not_awaited()
