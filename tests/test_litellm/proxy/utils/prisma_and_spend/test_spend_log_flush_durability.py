"""Regression tests for durable spend-log flushing.

These tests pin the behavior finance needs after high-load DB incidents:
  (1) failed flush batches are re-queued instead of dropped
  (2) deadlock and pool-timeout errors are retried, not only httpx connection errors
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import prisma.errors
import pytest

from litellm.proxy.utils import ProxyUpdateSpend, update_spend_logs_job


@pytest.mark.parametrize(
    "error,expected_retryable",
    [
        (httpx.ConnectError("connect"), True),
        (httpx.ReadError("read"), True),
        (httpx.PoolTimeout("pool timeout"), True),
        (prisma.errors.PrismaError("deadlock detected"), True),
        (prisma.errors.PrismaError("pool timeout waiting for connection"), True),
        (ValueError("invalid spend log row"), False),
    ],
)
def test_spend_log_flush_retryable_error_classifier(
    error: Exception, expected_retryable: bool
) -> None:
    from litellm.proxy._types import is_spend_log_flush_retryable_error

    assert is_spend_log_flush_retryable_error(error) is expected_retryable


@pytest.mark.asyncio
async def test_update_spend_logs_retries_on_deadlock_error(
    mock_prisma_client: object,
    make_spend_log_row: object,
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
    mock_prisma_client.db.litellm_spendlogs.create_many = create_many  # type: ignore[attr-defined]
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    logs = [make_spend_log_row(request_id="r1")]  # type: ignore[operator]

    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=1,
        prisma_client=mock_prisma_client,  # type: ignore[arg-type]
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
        logs_to_process=logs,
    )

    assert create_many.await_count == 2


@pytest.mark.asyncio
async def test_update_spend_logs_retries_on_pool_timeout(
    mock_prisma_client: object,
    make_spend_log_row: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.utils as utils_mod

    async def _fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(utils_mod.asyncio, "sleep", _fake_sleep)

    create_many = AsyncMock(
        side_effect=[
            httpx.PoolTimeout("pool timeout"),
            None,
        ]
    )
    mock_prisma_client.db.litellm_spendlogs.create_many = create_many  # type: ignore[attr-defined]
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    logs = [make_spend_log_row(request_id="r1")]  # type: ignore[operator]

    await ProxyUpdateSpend.update_spend_logs(
        n_retry_times=1,
        prisma_client=mock_prisma_client,  # type: ignore[arg-type]
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
        logs_to_process=logs,
    )

    assert create_many.await_count == 2


@pytest.mark.asyncio
async def test_update_spend_logs_requeues_after_retryable_flush_failure(
    mock_prisma_client: object,
    make_spend_log_row: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.utils as utils_mod

    async def _fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(utils_mod.asyncio, "sleep", _fake_sleep)

    failed_logs = [make_spend_log_row(request_id="r1")]  # type: ignore[operator]
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock(  # type: ignore[attr-defined]
        side_effect=httpx.ReadError("network blip")
    )
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()

    with pytest.raises(httpx.ReadError):
        await ProxyUpdateSpend.update_spend_logs(
            n_retry_times=1,
            prisma_client=mock_prisma_client,  # type: ignore[arg-type]
            db_writer_client=None,
            proxy_logging_obj=proxy_logging,
            logs_to_process=failed_logs,
        )

    assert mock_prisma_client.spend_log_transactions == failed_logs  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_update_spend_logs_job_requeues_after_retryable_flush_failure(
    mock_prisma_client: object,
    make_spend_log_row: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.guardrails.usage_tracking as guard_mod
    import litellm.proxy.db.spend_log_tool_index as tool_mod
    import litellm.proxy.utils as utils_mod

    async def _fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(utils_mod.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(
        guard_mod, "process_spend_logs_guardrail_usage", AsyncMock(), raising=False
    )
    monkeypatch.setattr(
        tool_mod, "process_spend_logs_tool_usage", AsyncMock(), raising=False
    )

    failed_row = make_spend_log_row(request_id="lost-if-not-requeued")  # type: ignore[operator]
    mock_prisma_client.spend_log_transactions = [failed_row]  # type: ignore[attr-defined]
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock(  # type: ignore[attr-defined]
        side_effect=httpx.ReadError("connection reset")
    )

    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    proxy_logging.db_spend_update_writer = MagicMock()
    proxy_logging.db_spend_update_writer.spend_log_redis_buffer = None

    with pytest.raises(httpx.ReadError):
        await update_spend_logs_job(
            prisma_client=mock_prisma_client,  # type: ignore[arg-type]
            db_writer_client=None,
            proxy_logging_obj=proxy_logging,
        )

    assert mock_prisma_client.spend_log_transactions == [failed_row]  # type: ignore[attr-defined]
