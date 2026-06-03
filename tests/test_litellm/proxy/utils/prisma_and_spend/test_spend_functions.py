"""Pin module-level spend functions.

Symbols pinned here:
  - ``update_spend``
  - ``update_daily_tag_spend``
  - ``update_spend_logs_job``
  - ``_monitor_spend_logs_queue``
  - ``_raise_failed_update_spend_exception``
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.utils import (
    _monitor_spend_logs_queue,
    _raise_failed_update_spend_exception,
    update_daily_tag_spend,
    update_spend,
    update_spend_logs_job,
)


@pytest.mark.asyncio
async def test_update_spend_invokes_writer_and_skips_empty_queue(
    mock_prisma_client: Any,
) -> None:
    proxy_logging = MagicMock()
    proxy_logging.db_spend_update_writer = MagicMock()
    proxy_logging.db_spend_update_writer.db_update_spend_transaction_handler = AsyncMock()
    mock_prisma_client.spend_log_transactions = []

    await update_spend(
        prisma_client=mock_prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
    )
    handler = proxy_logging.db_spend_update_writer.db_update_spend_transaction_handler
    pinned = {
        "handler_called": handler.await_count,
        "handler_kwargs": handler.await_args.kwargs,
        "queue_empty": mock_prisma_client.spend_log_transactions,
    }
    assert pinned == {
        "handler_called": 1,
        "handler_kwargs": {
            "prisma_client": mock_prisma_client,
            "n_retry_times": 3,
            "proxy_logging_obj": proxy_logging,
        },
        "queue_empty": [],
    }


@pytest.mark.asyncio
async def test_update_spend_processes_logs_when_queue_nonempty(
    mock_prisma_client: Any, make_spend_log_row: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy_logging = MagicMock()
    proxy_logging.db_spend_update_writer = MagicMock()
    proxy_logging.db_spend_update_writer.db_update_spend_transaction_handler = AsyncMock()
    mock_prisma_client.spend_log_transactions = [make_spend_log_row(request_id="r1")]

    import litellm.proxy.utils as utils_mod

    job_mock = AsyncMock()
    monkeypatch.setattr(utils_mod, "update_spend_logs_job", job_mock)

    await update_spend(
        prisma_client=mock_prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
    )
    assert job_mock.await_count == 1


@pytest.mark.asyncio
async def test_update_spend_handler_failure_propagates(
    mock_prisma_client: Any,
) -> None:
    proxy_logging = MagicMock()
    proxy_logging.db_spend_update_writer = MagicMock()
    proxy_logging.db_spend_update_writer.db_update_spend_transaction_handler = AsyncMock(
        side_effect=RuntimeError("handler down")
    )
    with pytest.raises(RuntimeError, match="handler down"):
        await update_spend(
            prisma_client=mock_prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging,
        )


@pytest.mark.asyncio
async def test_update_daily_tag_spend_redis_path_when_buffered(
    mock_prisma_client: Any,
) -> None:
    proxy_logging = MagicMock()
    writer = MagicMock()
    proxy_logging.db_spend_update_writer = writer
    writer.redis_update_buffer = MagicMock()
    writer.redis_update_buffer._should_commit_spend_updates_to_redis = MagicMock(
        return_value=True
    )
    writer._commit_daily_tag_spend_to_db_with_redis = AsyncMock()
    writer._commit_daily_tag_spend_to_db = AsyncMock()

    await update_daily_tag_spend(
        prisma_client=mock_prisma_client, proxy_logging_obj=proxy_logging
    )
    redis_kwargs = writer._commit_daily_tag_spend_to_db_with_redis.await_args.kwargs
    pinned = {
        "redis_calls": writer._commit_daily_tag_spend_to_db_with_redis.await_count,
        "direct_calls": writer._commit_daily_tag_spend_to_db.await_count,
        "redis_kwargs_keys": sorted(redis_kwargs.keys()),
        "redis_n_retries": redis_kwargs["n_retry_times"],
    }
    assert pinned == {
        "redis_calls": 1,
        "direct_calls": 0,
        "redis_kwargs_keys": sorted(
            ["prisma_client", "n_retry_times", "proxy_logging_obj"]
        ),
        "redis_n_retries": 3,
    }


@pytest.mark.asyncio
async def test_update_daily_tag_spend_direct_path_when_no_redis(
    mock_prisma_client: Any,
) -> None:
    proxy_logging = MagicMock()
    writer = MagicMock()
    proxy_logging.db_spend_update_writer = writer
    writer.redis_update_buffer = MagicMock()
    writer.redis_update_buffer._should_commit_spend_updates_to_redis = MagicMock(
        return_value=False
    )
    writer._commit_daily_tag_spend_to_db_with_redis = AsyncMock()
    writer._commit_daily_tag_spend_to_db = AsyncMock()

    await update_daily_tag_spend(
        prisma_client=mock_prisma_client, proxy_logging_obj=proxy_logging
    )
    assert writer._commit_daily_tag_spend_to_db.await_count == 1
    assert writer._commit_daily_tag_spend_to_db_with_redis.await_count == 0


@pytest.mark.asyncio
async def test_update_daily_tag_spend_logs_and_swallows_errors(
    mock_prisma_client: Any,
) -> None:
    """A failure in the commit path is logged but not re-raised; this matches
    the historical behavior of this site (see plain ``logger.error`` rather
    than ``spend_log_error``).
    """
    proxy_logging = MagicMock()
    proxy_logging.db_spend_update_writer = MagicMock()
    proxy_logging.db_spend_update_writer.redis_update_buffer = MagicMock()
    proxy_logging.db_spend_update_writer.redis_update_buffer._should_commit_spend_updates_to_redis = MagicMock(
        return_value=False
    )
    proxy_logging.db_spend_update_writer._commit_daily_tag_spend_to_db = AsyncMock(
        side_effect=RuntimeError("commit boom")
    )
    await update_daily_tag_spend(
        prisma_client=mock_prisma_client, proxy_logging_obj=proxy_logging
    )


@pytest.mark.asyncio
async def test_update_spend_logs_job_skips_when_queue_empty(
    mock_prisma_client: Any,
) -> None:
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    mock_prisma_client.spend_log_transactions = []
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock()
    await update_spend_logs_job(
        prisma_client=mock_prisma_client,
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
    )
    assert mock_prisma_client.db.litellm_spendlogs.create_many.await_count == 0


@pytest.mark.asyncio
async def test_update_spend_logs_job_processes_and_clears_queue(
    mock_prisma_client: Any, make_spend_log_row: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()
    mock_prisma_client.spend_log_transactions = [
        make_spend_log_row(request_id="r1"),
        make_spend_log_row(request_id="r2"),
    ]
    mock_prisma_client.db.litellm_spendlogs.create_many = AsyncMock()

    # Stub auxiliary imports so the test focuses on the spend-logs write path.
    import litellm.proxy.guardrails.usage_tracking as guard_mod
    import litellm.proxy.db.spend_log_tool_index as tool_mod

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
    pinned = {
        "create_many_calls": mock_prisma_client.db.litellm_spendlogs.create_many.await_count,
        "queue_after": mock_prisma_client.spend_log_transactions,
        "first_data_request_id": mock_prisma_client.db.litellm_spendlogs.create_many.await_args.kwargs[
            "data"
        ][0]["request_id"],
        "skip_duplicates_set": mock_prisma_client.db.litellm_spendlogs.create_many.await_args.kwargs[
            "skip_duplicates"
        ],
    }
    assert pinned == {
        "create_many_calls": 1,
        "queue_after": [],
        "first_data_request_id": "r1",
        "skip_duplicates_set": True,
    }


@pytest.mark.asyncio
async def test_monitor_spend_logs_queue_invokes_job_when_queue_nonempty(
    mock_prisma_client: Any,
    make_spend_log_row: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.utils as utils_mod
    import litellm.constants as constants_mod

    monkeypatch.setattr(constants_mod, "SPEND_LOG_QUEUE_POLL_INTERVAL", 0.0, raising=False)
    monkeypatch.setattr(constants_mod, "SPEND_LOG_QUEUE_SIZE_THRESHOLD", 1, raising=False)
    proxy_logging = MagicMock()
    mock_prisma_client.spend_log_transactions = [make_spend_log_row(request_id="r1")]

    cancel_after = {"n": 0}

    async def _fake_job(*args: Any, **kwargs: Any) -> None:
        cancel_after["n"] += 1
        if cancel_after["n"] >= 1:
            raise asyncio.CancelledError()

    monkeypatch.setattr(utils_mod, "update_spend_logs_job", _fake_job)

    with pytest.raises(asyncio.CancelledError):
        await _monitor_spend_logs_queue(
            prisma_client=mock_prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging,
        )
    assert cancel_after["n"] == 1


@pytest.mark.asyncio
async def test_monitor_spend_logs_queue_swallows_errors_and_backs_off(
    mock_prisma_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception inside the loop is logged with backoff and the loop
    continues running rather than crashing the monitor task.
    """
    import litellm.proxy.utils as utils_mod
    import litellm.constants as constants_mod

    monkeypatch.setattr(constants_mod, "SPEND_LOG_QUEUE_POLL_INTERVAL", 0.0, raising=False)

    sleep_count = {"n": 0}

    async def _short_sleep(_: float, *args: Any, **kwargs: Any) -> None:
        sleep_count["n"] += 1
        if sleep_count["n"] >= 3:
            raise asyncio.CancelledError()

    monkeypatch.setattr(utils_mod.asyncio, "sleep", _short_sleep)
    proxy_logging = MagicMock()

    bad_lock = MagicMock()
    bad_lock.__aenter__ = AsyncMock(side_effect=RuntimeError("lock broken"))
    bad_lock.__aexit__ = AsyncMock(return_value=False)
    mock_prisma_client._spend_log_transactions_lock = bad_lock

    with pytest.raises(asyncio.CancelledError):
        await _monitor_spend_logs_queue(
            prisma_client=mock_prisma_client,
            db_writer_client=None,
            proxy_logging_obj=proxy_logging,
        )
    assert sleep_count["n"] == 3


def test_raise_failed_update_spend_exception_emits_failure_handler() -> None:
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()

    async def _runner() -> Any:
        try:
            _raise_failed_update_spend_exception(
                e=RuntimeError("boom"),
                start_time=0.0,
                proxy_logging_obj=proxy_logging,
            )
        except RuntimeError as e:
            return e
        return None

    err = asyncio.run(_runner())
    pinned = {
        "raised": str(err),
        "failure_handler_called": proxy_logging.failure_handler.call_count,
        "call_type": (
            proxy_logging.failure_handler.call_args.kwargs.get("call_type")
            if proxy_logging.failure_handler.call_args
            else None
        ),
        "non_blocking_in_traceback": (
            "Non-Blocking"
            in proxy_logging.failure_handler.call_args.kwargs["traceback_str"]
            if proxy_logging.failure_handler.call_args
            else False
        ),
    }
    assert pinned == {
        "raised": "boom",
        "failure_handler_called": 1,
        "call_type": "update_spend",
        "non_blocking_in_traceback": True,
    }


def test_raise_failed_update_spend_exception_raises_original_error() -> None:
    """Error path: the function always re-raises the original exception so
    the caller can observe the failure.
    """
    proxy_logging = MagicMock()
    proxy_logging.failure_handler = AsyncMock()

    async def _runner() -> None:
        _raise_failed_update_spend_exception(
            e=ValueError("specific"),
            start_time=0.0,
            proxy_logging_obj=proxy_logging,
        )

    with pytest.raises(ValueError, match="specific"):
        asyncio.run(_runner())
