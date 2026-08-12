import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from prisma.errors import DataError, UniqueViolationError

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.db.log_db_metrics import (
    _handle_logging_db_exception,
    _record_db_pool_timeout_if_exhausted,
)


def _pool_timeout_error() -> DataError:
    return DataError(
        {
            "user_facing_error": {
                "error_code": "P2024",
                "meta": {"connection_limit": 10, "timeout": 60},
                "message": "Timed out fetching a new connection from the connection pool.",
            }
        }
    )


def test_pool_exhaustion_increments_the_timeout_counter():
    logger = MagicMock()
    with patch("litellm.proxy.db.log_db_metrics._prometheus_logger", return_value=logger):
        _record_db_pool_timeout_if_exhausted(_pool_timeout_error())
    logger.record_db_pool_timeout.assert_called_once()


def test_an_ordinary_db_error_does_not_increment_the_timeout_counter():
    logger = MagicMock()
    with patch("litellm.proxy.db.log_db_metrics._prometheus_logger", return_value=logger):
        _record_db_pool_timeout_if_exhausted(UniqueViolationError({"user_facing_error": {"error_code": "P2002"}}))
    logger.record_db_pool_timeout.assert_not_called()


def test_no_prometheus_logger_configured_is_not_an_error():
    with patch("litellm.proxy.db.log_db_metrics._prometheus_logger", return_value=None):
        _record_db_pool_timeout_if_exhausted(_pool_timeout_error())


@pytest.mark.asyncio
async def test_pool_exhaustion_is_counted_even_though_it_is_not_a_db_service_failure():
    """P2024 is a prisma ``DataError``, which ``_is_exception_related_to_db``
    classifies as DB-related, but the counter must not depend on that gate:
    exhaustion is the proxy running out of connections, and it has to be
    recorded on whichever side of the classification it lands."""
    logger = MagicMock()
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = MagicMock(
        side_effect=AssertionError("should not be reached when the error is not DB related")
    )

    with (
        patch("litellm.proxy.db.log_db_metrics._prometheus_logger", return_value=logger),
        patch("litellm.proxy.db.log_db_metrics._is_exception_related_to_db", return_value=False),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj),
    ):
        await _handle_logging_db_exception(
            e=_pool_timeout_error(),
            func=lambda: None,
            kwargs={},
            args=(),
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

    logger.record_db_pool_timeout.assert_called_once()


@pytest.mark.asyncio
async def test_a_stalled_pool_sample_does_not_delay_the_database_call():
    """The sample is dispatched, not awaited. Awaiting it would both add its
    latency to every sampled query and open a cancellation window after the
    query had already succeeded, in which a client disconnect would discard a
    completed result."""
    import asyncio

    from litellm.proxy.db.log_db_metrics import _pool_metrics_sampler, log_db_metrics

    started = asyncio.Event()

    async def _hang():
        started.set()
        await asyncio.sleep(30)

    @log_db_metrics
    async def fake_db_call(**kwargs):
        return "rows"

    _pool_metrics_sampler._last_sampled_at = None
    with patch("litellm.proxy.db.log_db_metrics._sample_db_pool_metrics", _hang):
        result = await asyncio.wait_for(fake_db_call(table_name="x"), timeout=1.0)

    assert result == "rows"
    await asyncio.wait_for(started.wait(), timeout=1.0), "the sample must still be dispatched"
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task() and task.get_coro().__name__ == "_hang":
            task.cancel()
