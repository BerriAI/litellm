import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

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


@pytest.mark.asyncio
async def test_one_pool_timeout_is_counted_once_across_nested_decorated_calls():
    """get_object_permission carries @log_db_metrics and is called from inside
    get_key_object, which also carries it, so a single P2024 passes through
    several except blocks on its way up. Counting per block would inflate the
    exhaustion metric by the nesting depth."""
    from litellm.proxy.db.log_db_metrics import log_db_metrics

    logger = MagicMock()
    error = _pool_timeout_error()

    @log_db_metrics
    async def inner(**kwargs):
        raise error

    @log_db_metrics
    async def outer(**kwargs):
        return await inner(**kwargs)

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    with (
        patch("litellm.proxy.db.log_db_metrics._prometheus_logger", return_value=logger),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj),
    ):
        with pytest.raises(DataError):
            await outer(table_name="x")

    assert logger.record_db_pool_timeout.call_count == 1


@pytest.mark.asyncio
async def test_two_separate_pool_timeouts_are_counted_separately():
    """The dedup must key on the exception instance, not suppress the metric."""
    from litellm.proxy.db.log_db_metrics import log_db_metrics

    logger = MagicMock()

    @log_db_metrics
    async def failing(**kwargs):
        raise _pool_timeout_error()

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()

    with (
        patch("litellm.proxy.db.log_db_metrics._prometheus_logger", return_value=logger),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj),
    ):
        for _ in range(2):
            with pytest.raises(DataError):
                await failing(table_name="x")

    assert logger.record_db_pool_timeout.call_count == 2


@pytest.mark.asyncio
async def test_a_proxy_without_prometheus_never_reads_the_query_engine():
    """The sample is worthless with no collector configured, so the engine read
    must be skipped rather than taken and discarded every interval."""
    from litellm.proxy.db.log_db_metrics import _pool_metrics_sampler, _sample_db_pool_metrics

    reads = []
    prisma_client = MagicMock()

    async def _sample():
        reads.append(1)
        raise AssertionError("the engine must not be read when prometheus is off")

    prisma_client.db.get_pool_sample = _sample
    _pool_metrics_sampler._last_sampled_at = None

    with (
        patch("litellm.proxy.db.log_db_metrics._prometheus_logger", return_value=None),
        patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
    ):
        await _sample_db_pool_metrics()

    assert reads == []


@pytest.mark.asyncio
async def test_the_decorated_path_actually_samples_end_to_end():
    """The decorator claims the interval and the dispatched task takes the
    sample. If both claimed, the task's claim would fail and no sample would
    ever be taken, while every throttle test kept passing."""
    import asyncio

    from litellm.proxy.db.db_pool_metrics import DBPoolSample
    from litellm.proxy.db.log_db_metrics import _pool_metrics_sampler, log_db_metrics

    sample = DBPoolSample(
        busy_connections=2.0,
        idle_connections=1.0,
        open_connections=3.0,
        pending_acquirers=0.0,
        acquire_wait_seconds_total=0.0,
        acquire_count_total=0,
        query_duration_seconds_total=0.0,
        query_count_total=0,
    )
    reads = []

    async def get_pool_sample():
        reads.append(1)
        return sample

    prisma_client = MagicMock()
    prisma_client.db.get_pool_sample = get_pool_sample
    logger = MagicMock()
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()

    @log_db_metrics
    async def fake_db_call(**kwargs):
        return "rows"

    _pool_metrics_sampler._last_sampled_at = None
    with (
        patch("litellm.proxy.db.log_db_metrics._prometheus_logger", return_value=logger),
        patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj),
    ):
        assert await fake_db_call(table_name="x") == "rows"
        for _ in range(50):
            await asyncio.sleep(0.01)
            if reads:
                break

    assert reads == [1], f"the dispatched task must take exactly one sample, got {len(reads)}"
    logger.record_db_pool_sample.assert_called_once()


@pytest.mark.asyncio
async def test_a_burst_of_database_calls_claims_only_one_sample():
    """The decorator gates task creation on the claim, so a burst of concurrent
    database calls must produce one sample rather than one per caller.

    Coroutines only interleave at await points, so this holds as long as the
    claim stays synchronous. Introducing an await between the due check and the
    timestamp write is what would break it.
    """
    import asyncio

    from litellm.proxy.db.db_pool_metrics import DBPoolMetricsSampler

    sampler = DBPoolMetricsSampler(min_interval_seconds=10.0)

    async def caller() -> bool:
        await asyncio.sleep(0)  # force a real scheduling point before claiming
        return sampler.try_claim()

    claims = await asyncio.gather(*[caller() for _ in range(50)])

    assert sum(claims) == 1, f"exactly one caller may claim the interval, got {sum(claims)}"
