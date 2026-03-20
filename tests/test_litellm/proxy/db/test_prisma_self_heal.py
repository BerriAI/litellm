import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.utils import PrismaClient, ProxyLogging


@pytest.fixture(autouse=True)
def mock_prisma_binary():
    """Mock prisma.Prisma to avoid requiring generated Prisma binaries for unit tests."""
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"prisma": mock_module}):
        yield


@pytest.fixture
def mock_proxy_logging():
    proxy_logging = AsyncMock(spec=ProxyLogging)
    proxy_logging.failure_handler = AsyncMock()
    return proxy_logging


@pytest.mark.asyncio
async def test_attempt_db_reconnect_should_succeed(mock_proxy_logging):
    client = PrismaClient(database_url="mock://test", proxy_logging_obj=mock_proxy_logging)
    client.db.disconnect = AsyncMock(return_value=None)
    client.db.connect = AsyncMock(return_value=None)
    client.db.query_raw = AsyncMock(return_value=[{"result": 1}])

    result = await client.attempt_db_reconnect(
        reason="unit_test_reconnect_success",
        force=True,
    )

    assert result is True
    client.db.disconnect.assert_awaited_once()
    client.db.connect.assert_awaited_once()
    client.db.query_raw.assert_awaited_once_with("SELECT 1")


@pytest.mark.asyncio
async def test_attempt_db_reconnect_should_skip_when_in_cooldown(mock_proxy_logging):
    client = PrismaClient(database_url="mock://test", proxy_logging_obj=mock_proxy_logging)
    client.db.disconnect = AsyncMock(return_value=None)
    client.db.connect = AsyncMock(return_value=None)
    client.db.query_raw = AsyncMock(return_value=[{"result": 1}])
    client._db_reconnect_cooldown_seconds = 120
    client._db_last_reconnect_attempt_ts = time.time()

    result = await client.attempt_db_reconnect(
        reason="unit_test_reconnect_cooldown",
        force=False,
    )

    assert result is False
    client.db.disconnect.assert_not_called()
    client.db.connect.assert_not_called()
    client.db.query_raw.assert_not_called()


@pytest.mark.asyncio
async def test_attempt_db_reconnect_should_skip_when_lock_timeout_expires(
    mock_proxy_logging,
):
    client = PrismaClient(database_url="mock://test", proxy_logging_obj=mock_proxy_logging)
    client.db.disconnect = AsyncMock(return_value=None)
    client.db.connect = AsyncMock(return_value=None)
    client.db.query_raw = AsyncMock(return_value=[{"result": 1}])

    await client._db_reconnect_lock.acquire()
    try:
        result = await client.attempt_db_reconnect(
            reason="unit_test_reconnect_lock_timeout",
            force=True,
            timeout_seconds=0.1,
            lock_timeout_seconds=0.01,
        )
    finally:
        client._db_reconnect_lock.release()

    assert result is False
    client.db.disconnect.assert_not_called()
    client.db.connect.assert_not_called()
    client.db.query_raw.assert_not_called()


@pytest.mark.asyncio
async def test_attempt_db_reconnect_should_not_leak_lock_on_timeout_race(
    mock_proxy_logging,
):
    client = PrismaClient(database_url="mock://test", proxy_logging_obj=mock_proxy_logging)
    client.db.disconnect = AsyncMock(return_value=None)
    client.db.connect = AsyncMock(return_value=None)
    client.db.query_raw = AsyncMock(return_value=[{"result": 1}])

    async def _fake_wait(tasks, timeout=None, return_when=None):
        # Let the acquire task run first, then emulate a timeout response
        # from asyncio.wait to exercise timeout-race cleanup.
        await asyncio.sleep(0)
        return set(), set(tasks)

    with patch("litellm.proxy.utils.asyncio.wait", side_effect=_fake_wait):
        result = await client.attempt_db_reconnect(
            reason="unit_test_reconnect_lock_timeout_race",
            force=True,
            timeout_seconds=0.1,
            lock_timeout_seconds=0.01,
        )

    assert result is False
    assert client._db_reconnect_lock.locked() is False
    client.db.disconnect.assert_not_called()
    client.db.connect.assert_not_called()
    client.db.query_raw.assert_not_called()


@pytest.mark.asyncio
async def test_attempt_db_reconnect_should_set_cooldown_after_attempt(mock_proxy_logging):
    client = PrismaClient(database_url="mock://test", proxy_logging_obj=mock_proxy_logging)
    client._db_last_reconnect_attempt_ts = 0.0
    client._db_reconnect_cooldown_seconds = 10
    client.db.disconnect = AsyncMock(return_value=None)
    client.db.connect = AsyncMock(return_value=None)
    client.db.query_raw = AsyncMock(return_value=[{"result": 1}])

    # Use a counter-based mock to avoid StopIteration when time.time() is called
    # more times than expected (varies by Python version / internal code paths).
    fake_clock = iter(range(100, 10000))
    with patch(
        "litellm.proxy.utils.time.time", side_effect=lambda: float(next(fake_clock))
    ):
        result = await client.attempt_db_reconnect(
            reason="unit_test_cooldown_timestamp_after_attempt",
            timeout_seconds=0.1,
        )

    assert result is True
    # The last time.time() call sets _db_last_reconnect_attempt_ts in the finally block.
    # Just verify it was updated to a value greater than the initial 0.0.
    assert client._db_last_reconnect_attempt_ts > 0.0


@pytest.mark.asyncio
async def test_run_reconnect_cycle_watchdog_should_use_direct_db_ops(mock_proxy_logging):
    client = PrismaClient(database_url="mock://test", proxy_logging_obj=mock_proxy_logging)
    client.disconnect = AsyncMock(side_effect=AssertionError("wrapper disconnect used"))
    client.connect = AsyncMock(side_effect=AssertionError("wrapper connect used"))
    client.db.disconnect = AsyncMock(return_value=None)
    client.db.connect = AsyncMock(return_value=None)
    client.db.query_raw = AsyncMock(return_value=[{"result": 1}])

    await client._run_reconnect_cycle(timeout_seconds=None)

    client.db.disconnect.assert_awaited_once()
    client.db.connect.assert_awaited_once()
    client.db.query_raw.assert_awaited_once_with("SELECT 1")


@pytest.mark.asyncio
async def test_run_reconnect_cycle_watchdog_should_use_default_timeout_budget(
    mock_proxy_logging,
):
    client = PrismaClient(database_url="mock://test", proxy_logging_obj=mock_proxy_logging)
    client._db_watchdog_reconnect_timeout_seconds = 0.1
    client.db.disconnect = AsyncMock(return_value=None)

    async def _slow_connect():
        await asyncio.sleep(0.08)

    async def _slow_query(_query: str):
        await asyncio.sleep(0.08)
        return [{"result": 1}]

    client.db.connect = AsyncMock(side_effect=_slow_connect)
    client.db.query_raw = AsyncMock(side_effect=_slow_query)

    with pytest.raises(asyncio.TimeoutError):
        await client._run_reconnect_cycle(timeout_seconds=None)


@pytest.mark.asyncio
async def test_run_reconnect_cycle_timeout_should_use_single_overall_budget(
    mock_proxy_logging,
):
    client = PrismaClient(database_url="mock://test", proxy_logging_obj=mock_proxy_logging)
    client.db.disconnect = AsyncMock(return_value=None)

    async def _slow_connect():
        await asyncio.sleep(0.08)

    async def _slow_query(_query: str):
        await asyncio.sleep(0.08)
        return [{"result": 1}]

    client.db.connect = AsyncMock(side_effect=_slow_connect)
    client.db.query_raw = AsyncMock(side_effect=_slow_query)

    with pytest.raises(asyncio.TimeoutError):
        await client._run_reconnect_cycle(timeout_seconds=0.1)


@pytest.mark.asyncio
async def test_db_health_watchdog_should_trigger_reconnect_on_db_error(mock_proxy_logging):
    client = PrismaClient(database_url="mock://test", proxy_logging_obj=mock_proxy_logging)
    client.db.query_raw = AsyncMock(side_effect=Exception("db connection dropped"))
    client.attempt_db_reconnect = AsyncMock(return_value=True)
    client._db_health_watchdog_interval_seconds = 1
    client._db_watchdog_reconnect_timeout_seconds = 7.0
    client._db_health_watchdog_probe_timeout_seconds = 0.2

    with patch(
        "litellm.proxy.utils.asyncio.sleep",
        AsyncMock(side_effect=[None, asyncio.CancelledError()]),
    ), patch(
        "litellm.proxy.db.exception_handler.PrismaDBExceptionHandler.is_database_connection_error",
        return_value=True,
    ):
        await client._db_health_watchdog_loop()

    client.attempt_db_reconnect.assert_awaited_once_with(
        reason="db_health_watchdog_connection_error",
        timeout_seconds=7.0,
    )


@pytest.mark.asyncio
async def test_db_health_watchdog_should_trigger_reconnect_on_probe_timeout(
    mock_proxy_logging,
):
    client = PrismaClient(database_url="mock://test", proxy_logging_obj=mock_proxy_logging)
    client.db.query_raw = AsyncMock(side_effect=asyncio.TimeoutError())
    client.attempt_db_reconnect = AsyncMock(return_value=True)
    client._db_health_watchdog_interval_seconds = 1
    client._db_watchdog_reconnect_timeout_seconds = 9.0
    client._db_health_watchdog_probe_timeout_seconds = 0.2

    with patch(
        "litellm.proxy.utils.asyncio.sleep",
        AsyncMock(side_effect=[None, asyncio.CancelledError()]),
    ), patch(
        "litellm.proxy.db.exception_handler.PrismaDBExceptionHandler.is_database_connection_error",
        return_value=False,
    ):
        await client._db_health_watchdog_loop()

    client.attempt_db_reconnect.assert_awaited_once_with(
        reason="db_health_watchdog_connection_error",
        timeout_seconds=9.0,
    )


@pytest.mark.asyncio
async def test_db_health_watchdog_start_stop_lifecycle(mock_proxy_logging):
    client = PrismaClient(database_url="mock://test", proxy_logging_obj=mock_proxy_logging)
    client._db_health_watchdog_enabled = True
    client._db_health_watchdog_interval_seconds = 3600

    loop = asyncio.get_running_loop()
    dummy_task = loop.create_task(asyncio.sleep(3600))

    def _fake_create_task(coro):
        # create_task is patched in this test, so explicitly close the incoming coroutine
        # to avoid "coroutine was never awaited" warnings.
        coro.close()
        return dummy_task

    with patch("litellm.proxy.utils.asyncio.create_task", side_effect=_fake_create_task):
        await client.start_db_health_watchdog_task()
        assert client._db_health_watchdog_task is dummy_task

        await client.stop_db_health_watchdog_task()
        assert client._db_health_watchdog_task is None
        assert dummy_task.cancelled() is True
