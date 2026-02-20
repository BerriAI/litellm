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
    client.disconnect = AsyncMock(return_value=None)
    client.connect = AsyncMock(return_value=None)
    client.health_check = AsyncMock(return_value=[{"result": 1}])

    result = await client.attempt_db_reconnect(
        reason="unit_test_reconnect_success",
        force=True,
    )

    assert result is True
    client.disconnect.assert_awaited_once()
    client.connect.assert_awaited_once()
    client.health_check.assert_awaited_once()


@pytest.mark.asyncio
async def test_attempt_db_reconnect_should_skip_when_in_cooldown(mock_proxy_logging):
    client = PrismaClient(database_url="mock://test", proxy_logging_obj=mock_proxy_logging)
    client.disconnect = AsyncMock(return_value=None)
    client.connect = AsyncMock(return_value=None)
    client.health_check = AsyncMock(return_value=[{"result": 1}])
    client._db_reconnect_cooldown_seconds = 120
    client._db_last_reconnect_attempt_ts = time.time()

    result = await client.attempt_db_reconnect(
        reason="unit_test_reconnect_cooldown",
        force=False,
    )

    assert result is False
    client.disconnect.assert_not_called()
    client.connect.assert_not_called()
    client.health_check.assert_not_called()


@pytest.mark.asyncio
async def test_db_health_watchdog_should_trigger_reconnect_on_db_error(mock_proxy_logging):
    client = PrismaClient(database_url="mock://test", proxy_logging_obj=mock_proxy_logging)
    client.health_check = AsyncMock(side_effect=Exception("db connection dropped"))
    client.attempt_db_reconnect = AsyncMock(return_value=True)
    client._db_health_watchdog_interval_seconds = 1

    with patch(
        "litellm.proxy.utils.asyncio.sleep",
        AsyncMock(side_effect=[None, asyncio.CancelledError()]),
    ), patch(
        "litellm.proxy.db.exception_handler.PrismaDBExceptionHandler.is_database_connection_error",
        return_value=True,
    ):
        await client._db_health_watchdog_loop()

    client.attempt_db_reconnect.assert_awaited_once_with(
        reason="db_health_watchdog_connection_error"
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
