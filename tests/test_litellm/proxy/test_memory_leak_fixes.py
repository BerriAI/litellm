"""
Tests for memory leak fixes in the LiteLLM proxy.

Covers:
- Spend log transactions queue capping (prevents unbounded growth)
- Redis connection pool max_connections default (prevents unbounded pool)
- Engine atexit cleanup registration (prevents orphaned query-engines)
- Logging pipeline cleanup after callbacks (prevents response retention)
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fix 1: Spend log transactions queue capping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spend_log_queue_caps_at_max_size():
    """
    When spend_log_transactions reaches MAX_SPEND_LOG_TRANSACTIONS_QUEUE_SIZE,
    old entries should be dropped to prevent unbounded memory growth.
    """
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

    writer = DBSpendUpdateWriter()

    # Create a mock PrismaClient with a real list and lock
    mock_prisma = MagicMock()
    mock_prisma.spend_log_transactions = []
    mock_prisma._spend_log_transactions_lock = asyncio.Lock()

    max_queue_size = 100  # Use a small value for testing

    with patch(
        "litellm.constants.MAX_SPEND_LOG_TRANSACTIONS_QUEUE_SIZE",
        max_queue_size,
    ):
        # Fill the queue to max
        for i in range(max_queue_size):
            mock_prisma.spend_log_transactions.append(
                {"request_id": f"req_{i}", "spend": 0.01}
            )

        assert len(mock_prisma.spend_log_transactions) == max_queue_size

        # Adding one more should trigger dropping oldest entries
        await writer._insert_spend_log_to_db(
            payload={"request_id": "req_new", "spend": 0.02},
            prisma_client=mock_prisma,
        )

        # Queue should not exceed max_queue_size + 1 (the new item)
        # After dropping 10% oldest + adding 1 new
        drop_count = max(1, max_queue_size // 10)
        expected_size = max_queue_size - drop_count + 1
        assert len(mock_prisma.spend_log_transactions) == expected_size

        # The newest entry should be the last one
        assert mock_prisma.spend_log_transactions[-1]["request_id"] == "req_new"

        # The oldest entries should have been dropped
        assert mock_prisma.spend_log_transactions[0]["request_id"] == f"req_{drop_count}"


@pytest.mark.asyncio
async def test_spend_log_queue_normal_append_when_below_max():
    """
    When queue is below MAX_SPEND_LOG_TRANSACTIONS_QUEUE_SIZE,
    entries should be appended normally without dropping.
    """
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

    writer = DBSpendUpdateWriter()

    mock_prisma = MagicMock()
    mock_prisma.spend_log_transactions = []
    mock_prisma._spend_log_transactions_lock = asyncio.Lock()

    # Default MAX_SPEND_LOG_TRANSACTIONS_QUEUE_SIZE is 10000, so 1 item is well below
    await writer._insert_spend_log_to_db(
        payload={"request_id": "req_1", "spend": 0.01},
        prisma_client=mock_prisma,
    )

    assert len(mock_prisma.spend_log_transactions) == 1
    assert mock_prisma.spend_log_transactions[0]["request_id"] == "req_1"


@pytest.mark.asyncio
async def test_spend_log_queue_no_prisma_client():
    """When prisma_client is None, no error should occur."""
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

    writer = DBSpendUpdateWriter()
    result = await writer._insert_spend_log_to_db(
        payload={"request_id": "req_1", "spend": 0.01},
        prisma_client=None,
    )
    assert result is None


# ---------------------------------------------------------------------------
# Fix 2: Redis connection pool max_connections default
# ---------------------------------------------------------------------------


def test_redis_connection_pool_has_bounded_max_connections():
    """
    get_redis_connection_pool should always set max_connections to prevent
    unbounded pool growth (redis-py defaults to 2**31).
    """
    from litellm.constants import DEFAULT_REDIS_MAX_CONNECTIONS

    with patch("litellm._redis._get_redis_client_logic") as mock_logic:
        # Simulate a non-URL Redis config (host/port style)
        mock_logic.return_value = {
            "host": "localhost",
            "port": 6379,
        }

        with patch("litellm._redis.async_redis.BlockingConnectionPool") as mock_pool:
            from litellm._redis import get_redis_connection_pool

            get_redis_connection_pool()

            # Verify max_connections was passed
            call_kwargs = mock_pool.call_args
            assert "max_connections" in call_kwargs.kwargs or any(
                "max_connections" in str(arg) for arg in call_kwargs.args
            ), "max_connections must be set on BlockingConnectionPool"


def test_redis_connection_pool_url_has_bounded_max_connections():
    """
    get_redis_connection_pool with URL should also set max_connections.
    """
    from litellm.constants import DEFAULT_REDIS_MAX_CONNECTIONS

    with patch("litellm._redis._get_redis_client_logic") as mock_logic:
        mock_logic.return_value = {
            "url": "redis://localhost:6379",
        }

        with patch(
            "litellm._redis.async_redis.BlockingConnectionPool.from_url"
        ) as mock_pool:
            from litellm._redis import get_redis_connection_pool

            get_redis_connection_pool()

            call_kwargs = mock_pool.call_args.kwargs
            assert "max_connections" in call_kwargs
            assert call_kwargs["max_connections"] == DEFAULT_REDIS_MAX_CONNECTIONS


def test_redis_connection_pool_user_override_respected():
    """
    When the user explicitly sets max_connections, their value should be used.
    """
    with patch("litellm._redis._get_redis_client_logic") as mock_logic:
        mock_logic.return_value = {
            "url": "redis://localhost:6379",
            "max_connections": "50",
        }

        with patch(
            "litellm._redis.async_redis.BlockingConnectionPool.from_url"
        ) as mock_pool:
            from litellm._redis import get_redis_connection_pool

            get_redis_connection_pool()

            call_kwargs = mock_pool.call_args.kwargs
            assert call_kwargs["max_connections"] == 50  # User's value, not default


def test_redis_async_client_from_url_has_bounded_max_connections():
    """
    get_redis_async_client with URL (no connection pool) should set
    max_connections on the internal pool created by Redis.from_url().
    """
    from litellm.constants import DEFAULT_REDIS_MAX_CONNECTIONS

    with patch("litellm._redis._get_redis_client_logic") as mock_logic:
        mock_logic.return_value = {
            "url": "redis://localhost:6379",
        }

        with patch("litellm._redis._get_redis_url_kwargs") as mock_url_kwargs:
            mock_url_kwargs.return_value = ["url", "max_connections"]

            with patch("litellm._redis.async_redis.Redis.from_url") as mock_from_url:
                from litellm._redis import get_redis_async_client

                get_redis_async_client(connection_pool=None)

                call_kwargs = mock_from_url.call_args.kwargs
                assert "max_connections" in call_kwargs
                assert call_kwargs["max_connections"] == DEFAULT_REDIS_MAX_CONNECTIONS


# ---------------------------------------------------------------------------
# Fix 3: Engine atexit cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_prisma_binary():
    """Mock prisma.Prisma to avoid requiring generated Prisma binaries for unit tests."""
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"prisma": mock_module}):
        yield


@pytest.fixture
def mock_proxy_logging():
    from litellm.proxy.utils import ProxyLogging

    proxy_logging = AsyncMock(spec=ProxyLogging)
    proxy_logging.failure_handler = AsyncMock()
    return proxy_logging


@pytest.fixture
def engine_client(mock_proxy_logging):
    from litellm.proxy.utils import PrismaClient

    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client.db = MagicMock()
    client.db.recreate_prisma_client = AsyncMock()
    return client


def test_register_engine_cleanup_registers_atexit(engine_client):
    """_register_engine_cleanup should register an atexit handler."""
    with patch("atexit.register") as mock_register:
        engine_client._register_engine_cleanup(pid=12345)
        mock_register.assert_called_once()


def test_engine_cleanup_kills_engine_on_exit(engine_client):
    """The registered atexit handler should send SIGTERM to the engine PID."""
    import signal

    registered_func = None

    def capture_register(func):
        nonlocal registered_func
        registered_func = func

    with patch("atexit.register", side_effect=capture_register):
        engine_client._register_engine_cleanup(pid=12345)

    assert registered_func is not None

    # Simulate calling the atexit handler
    with patch("os.kill") as mock_kill:
        registered_func()
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)


def test_engine_cleanup_handles_already_dead_process(engine_client):
    """The atexit handler should handle ProcessLookupError gracefully."""
    registered_func = None

    def capture_register(func):
        nonlocal registered_func
        registered_func = func

    with patch("atexit.register", side_effect=capture_register):
        engine_client._register_engine_cleanup(pid=99999)

    with patch("os.kill", side_effect=ProcessLookupError):
        # Should not raise
        registered_func()


def test_reap_all_zombies_targets_engine_first(engine_client):
    """_reap_all_zombies should try to reap the engine PID first."""
    engine_client._engine_pid = 12345

    with patch("os.waitpid") as mock_waitpid:
        # First call: reap engine PID, second call: no more children
        mock_waitpid.side_effect = [
            (12345, 0),  # Engine PID reaped
            ChildProcessError(),  # No more children
        ]
        reaped = engine_client._reap_all_zombies()
        assert 12345 in reaped

        # Verify first call was for specific engine PID
        first_call = mock_waitpid.call_args_list[0]
        assert first_call.args[0] == 12345


# ---------------------------------------------------------------------------
# Fix 4: Logging cleanup after callbacks
# ---------------------------------------------------------------------------


def test_cleanup_after_logging_clears_streaming_chunks():
    """_cleanup_after_logging should clear streaming chunks lists."""
    from litellm.litellm_core_utils.litellm_logging import Logging

    logging_obj = MagicMock(spec=Logging)
    logging_obj.streaming_chunks = [{"chunk": 1}, {"chunk": 2}]
    logging_obj.sync_streaming_chunks = [{"chunk": 3}]
    logging_obj.model_call_details = {
        "model": "gpt-4",
        "complete_streaming_response": {"choices": [{"message": {"content": "hi"}}]},
        "httpx_response": MagicMock(),
    }

    # Call the real method
    Logging._cleanup_after_logging(logging_obj)

    assert len(logging_obj.streaming_chunks) == 0
    assert len(logging_obj.sync_streaming_chunks) == 0
    assert "complete_streaming_response" not in logging_obj.model_call_details
    assert "httpx_response" not in logging_obj.model_call_details
    # Non-response keys should be preserved
    assert "model" in logging_obj.model_call_details


def test_cleanup_after_logging_handles_missing_keys():
    """_cleanup_after_logging should handle model_call_details without response keys."""
    from litellm.litellm_core_utils.litellm_logging import Logging

    logging_obj = MagicMock(spec=Logging)
    logging_obj.streaming_chunks = []
    logging_obj.sync_streaming_chunks = []
    logging_obj.model_call_details = {"model": "gpt-4"}

    # Should not raise
    Logging._cleanup_after_logging(logging_obj)
    assert "model" in logging_obj.model_call_details
