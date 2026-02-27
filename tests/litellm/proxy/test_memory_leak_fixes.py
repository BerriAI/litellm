"""
Tests for memory leak fixes in the LiteLLM proxy.

Covers:
1. Redis connection pool max_connections default
2. spend_log_transactions queue cap
3. Logging object cleanup of heavy references
4. PrismaClient engine cleanup helpers
5. Memory diagnostics endpoint helpers
"""

import asyncio
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRedisConnectionPoolMaxConnections:
    """Test that Redis connection pool has a sensible max_connections default."""

    def test_redis_default_max_connections_constant(self):
        from litellm.constants import REDIS_DEFAULT_MAX_CONNECTIONS

        assert REDIS_DEFAULT_MAX_CONNECTIONS == 100

    def test_redis_default_max_connections_env_override(self):
        with patch.dict(os.environ, {"REDIS_MAX_CONNECTIONS": "200"}):
            import importlib

            import litellm.constants
            importlib.reload(litellm.constants)
            assert litellm.constants.REDIS_DEFAULT_MAX_CONNECTIONS == 200
            # Restore
            importlib.reload(litellm.constants)

    def test_get_redis_connection_pool_url_has_max_connections(self):
        """When using URL-based pool, max_connections should be set."""
        from unittest.mock import patch as mock_patch

        with mock_patch("litellm._redis._get_redis_client_logic") as mock_logic:
            mock_logic.return_value = {"url": "redis://localhost:6379"}
            with mock_patch("redis.asyncio.BlockingConnectionPool.from_url") as mock_pool:
                mock_pool.return_value = MagicMock()
                from litellm._redis import get_redis_connection_pool

                get_redis_connection_pool()
                call_kwargs = mock_pool.call_args
                assert "max_connections" in call_kwargs.kwargs or any(
                    "max_connections" in str(a) for a in call_kwargs.args
                )

    def test_get_redis_connection_pool_kwargs_has_max_connections(self):
        """When using kwargs-based pool, max_connections should be set."""
        from unittest.mock import patch as mock_patch

        with mock_patch("litellm._redis._get_redis_client_logic") as mock_logic:
            mock_logic.return_value = {"host": "localhost", "port": 6379}
            with mock_patch("redis.asyncio.BlockingConnectionPool") as mock_pool:
                mock_pool.return_value = MagicMock()
                from litellm._redis import get_redis_connection_pool

                get_redis_connection_pool()
                call_kwargs = mock_pool.call_args
                assert "max_connections" in call_kwargs.kwargs

    def test_get_redis_connection_pool_user_override_preserved(self):
        """User-specified max_connections should override the default."""
        from unittest.mock import patch as mock_patch

        with mock_patch("litellm._redis._get_redis_client_logic") as mock_logic:
            mock_logic.return_value = {
                "host": "localhost",
                "port": 6379,
                "max_connections": 50,
            }
            with mock_patch("redis.asyncio.BlockingConnectionPool") as mock_pool:
                mock_pool.return_value = MagicMock()
                from litellm._redis import get_redis_connection_pool

                get_redis_connection_pool()
                call_kwargs = mock_pool.call_args
                assert call_kwargs.kwargs.get("max_connections") == 50


class TestSpendLogQueueCap:
    """Test that spend_log_transactions queue is bounded."""

    def test_max_spend_log_queue_size_constant(self):
        from litellm.constants import MAX_SPEND_LOG_QUEUE_SIZE

        assert MAX_SPEND_LOG_QUEUE_SIZE == 10000

    @pytest.mark.asyncio
    async def test_queue_cap_drops_oldest_when_full(self):
        """When queue is at capacity, oldest entry should be dropped."""
        from litellm.constants import MAX_SPEND_LOG_QUEUE_SIZE
        from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

        writer = DBSpendUpdateWriter()

        mock_prisma = MagicMock()
        mock_prisma._spend_log_transactions_lock = asyncio.Lock()
        mock_prisma.spend_log_transactions = [
            {"request_id": f"old-{i}"} for i in range(MAX_SPEND_LOG_QUEUE_SIZE)
        ]

        payload = {"request_id": "new-entry", "spend": 0.01}
        await writer._insert_spend_log_to_db(
            payload=payload,
            prisma_client=mock_prisma,
            spend_logs_url=None,
        )

        assert len(mock_prisma.spend_log_transactions) == MAX_SPEND_LOG_QUEUE_SIZE
        assert mock_prisma.spend_log_transactions[-1]["request_id"] == "new-entry"
        assert mock_prisma.spend_log_transactions[0]["request_id"] == "old-1"

    @pytest.mark.asyncio
    async def test_queue_appends_normally_when_not_full(self):
        """When queue is not full, entries are appended normally."""
        from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

        writer = DBSpendUpdateWriter()

        mock_prisma = MagicMock()
        mock_prisma._spend_log_transactions_lock = asyncio.Lock()
        mock_prisma.spend_log_transactions = []

        payload = {"request_id": "test-entry", "spend": 0.01}
        await writer._insert_spend_log_to_db(
            payload=payload,
            prisma_client=mock_prisma,
            spend_logs_url=None,
        )

        assert len(mock_prisma.spend_log_transactions) == 1
        assert mock_prisma.spend_log_transactions[0]["request_id"] == "test-entry"


class TestLoggingCleanupHeavyReferences:
    """Test that the Logging object cleans up heavy references after callbacks."""

    def _make_logging_obj(self):
        """Create a minimal Logging object for testing."""
        from litellm.litellm_core_utils.litellm_logging import Logging

        logging_obj = Logging(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            call_type="acompletion",
            start_time="2024-01-01T00:00:00",
            litellm_call_id="test-call-id",
            function_id="test-function-id",
        )
        return logging_obj

    def test_cleanup_removes_httpx_response(self):
        logging_obj = self._make_logging_obj()
        fake_response = MagicMock()
        fake_response.headers = OrderedDict([("content-type", "application/json")])
        logging_obj.model_call_details["httpx_response"] = fake_response

        logging_obj._cleanup_heavy_references()

        assert "httpx_response" not in logging_obj.model_call_details

    def test_cleanup_removes_response_headers(self):
        logging_obj = self._make_logging_obj()
        logging_obj.model_call_details["response_headers"] = OrderedDict(
            [("x-request-id", "abc123")]
        )

        logging_obj._cleanup_heavy_references()

        assert "response_headers" not in logging_obj.model_call_details

    def test_cleanup_removes_all_heavy_keys(self):
        logging_obj = self._make_logging_obj()
        logging_obj.model_call_details["httpx_response"] = MagicMock()
        logging_obj.model_call_details["response_headers"] = OrderedDict()
        logging_obj.model_call_details["raw_request_typed_dict"] = {"large": "data"}
        logging_obj.model_call_details["complete_streaming_response"] = MagicMock()

        logging_obj._cleanup_heavy_references()

        for key in ("httpx_response", "response_headers", "raw_request_typed_dict", "complete_streaming_response"):
            assert key not in logging_obj.model_call_details

    def test_cleanup_preserves_other_keys(self):
        logging_obj = self._make_logging_obj()
        logging_obj.model_call_details["model"] = "gpt-4"
        logging_obj.model_call_details["httpx_response"] = MagicMock()

        logging_obj._cleanup_heavy_references()

        assert logging_obj.model_call_details["model"] == "gpt-4"

    def test_cleanup_is_idempotent(self):
        logging_obj = self._make_logging_obj()
        logging_obj.model_call_details["httpx_response"] = MagicMock()

        logging_obj._cleanup_heavy_references()
        logging_obj._cleanup_heavy_references()  # should not raise

        assert "httpx_response" not in logging_obj.model_call_details


class TestSpendLogQueueDiagnostics:
    """Test the spend log queue diagnostics helper."""

    def test_diagnostics_with_no_prisma_client(self):
        from litellm.proxy.common_utils.debug_utils import _get_spend_log_queue_info

        result = _get_spend_log_queue_info(None)
        assert result == {"enabled": False}

    def test_diagnostics_with_empty_queue(self):
        from litellm.proxy.common_utils.debug_utils import _get_spend_log_queue_info

        mock_prisma = MagicMock()
        mock_prisma.spend_log_transactions = []
        result = _get_spend_log_queue_info(mock_prisma)

        assert result["queue_length"] == 0
        assert result["usage_percent"] == 0.0
        assert result["warning"] is None

    def test_diagnostics_with_near_full_queue(self):
        from litellm.proxy.common_utils.debug_utils import _get_spend_log_queue_info

        mock_prisma = MagicMock()
        mock_prisma.spend_log_transactions = [{}] * 9000  # 90% of 10K
        result = _get_spend_log_queue_info(mock_prisma)

        assert result["queue_length"] == 9000
        assert result["usage_percent"] == 90.0
        assert result["warning"] is not None


class TestPrismaClientEngineCleanup:
    """Test PrismaClient engine lifecycle helpers."""

    def test_get_engine_pid_returns_0_when_no_engine(self):
        """_get_engine_pid should return 0 when engine is not available."""
        from litellm.proxy.utils import PrismaClient

        mock_client = MagicMock(spec=PrismaClient)
        mock_client.db = MagicMock()
        mock_client.db._original_prisma = MagicMock()
        mock_client.db._original_prisma._engine = None

        result = PrismaClient._get_engine_pid(mock_client)
        assert result == 0

    def test_get_engine_pid_returns_pid_when_engine_exists(self):
        """_get_engine_pid should return the engine PID."""
        from litellm.proxy.utils import PrismaClient

        mock_client = MagicMock(spec=PrismaClient)
        mock_client.db = MagicMock()
        mock_client.db._original_prisma = MagicMock()
        mock_client.db._original_prisma._engine = MagicMock()
        mock_client.db._original_prisma._engine.process = MagicMock()
        mock_client.db._original_prisma._engine.process.pid = 12345

        result = PrismaClient._get_engine_pid(mock_client)
        assert result == 12345

    def test_cleanup_orphaned_engines_no_proc(self):
        """cleanup_orphaned_query_engines should handle missing /proc gracefully."""
        from litellm.proxy.utils import PrismaClient

        with patch("os.listdir", side_effect=FileNotFoundError):
            result = PrismaClient.cleanup_orphaned_query_engines()
        assert result == 0

    def test_atexit_kill_engine_no_engine(self):
        """_atexit_kill_engine should not raise when no engine exists."""
        from litellm.proxy.utils import PrismaClient

        mock_client = MagicMock(spec=PrismaClient)
        mock_client._get_engine_pid = MagicMock(return_value=0)

        PrismaClient._atexit_kill_engine(mock_client)

    def test_atexit_kill_engine_sends_sigterm(self):
        """_atexit_kill_engine should send SIGTERM to engine PID."""
        import signal

        from litellm.proxy.utils import PrismaClient

        mock_client = MagicMock(spec=PrismaClient)
        mock_client._get_engine_pid = MagicMock(return_value=99999)

        with patch("os.kill") as mock_kill:
            PrismaClient._atexit_kill_engine(mock_client)
            mock_kill.assert_called_once_with(99999, signal.SIGTERM)

    def test_atexit_kill_engine_handles_process_not_found(self):
        """_atexit_kill_engine should handle ProcessLookupError gracefully."""
        import signal

        from litellm.proxy.utils import PrismaClient

        mock_client = MagicMock(spec=PrismaClient)
        mock_client._get_engine_pid = MagicMock(return_value=99999)

        with patch("os.kill", side_effect=ProcessLookupError):
            PrismaClient._atexit_kill_engine(mock_client)  # should not raise
