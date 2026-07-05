import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.litellm_core_utils.specialty_caches.dynamic_logging_cache import (
    LangfuseInMemoryCache,
)


class TestLangfuseInMemoryCache:
    """Simple tests to ensure langfuse client cleanup works correctly."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.cache = LangfuseInMemoryCache(max_size_in_memory=2, default_ttl=1)

    @patch("litellm.initialized_langfuse_clients", 5)
    def test_langfuse_client_count_decrements_on_eviction(self):
        """Test that langfuse client count decrements when elements get evicted from cache."""

        # Create a mock LangFuseLogger class
        class MockLangFuseLogger:
            def __init__(self):
                self.Langfuse = MagicMock()
                self.Langfuse.flush = MagicMock()
                self.Langfuse.shutdown = MagicMock()

        mock_logger = MockLangFuseLogger()

        # Patch the LangFuseLogger import to return our mock class
        with patch(
            "litellm.integrations.langfuse.langfuse.LangFuseLogger", MockLangFuseLogger
        ):
            # Add the mock logger to cache with expired TTL
            expired_time = time.time() - 1  # Already expired
            self.cache.cache_dict["test_key"] = mock_logger
            self.cache.ttl_dict["test_key"] = expired_time
            self.cache.expiration_heap = [(expired_time, "test_key")]

            initial_count = litellm.initialized_langfuse_clients

            # Trigger eviction
            self.cache.evict_cache()

            # Verify client count was decremented
            assert litellm.initialized_langfuse_clients == initial_count - 1

    @patch("litellm.initialized_langfuse_clients", 3)
    def test_langfuse_client_shutdown_called_on_eviction(self):
        """Test that langfuse client shutdown is called to close the thread."""

        # Create a mock LangFuseLogger class
        class MockLangFuseLogger:
            def __init__(self):
                self.Langfuse = MagicMock()
                self.Langfuse.flush = MagicMock()
                self.Langfuse.shutdown = MagicMock()

        mock_logger = MockLangFuseLogger()

        # Patch the LangFuseLogger import to return our mock class
        with patch(
            "litellm.integrations.langfuse.langfuse.LangFuseLogger", MockLangFuseLogger
        ):
            # Add the mock logger to cache
            self.cache.cache_dict["test_key"] = mock_logger
            self.cache.ttl_dict["test_key"] = time.time() + 100

            # Remove the key (this should trigger cleanup)
            self.cache._remove_key("test_key")

            # Verify flush and shutdown were called
            mock_logger.Langfuse.flush.assert_called_once()
            mock_logger.Langfuse.shutdown.assert_called_once()


class FakeHttpxClient:
    """A shared httpx client, the kind ``_get_httpx_client()`` hands out. An
    in-flight caller keeps a reference to it while a request is on the wire, so
    closing it out from under that caller is what raises ``RuntimeError: Cannot
    send a request, as the client has been closed``."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def send(self) -> str:
        if self.closed:
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return "ok"


class FakeLangfuseSDK:
    """A Langfuse SDK client whose ``shutdown()`` tears down the shared httpx
    client it was built on, mirroring the real SDK: shutdown joins the flush
    thread and closes the underlying httpx client (LIT-3221 / GH #13034)."""

    def __init__(self, httpx_client: FakeHttpxClient) -> None:
        self._httpx_client = httpx_client

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        self._httpx_client.close()


class FakeLangFuseLogger:
    """Stand-in for ``LangFuseLogger`` holding a Langfuse SDK client that shares
    the process-wide httpx client. Injected into the cache so eviction runs the
    real ``_remove_key`` cleanup path against controllable fakes."""

    def __init__(self, httpx_client: FakeHttpxClient) -> None:
        self.Langfuse = FakeLangfuseSDK(httpx_client)


class TestLangfuseEvictionKeepsInflightClientOpen:
    """LIT-3221 / GH #13034: evicting a cached Langfuse logger must NOT close the
    shared httpx client, because in-flight callers still hold it. This is the
    same invariant ``LLMClientCache`` already guarantees (it deliberately does
    not close on eviction); ``LangfuseInMemoryCache`` violates it by calling
    ``Langfuse.shutdown()`` on ``_remove_key``, so after ~1h of uptime evictions
    start closing clients other requests are mid-send on."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "LIT-3221 / GH #13034: LangfuseInMemoryCache eviction shuts down a "
            "client still in use by the logging executor"
        ),
    )
    @patch("litellm.initialized_langfuse_clients", 2)
    def test_eviction_does_not_close_inflight_shared_client(self) -> None:
        cache = LangfuseInMemoryCache(max_size_in_memory=2, default_ttl=1)

        shared_client = FakeHttpxClient()
        logger = FakeLangFuseLogger(shared_client)

        with patch(
            "litellm.integrations.langfuse.langfuse.LangFuseLogger", FakeLangFuseLogger
        ):
            expired = time.time() - 1
            cache.cache_dict["evicted"] = logger
            cache.ttl_dict["evicted"] = expired
            cache.expiration_heap = [(expired, "evicted")]

            cache.evict_cache()

            assert "evicted" not in cache.cache_dict, "logger was not evicted"
            assert shared_client.closed is False, (
                "eviction closed the shared httpx client an in-flight caller still holds"
            )
            assert shared_client.send() == "ok", (
                "in-flight caller can no longer send after the logger was evicted"
            )
