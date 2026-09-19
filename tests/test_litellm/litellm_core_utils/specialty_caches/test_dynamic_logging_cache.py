import json
import time
from unittest.mock import MagicMock, patch

import pytest


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
            pass

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
    def test_evicted_logger_releases_its_hold_on_the_shared_export_channel(self):
        """Export channels are shared per credential set: eviction gives this logger's hold back
        while a sibling logger keeps exporting, and the channel is retired once the last hold goes."""
        from litellm.integrations.langfuse.langfuse import LangFuseLogger
        from litellm.integrations.langfuse.langfuse_sdk import acquire_langfuse_tracing, release_langfuse_tracing

        def acquire():
            return acquire_langfuse_tracing(
                public_key="pk-eviction-test",
                secret_key="sk",
                base_url="http://127.0.0.1:1",
                environment=None,
                release=None,
                flush_interval=1.0,
                mock_mode=True,
            )

        logger = LangFuseLogger.__new__(LangFuseLogger)
        logger.api_client = MagicMock()
        logger.api_client.get_prompt.return_value = "prompt-after-eviction"
        logger.tracing = acquire()
        sibling = acquire()
        self.cache.cache_dict["test_key"] = logger
        self.cache.ttl_dict["test_key"] = time.time() + 100

        self.cache._remove_key("test_key")

        assert litellm.initialized_langfuse_clients == 2
        assert logger.api_client.get_prompt("greeting") == "prompt-after-eviction"
        with sibling.tracer.start_as_current_span("still-open"):
            pass
        assert sibling.flush(1000) is True

        release_langfuse_tracing(sibling, grace_seconds=0.0)
        assert acquire() is not logger.tracing, "eviction did not release the evicted logger's hold"
