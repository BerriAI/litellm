import json
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
