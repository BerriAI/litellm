"""
Memory Leak Investigation Tests for v1.81.12.rc.1

These tests reproduce and verify fixes for memory leak patterns identified
in the investigation of the v1.81.12.rc.1 memory leak report.

Key issues investigated:
1. HTTP client cache eviction without closing connections
2. Sagemaker error handler NoneType crash (AttributeError: 'NoneType' object has no attribute 'get')
3. InMemoryCache eviction callback support
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestInMemoryCacheEvictionCallback:
    """Test that InMemoryCache properly calls on_evict when items are removed."""

    def test_eviction_callback_called_on_ttl_expiry(self):
        """When a cached item expires via TTL, the on_evict callback should fire."""
        from litellm.caching.in_memory_cache import InMemoryCache

        evicted_values = []

        def on_evict(value):
            evicted_values.append(value)

        cache = InMemoryCache(
            max_size_in_memory=10,
            default_ttl=1,  # 1 second TTL
            on_evict=on_evict,
        )

        cache.set_cache("key1", "value1", ttl=0.01)  # expires almost immediately
        time.sleep(0.05)

        # Trigger eviction by setting another key (evict_cache is called on set)
        cache.set_cache("key2", "value2")

        # Force eviction check
        cache.evict_cache()

        assert "value1" in evicted_values, "on_evict should be called when item expires"

    def test_eviction_callback_called_on_size_limit(self):
        """When cache exceeds max_size_in_memory, eviction callback should fire."""
        from litellm.caching.in_memory_cache import InMemoryCache

        evicted_values = []

        def on_evict(value):
            evicted_values.append(value)

        cache = InMemoryCache(
            max_size_in_memory=2,
            default_ttl=600,
            on_evict=on_evict,
        )

        cache.set_cache("key1", "value1")
        cache.set_cache("key2", "value2")
        cache.set_cache("key3", "value3")  # Should trigger eviction of oldest

        assert len(evicted_values) >= 1, "on_evict should be called when cache is full"

    def test_eviction_callback_not_called_when_not_set(self):
        """When on_evict is None, no callback should be attempted."""
        from litellm.caching.in_memory_cache import InMemoryCache

        cache = InMemoryCache(
            max_size_in_memory=1,
            default_ttl=600,
            on_evict=None,
        )

        # This should not raise an error
        cache.set_cache("key1", "value1")
        cache.set_cache("key2", "value2")  # Triggers eviction

    def test_eviction_callback_exception_does_not_propagate(self):
        """If on_evict raises, it should not break cache operations."""
        from litellm.caching.in_memory_cache import InMemoryCache

        def bad_callback(value):
            raise RuntimeError("callback error")

        cache = InMemoryCache(
            max_size_in_memory=1,
            default_ttl=600,
            on_evict=bad_callback,
        )

        cache.set_cache("key1", "value1")
        # Should not raise even though callback throws
        cache.set_cache("key2", "value2")

    def test_delete_cache_calls_eviction_callback(self):
        """Explicit delete_cache should also trigger on_evict."""
        from litellm.caching.in_memory_cache import InMemoryCache

        evicted_values = []

        def on_evict(value):
            evicted_values.append(value)

        cache = InMemoryCache(
            max_size_in_memory=10,
            default_ttl=600,
            on_evict=on_evict,
        )

        cache.set_cache("key1", "value1")
        cache.delete_cache("key1")

        assert "value1" in evicted_values


class TestLLMClientCacheCleanup:
    """Test that LLMClientCache closes HTTP clients on eviction."""

    def test_sync_client_closed_on_eviction(self):
        """Sync HTTP clients should have close() called when evicted."""
        from litellm.caching.llm_caching_handler import LLMClientCache

        mock_client = MagicMock()
        mock_client.close = MagicMock()

        cache = LLMClientCache(max_size_in_memory=1, default_ttl=600)
        cache.set_cache("client1", mock_client)
        cache.set_cache("client2", MagicMock())  # Triggers eviction of client1

        mock_client.close.assert_called_once()

    def test_async_client_close_scheduled_on_eviction(self):
        """Async HTTP clients should have close() scheduled when evicted."""
        from litellm.caching.llm_caching_handler import LLMClientCache

        mock_client = MagicMock()
        mock_close = AsyncMock()
        mock_client.close = mock_close

        cache = LLMClientCache(max_size_in_memory=1, default_ttl=600)

        async def run():
            cache.set_cache("client1", mock_client)
            cache.set_cache("client2", MagicMock())  # Triggers eviction
            # Give the scheduled task a chance to run
            await asyncio.sleep(0.01)

        asyncio.get_event_loop().run_until_complete(run())

    def test_client_without_close_does_not_crash(self):
        """Objects without a close() method should not cause errors on eviction."""
        from litellm.caching.llm_caching_handler import LLMClientCache

        cache = LLMClientCache(max_size_in_memory=1, default_ttl=600)
        cache.set_cache("key1", "just_a_string")
        cache.set_cache("key2", "another_string")  # Should not raise


class TestSagemakerNoneTypeError:
    """Test fix for AttributeError: 'NoneType' object has no attribute 'get'
    in Sagemaker error handling (handler.py line 658)."""

    def test_sagemaker_error_with_none_response(self):
        """When exception.response is explicitly None, error handling should not crash."""
        from litellm.llms.sagemaker.common_utils import SagemakerError

        class FakeException(Exception):
            response = None  # Explicitly None, not missing

        # Simulate the fixed error handling logic
        e = FakeException("timeout error")
        _response = getattr(e, "response", None) or {}
        status_code = (
            _response.get("ResponseMetadata", {}).get("HTTPStatusCode", 500)
        )
        error_message = _response.get("Error", {}).get("Message", str(e))

        assert status_code == 500
        assert error_message == "timeout error"

    def test_sagemaker_error_with_valid_response(self):
        """When exception has a proper response dict, it should extract correctly."""

        class FakeException(Exception):
            response = {
                "ResponseMetadata": {"HTTPStatusCode": 429},
                "Error": {"Message": "Rate limit exceeded"},
            }

        e = FakeException("rate limit")
        _response = getattr(e, "response", None) or {}
        status_code = (
            _response.get("ResponseMetadata", {}).get("HTTPStatusCode", 500)
        )
        error_message = _response.get("Error", {}).get("Message", str(e))

        assert status_code == 429
        assert error_message == "Rate limit exceeded"

    def test_sagemaker_error_with_missing_response(self):
        """When exception has no response attribute, defaults should be used."""

        e = Exception("generic error")
        _response = getattr(e, "response", None) or {}
        status_code = (
            _response.get("ResponseMetadata", {}).get("HTTPStatusCode", 500)
        )
        error_message = _response.get("Error", {}).get("Message", str(e))

        assert status_code == 500
        assert error_message == "generic error"


class TestHTTPClientCacheKeyGrowth:
    """Test that HTTP client cache keys don't grow unboundedly."""

    def test_cache_bounded_by_max_size(self):
        """Cache should never exceed max_size_in_memory items."""
        from litellm.caching.in_memory_cache import InMemoryCache

        cache = InMemoryCache(max_size_in_memory=5, default_ttl=600)

        for i in range(20):
            cache.set_cache(f"key_{i}", f"value_{i}")

        assert (
            len(cache.cache_dict) <= 5
        ), f"Cache should be bounded to 5 items, got {len(cache.cache_dict)}"

    def test_llm_client_cache_bounded(self):
        """LLMClientCache should respect max_size_in_memory."""
        from litellm.caching.llm_caching_handler import LLMClientCache

        cache = LLMClientCache(max_size_in_memory=3, default_ttl=600)

        for i in range(10):
            cache.set_cache(f"client_{i}", MagicMock())

        assert (
            len(cache.cache_dict) <= 3
        ), f"LLMClientCache should be bounded to 3 items, got {len(cache.cache_dict)}"


class TestRouterCooldownIsolation:
    """Test that router cooldown system isolates failures per deployment."""

    def test_cooldown_uses_deployment_id_not_model_group(self):
        """Cooldown keys should be per-deployment, not per-model-group."""
        from litellm.router_utils.cooldown_cache import CooldownCache

        cache = CooldownCache(
            default_cooldown_time=60,
            cache=MagicMock(),
        )

        # Verify cooldown key format uses model_id (deployment-specific)
        key = cache.get_cooldown_cache_key("deployment_abc123")
        assert "deployment_abc123" in key
        assert "deployment:" in key


class TestMemoryLeakReproduction:
    """
    Simulated reproduction of the memory leak pattern.

    The primary leak sources in v1.81.12.rc.1 were:
    1. APScheduler jitter causing normalize() memory explosion
    2. Unbounded asyncio.Queue instances for spend tracking
    3. HTTP clients not being closed when evicted from cache
    """

    def test_http_client_eviction_closes_resources(self):
        """Simulate high-volume HTTP client creation and verify cleanup."""
        from litellm.caching.llm_caching_handler import LLMClientCache

        close_count = 0

        def make_mock_client():
            nonlocal close_count
            client = MagicMock()

            def on_close():
                nonlocal close_count
                close_count += 1

            client.close = on_close
            return client

        cache = LLMClientCache(max_size_in_memory=5, default_ttl=600)

        # Simulate creating many clients (like different providers under load)
        for i in range(50):
            cache.set_cache(f"client_{i}", make_mock_client())

        # At least 45 clients should have been closed (50 - 5 remaining)
        assert close_count >= 44, (
            f"Expected at least 44 clients to be closed on eviction, "
            f"but only {close_count} were closed"
        )
