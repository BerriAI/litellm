"""
Tests for LLMClientCache.

The cache intentionally does NOT close clients on eviction because evicted
clients may still be referenced by in-flight requests.  Closing them eagerly
causes ``RuntimeError: Cannot send a request, as the client has been closed.``

See: https://github.com/BerriAI/litellm/pull/22247
"""

import asyncio
import os
import sys
import warnings

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.caching.llm_caching_handler import LLMClientCache


class MockAsyncClient:
    """Mock async HTTP client with an async close method."""

    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class MockSyncClient:
    """Mock sync HTTP client with a sync close method."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_remove_key_does_not_close_async_client():
    """
    Evicting an async client from LLMClientCache must NOT close it because
    an in-flight request may still hold a reference to the client.

    Regression test for production 'client has been closed' crashes.
    """
    cache = LLMClientCache(max_size_in_memory=2)

    mock_client = MockAsyncClient()
    cache.cache_dict["test-key"] = mock_client
    cache.ttl_dict["test-key"] = 0  # expired

    cache._remove_key("test-key")
    # Give the event loop a chance to run any background tasks
    await asyncio.sleep(0.1)

    # Client must NOT be closed — it may still be in use
    assert mock_client.closed is False
    assert "test-key" not in cache.cache_dict
    assert "test-key" not in cache.ttl_dict


def test_remove_key_does_not_close_sync_client():
    """
    Evicting a sync client from the cache must NOT close it.
    """
    cache = LLMClientCache(max_size_in_memory=2)

    mock_client = MockSyncClient()
    cache.cache_dict["test-key"] = mock_client
    cache.ttl_dict["test-key"] = 0

    cache._remove_key("test-key")

    assert mock_client.closed is False
    assert "test-key" not in cache.cache_dict


@pytest.mark.asyncio
async def test_eviction_does_not_close_async_clients():
    """
    When the cache is full and an entry is evicted, the evicted async client
    must remain open and must not produce 'coroutine was never awaited' warnings.
    """
    cache = LLMClientCache(max_size_in_memory=2, default_ttl=1)

    clients = []
    for i in range(2):
        client = MockAsyncClient()
        clients.append(client)
        cache.set_cache(f"key-{i}", client)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        # This should trigger eviction of one of the existing entries
        cache.set_cache("key-new", "new-value")
        await asyncio.sleep(0.1)

    coroutine_warnings = [
        w for w in caught_warnings if "coroutine" in str(w.message).lower()
    ]
    assert (
        len(coroutine_warnings) == 0
    ), f"Got unawaited coroutine warnings: {coroutine_warnings}"

    # Evicted clients must NOT be closed
    for client in clients:
        assert client.closed is False


@pytest.mark.asyncio
async def test_eviction_no_unawaited_coroutine_warning():
    """
    Evicting an async client from LLMClientCache must not produce
    'coroutine was never awaited' warnings.

    Regression test for https://github.com/BerriAI/litellm/issues/22128
    """
    cache = LLMClientCache(max_size_in_memory=2)

    mock_client = MockAsyncClient()
    cache.cache_dict["test-key"] = mock_client
    cache.ttl_dict["test-key"] = 0  # expired

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        cache._remove_key("test-key")
        await asyncio.sleep(0.1)

    coroutine_warnings = [
        w for w in caught_warnings if "coroutine" in str(w.message).lower()
    ]
    assert (
        len(coroutine_warnings) == 0
    ), f"Got unawaited coroutine warnings: {coroutine_warnings}"


def test_remove_key_no_event_loop():
    """
    _remove_key works correctly even when there's no running event loop.
    """
    cache = LLMClientCache(max_size_in_memory=2)

    mock_client = MockAsyncClient()
    cache.cache_dict["test-key"] = mock_client
    cache.ttl_dict["test-key"] = 0

    # Should not raise even though there's no running event loop
    cache._remove_key("test-key")
    assert "test-key" not in cache.cache_dict


def test_remove_key_removes_plain_values():
    """
    _remove_key correctly removes non-client values (strings, dicts, etc.).
    """
    cache = LLMClientCache(max_size_in_memory=5)

    cache.cache_dict["str-key"] = "hello"
    cache.ttl_dict["str-key"] = 0
    cache.cache_dict["dict-key"] = {"foo": "bar"}
    cache.ttl_dict["dict-key"] = 0

    cache._remove_key("str-key")
    cache._remove_key("dict-key")

    assert "str-key" not in cache.cache_dict
    assert "dict-key" not in cache.cache_dict
