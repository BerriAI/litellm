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
async def test_remove_key_no_unawaited_coroutine_warning():
    """
    Test that evicting an async client from LLMClientCache does not produce
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
        # Let the event loop process the close task
        await asyncio.sleep(0.1)

    coroutine_warnings = [
        w for w in caught_warnings if "coroutine" in str(w.message).lower()
    ]
    assert (
        len(coroutine_warnings) == 0
    ), f"Got unawaited coroutine warnings: {coroutine_warnings}"


@pytest.mark.asyncio
async def test_remove_key_closes_async_client():
    """
    Test that evicting an async client from the cache properly closes it.
    """
    cache = LLMClientCache(max_size_in_memory=2)

    mock_client = MockAsyncClient()
    cache.cache_dict["test-key"] = mock_client
    cache.ttl_dict["test-key"] = 0

    cache._remove_key("test-key")
    # Let the event loop process the close task
    await asyncio.sleep(0.1)

    assert mock_client.closed is True
    assert "test-key" not in cache.cache_dict
    assert "test-key" not in cache.ttl_dict


def test_remove_key_closes_sync_client():
    """
    Test that evicting a sync client from the cache properly closes it.
    """
    cache = LLMClientCache(max_size_in_memory=2)

    mock_client = MockSyncClient()
    cache.cache_dict["test-key"] = mock_client
    cache.ttl_dict["test-key"] = 0

    cache._remove_key("test-key")

    assert mock_client.closed is True
    assert "test-key" not in cache.cache_dict


@pytest.mark.asyncio
async def test_eviction_closes_async_clients():
    """
    Test that cache eviction (when cache is full) properly closes async clients
    without producing warnings.
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


def test_remove_key_no_event_loop():
    """
    Test that _remove_key doesn't raise when there's no running event loop
    (falls through to the RuntimeError except branch).
    """
    cache = LLMClientCache(max_size_in_memory=2)

    mock_client = MockAsyncClient()
    cache.cache_dict["test-key"] = mock_client
    cache.ttl_dict["test-key"] = 0

    # Should not raise even though there's no running event loop
    cache._remove_key("test-key")
    assert "test-key" not in cache.cache_dict


@pytest.mark.asyncio
async def test_background_tasks_cleaned_up_after_completion():
    """
    Test that completed close tasks are removed from the _background_tasks set.
    """
    cache = LLMClientCache(max_size_in_memory=2)

    mock_client = MockAsyncClient()
    cache.cache_dict["test-key"] = mock_client
    cache.ttl_dict["test-key"] = 0

    cache._remove_key("test-key")
    # Let the task complete
    await asyncio.sleep(0.1)

    assert len(cache._background_tasks) == 0
