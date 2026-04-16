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


class _ClosableClient:
    """Mock httpx-like client that exposes ``is_closed``."""

    def __init__(self, closed: bool = False):
        self.is_closed = closed


class _WrappedClient:
    """Mock SDK wrapper (AsyncAzureOpenAI-style) whose httpx is on ``_client``."""

    def __init__(self, inner):
        self._client = inner


def test_get_cache_returns_value_when_client_is_open():
    """Baseline: an open cached client comes back unchanged."""
    cache = LLMClientCache(max_size_in_memory=2)
    inner = _ClosableClient(closed=False)
    cache.set_cache("k", _WrappedClient(inner))

    assert cache.get_cache("k") is not None
    assert "k-" + str(id(None)) not in cache.cache_dict  # sanity: no loop id yet
    # Key lives under the resolved (no-loop) key because this test is sync.
    assert "k" in cache.cache_dict


def test_get_cache_drops_wrapped_closed_client():
    """If the inner httpx on ``_client`` reports is_closed=True, the cached
    wrapper is evicted and None is returned so the caller builds a fresh one.
    """
    cache = LLMClientCache(max_size_in_memory=2)
    inner = _ClosableClient(closed=True)
    wrapped = _WrappedClient(inner)
    cache.set_cache("k", wrapped)
    # Pre-condition: the entry is there.
    assert "k" in cache.cache_dict

    result = cache.get_cache("k")

    assert result is None
    assert "k" not in cache.cache_dict  # evicted


def test_get_cache_drops_handler_closed_client():
    """AsyncHTTPHandler-style wrapper stores httpx on ``.client`` (no underscore).
    That shape should also be detected.
    """
    cache = LLMClientCache(max_size_in_memory=2)

    class _Handler:
        def __init__(self, inner):
            self.client = inner

    inner = _ClosableClient(closed=True)
    cache.set_cache("k", _Handler(inner))

    assert cache.get_cache("k") is None
    assert "k" not in cache.cache_dict


def test_get_cache_drops_bare_closed_httpx_client():
    """A bare ``httpx.AsyncClient`` with ``is_closed=True`` is also evicted."""
    cache = LLMClientCache(max_size_in_memory=2)
    cache.set_cache("k", _ClosableClient(closed=True))

    assert cache.get_cache("k") is None
    assert "k" not in cache.cache_dict


def test_get_cache_leaves_value_without_is_closed_attr_alone():
    """Non-httpx values (e.g. aiohttp handlers) don't expose is_closed — we
    must not false-positive on them and evict otherwise-fine entries.
    """
    cache = LLMClientCache(max_size_in_memory=2)

    class _AiohttpLike:
        """Deliberately no is_closed attribute."""

        pass

    value = _AiohttpLike()
    cache.set_cache("k", value)

    assert cache.get_cache("k") is value
    assert "k" in cache.cache_dict


@pytest.mark.asyncio
async def test_async_get_cache_drops_closed_client():
    """Same eviction-on-closed behaviour from the async read path."""
    cache = LLMClientCache(max_size_in_memory=2)
    wrapped = _WrappedClient(_ClosableClient(closed=True))
    await cache.async_set_cache("k", wrapped)

    assert await cache.async_get_cache("k") is None
    # Under a running loop the resolved key includes the loop id; check that
    # no stale "k..." key remains in the underlying dict.
    assert not any(str(key).startswith("k-") for key in cache.cache_dict)


def test_get_cache_survives_weird_is_closed_raising():
    """A cached object whose is_closed access raises should be treated as open,
    not crash the cache read.
    """
    cache = LLMClientCache(max_size_in_memory=2)

    class _Hostile:
        @property
        def is_closed(self):
            raise RuntimeError("boom")

    value = _Hostile()
    cache.set_cache("k", value)
    # Should not raise, and should return the value.
    assert cache.get_cache("k") is value
