"""
Test that httpx clients are NOT closed when evicted from the LLMClientCache.

Reproduces the bug: "Cannot send a request, as the client has been closed."
Root cause: LLMClientCache._remove_key() was closing async/sync httpx clients
on cache eviction (TTL expiry or LRU eviction), but callers still held
references to those clients.

See: https://github.com/BerriAI/litellm/issues/XXXX
"""

import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)


class TestLLMClientCacheEvictionDoesNotCloseClients:
    """Verify that cache eviction does not close clients that may still be in use."""

    def test_async_client_not_closed_on_ttl_expiry(self):
        """
        Simulate TTL expiry: a cached async client should NOT be closed when
        its cache entry expires, because callers may still hold a reference.
        """
        cache = LLMClientCache()

        mock_client = AsyncMock(spec=AsyncHTTPHandler)
        mock_client.close = AsyncMock()
        mock_client.aclose = AsyncMock()

        cache.set_cache("test_async_client", mock_client, ttl=1)

        cached = cache.get_cache("test_async_client")
        assert cached is mock_client

        time.sleep(1.5)

        expired = cache.get_cache("test_async_client")
        assert expired is None

        mock_client.close.assert_not_called()
        mock_client.aclose.assert_not_called()

    def test_sync_client_not_closed_on_ttl_expiry(self):
        """
        Simulate TTL expiry for a sync client: should NOT be closed on eviction.
        """
        cache = LLMClientCache()

        mock_client = MagicMock(spec=HTTPHandler)
        mock_client.close = MagicMock()

        cache.set_cache("test_sync_client", mock_client, ttl=1)

        cached = cache.get_cache("test_sync_client")
        assert cached is mock_client

        time.sleep(1.5)

        expired = cache.get_cache("test_sync_client")
        assert expired is None

        mock_client.close.assert_not_called()

    def test_client_not_closed_on_lru_eviction(self):
        """
        When cache hits max_size and evicts LRU entries, evicted clients
        should NOT be closed.
        """
        cache = LLMClientCache(max_size_in_memory=2)

        client_a = AsyncMock(spec=AsyncHTTPHandler)
        client_a.aclose = AsyncMock()
        client_b = AsyncMock(spec=AsyncHTTPHandler)
        client_b.aclose = AsyncMock()
        client_c = AsyncMock(spec=AsyncHTTPHandler)
        client_c.aclose = AsyncMock()

        cache.set_cache("client_a", client_a, ttl=3600)
        cache.set_cache("client_b", client_b, ttl=3600)
        cache.set_cache("client_c", client_c, ttl=3600)

        client_a.aclose.assert_not_called()
        client_b.aclose.assert_not_called()
        client_c.aclose.assert_not_called()

    @pytest.mark.asyncio
    async def test_module_level_aclient_survives_cache_eviction(self):
        """
        Reproduce the exact production scenario:
        1. get_async_httpx_client() creates a client and caches it
        2. The client is also stored as litellm.module_level_aclient
        3. Cache TTL expires, entry is evicted
        4. The module-level reference should still point to a USABLE client
        """
        cache = LLMClientCache()

        with patch.object(litellm, "in_memory_llm_clients_cache", cache):
            client = get_async_httpx_client(
                llm_provider="test_provider",
                params={"timeout": 10, "client_alias": "test"},
            )

            assert client is not None
            assert not client.client.is_closed

            for key in list(cache.cache_dict.keys()):
                cache.ttl_dict[key] = time.time() - 1

            cache.evict_cache()

            assert not client.client.is_closed, (
                "Client must NOT be closed after cache eviction - "
                "callers still hold references to it"
            )

    @pytest.mark.asyncio
    async def test_get_async_httpx_client_returns_new_client_after_eviction(self):
        """
        After cache eviction, get_async_httpx_client should return a new client
        while the old client remains usable (not closed).
        """
        cache = LLMClientCache()

        with patch.object(litellm, "in_memory_llm_clients_cache", cache):
            old_client = get_async_httpx_client(
                llm_provider="test_provider2",
                params={"timeout": 10, "client_alias": "test2"},
            )

            for key in list(cache.cache_dict.keys()):
                cache.ttl_dict[key] = time.time() - 1

            new_client = get_async_httpx_client(
                llm_provider="test_provider2",
                params={"timeout": 10, "client_alias": "test2"},
            )

            assert old_client is not new_client, "Should create a new client after eviction"
            assert not old_client.client.is_closed, "Old client must remain usable"
            assert not new_client.client.is_closed, "New client must be usable"
