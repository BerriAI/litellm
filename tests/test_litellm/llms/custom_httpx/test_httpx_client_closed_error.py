"""
Regression tests for httpx "Cannot send a request, as the client has been closed" error.

Root cause: LLMClientCache._remove_key() used to close httpx clients on cache
eviction (TTL or size-based). But other code still held references to those
clients (e.g. litellm.module_level_aclient, in-flight requests), causing
RuntimeError when they tried to use the now-closed client.

Fix:
1. LLMClientCache._remove_key() no longer closes clients on eviction.
2. get_async_httpx_client() / _get_httpx_client() check if a cached client
   is closed before returning it; if so, create a new one.
3. AsyncHTTPHandler HTTP methods catch RuntimeError("client has been closed")
   and retry with a fresh underlying httpx.AsyncClient.
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
    _is_async_client_closed,
    _is_sync_client_closed,
    get_async_httpx_client,
    _get_httpx_client,
)


class TestLLMClientCacheNoCloseOnEviction:
    """Verify that LLMClientCache does NOT close clients when evicting them."""

    @pytest.mark.asyncio
    async def test_should_not_close_async_client_on_ttl_eviction(self):
        """Client evicted due to TTL expiry must not be closed."""
        cache = LLMClientCache(max_size_in_memory=10, default_ttl=1)

        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        cache.set_cache(key="test-client", value=mock_client, ttl=0.01)

        await asyncio.sleep(0.05)

        cache.get_cache(key="test-client")

        await asyncio.sleep(0.05)
        assert mock_client.aclose.await_count == 0

    @pytest.mark.asyncio
    async def test_should_not_close_async_client_on_size_eviction(self):
        """Client evicted due to cache size limit must not be closed."""
        cache = LLMClientCache(max_size_in_memory=2, default_ttl=600)

        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()

        cache.set_cache(key="client-a", value=mock_client)
        cache.set_cache(key="filler", value="x")
        cache.set_cache(key="trigger", value="y")

        await asyncio.sleep(0.05)
        assert mock_client.aclose.await_count == 0

    def test_should_not_close_sync_client_on_eviction(self):
        """Sync client evicted from cache must not have close() called."""
        cache = LLMClientCache(max_size_in_memory=2, default_ttl=600)

        mock_client = MagicMock()
        mock_client.close = MagicMock()

        cache.set_cache(key="sync-client", value=mock_client)
        cache.set_cache(key="filler", value="x")
        cache.set_cache(key="trigger", value="y")

        assert mock_client.close.call_count == 0


class TestIsClientClosedHelpers:
    """Verify the is_closed helper functions."""

    def test_should_detect_closed_async_client(self):
        handler = AsyncHTTPHandler()
        assert _is_async_client_closed(handler) is False
        handler.client._state = httpx._client.ClientState.CLOSED
        assert _is_async_client_closed(handler) is True

    def test_should_detect_closed_sync_client(self):
        handler = HTTPHandler()
        assert _is_sync_client_closed(handler) is False
        handler.client._state = httpx._client.ClientState.CLOSED
        assert _is_sync_client_closed(handler) is True

    def test_should_return_true_for_broken_handler(self):
        handler = AsyncHTTPHandler()
        del handler.client
        assert _is_async_client_closed(handler) is True


class TestGetAsyncHttpxClientClosedCheck:
    """get_async_httpx_client should create a new client if cached one is closed."""

    @pytest.mark.asyncio
    async def test_should_return_new_client_when_cached_is_closed(self):
        cache = LLMClientCache(max_size_in_memory=200, default_ttl=3600)
        original_cache = getattr(litellm, "in_memory_llm_clients_cache", None)

        try:
            litellm.in_memory_llm_clients_cache = cache

            client_1 = get_async_httpx_client(
                llm_provider="test_closed_provider",
            )

            await client_1.client.aclose()
            assert client_1.client.is_closed is True

            client_2 = get_async_httpx_client(
                llm_provider="test_closed_provider",
            )

            assert client_2 is not client_1
            assert client_2.client.is_closed is False
        finally:
            if original_cache is not None:
                litellm.in_memory_llm_clients_cache = original_cache

    @pytest.mark.asyncio
    async def test_should_reuse_client_when_cached_is_open(self):
        cache = LLMClientCache(max_size_in_memory=200, default_ttl=3600)
        original_cache = getattr(litellm, "in_memory_llm_clients_cache", None)

        try:
            litellm.in_memory_llm_clients_cache = cache

            client_1 = get_async_httpx_client(
                llm_provider="test_open_provider",
            )

            client_2 = get_async_httpx_client(
                llm_provider="test_open_provider",
            )

            assert client_2 is client_1
        finally:
            if original_cache is not None:
                litellm.in_memory_llm_clients_cache = original_cache


class TestGetSyncHttpxClientClosedCheck:
    """_get_httpx_client should create a new client if cached one is closed."""

    def test_should_return_new_client_when_cached_is_closed(self):
        cache = LLMClientCache(max_size_in_memory=200, default_ttl=3600)
        original_cache = getattr(litellm, "in_memory_llm_clients_cache", None)

        try:
            litellm.in_memory_llm_clients_cache = cache

            client_1 = _get_httpx_client()

            client_1.client.close()
            assert client_1.client.is_closed is True

            client_2 = _get_httpx_client()

            assert client_2 is not client_1
            assert client_2.client.is_closed is False
        finally:
            if original_cache is not None:
                litellm.in_memory_llm_clients_cache = original_cache


class TestAsyncHTTPHandlerClosedClientRecovery:
    """AsyncHTTPHandler should recover from 'client has been closed' by creating a new client."""

    @pytest.mark.asyncio
    async def test_should_recover_post_when_client_closed(self):
        handler = AsyncHTTPHandler(timeout=httpx.Timeout(5.0))

        await handler.client.aclose()
        assert handler.client.is_closed is True

        mock_response = httpx.Response(200, text="ok")
        with patch.object(
            AsyncHTTPHandler,
            "single_connection_post_request",
            return_value=mock_response,
        ) as mock_post:
            response = await handler.post(
                url="https://example.com/api",
                json={"test": True},
            )
            assert response.status_code == 200
            mock_post.assert_called_once()

        assert handler.client.is_closed is False

    @pytest.mark.asyncio
    async def test_should_recover_get_when_client_closed(self):
        handler = AsyncHTTPHandler(timeout=httpx.Timeout(5.0))

        await handler.client.aclose()
        assert handler.client.is_closed is True

        mock_response = httpx.Response(200, text="ok")
        call_count = 0

        original_get = httpx.AsyncClient.get

        async def side_effect(self_client, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if self_client.is_closed:
                raise RuntimeError("Cannot send a request, as the client has been closed.")
            return mock_response

        with patch.object(httpx.AsyncClient, "get", side_effect):
            response = await handler.get(url="https://example.com/api")
            assert response.status_code == 200

        assert handler.client.is_closed is False

    @pytest.mark.asyncio
    async def test_should_recover_put_when_client_closed(self):
        handler = AsyncHTTPHandler(timeout=httpx.Timeout(5.0))

        await handler.client.aclose()

        mock_response = httpx.Response(200, text="ok")
        with patch.object(
            AsyncHTTPHandler,
            "single_connection_post_request",
            return_value=mock_response,
        ):
            response = await handler.put(
                url="https://example.com/api",
                json={"test": True},
            )
            assert response.status_code == 200

        assert handler.client.is_closed is False

    @pytest.mark.asyncio
    async def test_should_recover_delete_when_client_closed(self):
        handler = AsyncHTTPHandler(timeout=httpx.Timeout(5.0))

        await handler.client.aclose()

        mock_response = httpx.Response(200, text="ok")
        with patch.object(
            AsyncHTTPHandler,
            "single_connection_post_request",
            return_value=mock_response,
        ):
            response = await handler.delete(
                url="https://example.com/api",
            )
            assert response.status_code == 200

        assert handler.client.is_closed is False

    @pytest.mark.asyncio
    async def test_should_recover_patch_when_client_closed(self):
        handler = AsyncHTTPHandler(timeout=httpx.Timeout(5.0))

        await handler.client.aclose()

        mock_response = httpx.Response(200, text="ok")
        with patch.object(
            AsyncHTTPHandler,
            "single_connection_post_request",
            return_value=mock_response,
        ):
            response = await handler.patch(
                url="https://example.com/api",
                json={"test": True},
            )
            assert response.status_code == 200

        assert handler.client.is_closed is False

    @pytest.mark.asyncio
    async def test_should_reraise_unrelated_runtime_error(self):
        """RuntimeErrors not related to closed client should propagate."""
        handler = AsyncHTTPHandler(timeout=httpx.Timeout(5.0))

        with patch.object(
            httpx.AsyncClient,
            "send",
            side_effect=RuntimeError("some other error"),
        ):
            with pytest.raises(RuntimeError, match="some other error"):
                await handler.post(
                    url="https://example.com/api",
                    json={"test": True},
                )


class TestCacheTTLExpiryDoesNotBreakClients:
    """End-to-end: simulate cache TTL expiry and verify client still works."""

    @pytest.mark.asyncio
    async def test_should_not_break_reference_after_ttl_expiry(self):
        """Simulate the exact production scenario:
        1. Client is cached and a reference is held externally
        2. Cache TTL expires, client is evicted
        3. External reference should still be usable (not closed)
        """
        cache = LLMClientCache(max_size_in_memory=200, default_ttl=0.01)
        original_cache = getattr(litellm, "in_memory_llm_clients_cache", None)

        try:
            litellm.in_memory_llm_clients_cache = cache

            client = get_async_httpx_client(
                llm_provider="test_ttl_provider",
            )

            external_ref = client

            await asyncio.sleep(0.05)

            _result = cache.get_cache("anything")

            assert external_ref.client.is_closed is False, (
                "Client should NOT be closed after cache eviction"
            )
        finally:
            if original_cache is not None:
                litellm.in_memory_llm_clients_cache = original_cache
