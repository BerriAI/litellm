"""
Tests for aiohttp client session cleanup.

Ensures that:
1. ``litellm.aclose()`` closes all cached aiohttp sessions.
2. ``_close_aiohttp_sessions_sync()`` closes connectors synchronously.
3. ``_collect_aiohttp_sessions()`` finds sessions in the cache.
4. Sessions are not leaked after cleanup.

Fixes https://github.com/BerriAI/litellm/issues/13251
"""

import aiohttp
import pytest

import litellm
from litellm.llms.custom_httpx import async_client_cleanup as _cleanup_mod
from litellm.llms.custom_httpx.async_client_cleanup import (
    _close_aiohttp_sessions_sync,
    _collect_aiohttp_sessions,
    close_litellm_async_clients,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


@pytest.fixture(autouse=True)
def _clean_cache():
    """Ensure a fresh LLM client cache and reset cleanup guard for each test."""
    from litellm.caching.llm_caching_handler import LLMClientCache

    original = getattr(litellm, "in_memory_llm_clients_cache", None)
    litellm.in_memory_llm_clients_cache = LLMClientCache()
    _cleanup_mod._cleanup_done = False
    yield
    _cleanup_mod._cleanup_done = False
    litellm.in_memory_llm_clients_cache = original or LLMClientCache()


def _create_cached_handler(key: str = "test-handler") -> AsyncHTTPHandler:
    """Create an AsyncHTTPHandler and place it in the LLM client cache."""
    handler = AsyncHTTPHandler(timeout=30.0)
    cache = litellm.in_memory_llm_clients_cache
    cache.cache_dict[key] = handler
    return handler


def _force_session_creation(handler: AsyncHTTPHandler) -> aiohttp.ClientSession:
    """Force the transport to create a real aiohttp session (lazy creation)."""
    transport = getattr(handler.client, "_transport", None)
    if transport is not None and hasattr(transport, "_get_valid_client_session"):
        session = transport._get_valid_client_session()
        return session
    pytest.skip("Transport does not support _get_valid_client_session")


class TestCollectAiohttpSessions:
    """_collect_aiohttp_sessions finds sessions in the cache."""

    def test_empty_cache_returns_empty(self):
        assert _collect_aiohttp_sessions() == []

    @pytest.mark.asyncio
    async def test_finds_session_from_async_handler(self):
        handler = _create_cached_handler()
        session = _force_session_creation(handler)
        sessions = _collect_aiohttp_sessions()
        assert len(sessions) >= 1
        assert session in sessions

    @pytest.mark.asyncio
    async def test_skips_closed_sessions(self):
        handler = _create_cached_handler()
        session = _force_session_creation(handler)
        connector = session.connector
        if connector is not None:
            await connector.close()
        sessions = _collect_aiohttp_sessions()
        assert session not in sessions


class TestCloseAiohttpSessionsSync:
    """_close_aiohttp_sessions_sync closes connectors without an event loop."""

    @pytest.mark.asyncio
    async def test_closes_connector(self):
        handler = _create_cached_handler()
        session = _force_session_creation(handler)
        assert not session.closed

        _close_aiohttp_sessions_sync()

        assert session.closed

    @pytest.mark.asyncio
    async def test_idempotent(self):
        handler = _create_cached_handler()
        _force_session_creation(handler)
        _close_aiohttp_sessions_sync()
        _close_aiohttp_sessions_sync()  # should not raise


class TestCloseAsyncClients:
    """close_litellm_async_clients closes sessions via await."""

    @pytest.mark.asyncio
    async def test_closes_session(self):
        handler = _create_cached_handler()
        session = _force_session_creation(handler)
        assert not session.closed

        await close_litellm_async_clients()

        assert session.closed or handler.client.is_closed

    @pytest.mark.asyncio
    async def test_no_error_on_empty_cache(self):
        await close_litellm_async_clients()  # should not raise


class TestLitellmAclose:
    """Public litellm.aclose() API."""

    @pytest.mark.asyncio
    async def test_aclose_closes_sessions(self):
        handler = _create_cached_handler()
        session = _force_session_creation(handler)
        assert not session.closed

        await litellm.aclose()

        assert session.closed or handler.client.is_closed

    @pytest.mark.asyncio
    async def test_aclose_is_idempotent(self):
        _create_cached_handler()
        await litellm.aclose()
        await litellm.aclose()  # should not raise

    @pytest.mark.asyncio
    async def test_aclose_no_sessions(self):
        """aclose() is a no-op when no sessions exist."""
        await litellm.aclose()
