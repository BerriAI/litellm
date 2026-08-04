#!/usr/bin/env python3
"""
Test for the AsyncOpenAI cleanup gap: close_litellm_async_clients() must close
cached clients whose async teardown is named close() (the openai SDK's
AsyncOpenAI / AsyncAzureOpenAI), not only those exposing aclose().

Proposed path: tests/test_litellm/llms/custom_httpx/test_openai_session_leak.py
"""

import pytest

import litellm
from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients


class _AsyncCloseOnlyClient:
    """Mirrors openai.AsyncOpenAI: async close(), no aclose(), no .client."""

    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class _AsyncACloseOnlyClient:
    """The pre-existing path: an async client whose teardown is aclose()."""

    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


class _SyncCloseAsyncACloseClient:
    """
    The `or` short-circuit case: a sync aclose() alongside an async close().

    `getattr(handler, "aclose") or getattr(handler, "close")` would pick the
    truthy-but-synchronous aclose(), fail the iscoroutinefunction guard, and then
    never reach the real async close() -- leaking the client. Each candidate must
    be checked independently.
    """

    def __init__(self):
        self.closed = False

    def aclose(self):  # sync -- a decoy that must not satisfy cleanup
        pass

    async def close(self):
        self.closed = True


class _SyncClient:
    """A sync client (close() is not a coroutine) -- must be left untouched."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_cleanup_closes_async_close_only_clients():
    async_client = _AsyncCloseOnlyClient()
    sync_client = _SyncClient()
    litellm.in_memory_llm_clients_cache.set_cache(
        key="test_async_close_only", value=async_client, ttl=60
    )
    litellm.in_memory_llm_clients_cache.set_cache(
        key="test_sync_close", value=sync_client, ttl=60
    )

    await close_litellm_async_clients()

    assert async_client.closed, "async close()-only client should be closed"
    assert not sync_client.closed, "sync client should be left untouched"


@pytest.mark.asyncio
async def test_cleanup_closes_async_aclose_clients():
    """Regression guard for the pre-existing aclose() path."""
    aclose_client = _AsyncACloseOnlyClient()
    litellm.in_memory_llm_clients_cache.set_cache(
        key="test_async_aclose_only", value=aclose_client, ttl=60
    )

    await close_litellm_async_clients()

    assert aclose_client.closed, "async aclose() client should be closed"


@pytest.mark.asyncio
async def test_cleanup_prefers_async_close_over_sync_aclose():
    """A sync aclose() must not shadow a real async close() (the `or` gap)."""
    client = _SyncCloseAsyncACloseClient()
    litellm.in_memory_llm_clients_cache.set_cache(
        key="test_sync_aclose_async_close", value=client, ttl=60
    )

    await close_litellm_async_clients()

    assert client.closed, "async close() should be called when aclose() is sync"
