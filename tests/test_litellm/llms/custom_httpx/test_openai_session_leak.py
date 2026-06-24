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
