import pytest

import litellm
from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


@pytest.mark.asyncio
async def test_second_cleanup_pass_does_not_resurrect_owned_client():
    handler = AsyncHTTPHandler()
    original_client = handler._client
    cache_key = "test-cleanup-no-resurrect"
    litellm.in_memory_llm_clients_cache.cache_dict[cache_key] = handler
    try:
        await close_litellm_async_clients()
        assert original_client.is_closed
        await close_litellm_async_clients()
    finally:
        litellm.in_memory_llm_clients_cache.cache_dict.pop(cache_key, None)

    assert handler._client is original_client
