from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching import (
    ContextCachingEndpoints,
)

_CACHE_RESPONSE: Final = {
    "name": "cachedContents/new-cache-name",
    "model": "gemini-2.5-flash",
    "usageMetadata": {"totalTokenCount": 2048},
    "createTime": "2026-09-01T00:00:00Z",
    "expireTime": "2026-09-01T01:00:00Z",
}

_EXPECTED_CREATION: Final = {
    "name": "cachedContents/new-cache-name",
    "model": "gemini-2.5-flash",
    "total_token_count": 2048,
    "create_time": "2026-09-01T00:00:00Z",
    "expire_time": "2026-09-01T01:00:00Z",
}


@patch(
    "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
)
@patch("litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.local_cache_obj")
@patch(
    "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.transform_openai_messages_to_gemini_context_caching"
)
@patch(
    "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.is_prompt_caching_valid_prompt"
)
@patch.object(ContextCachingEndpoints, "check_cache")
@patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching")
def test_check_and_create_cache_stores_cached_content_creation_on_logging(
    mock_get_token_url, mock_check_cache, mock_valid_prompt, mock_transform, mock_cache_obj, mock_separate
) -> None:
    mock_separate.return_value = ([{"role": "user", "content": "cached"}], [{"role": "user", "content": "fresh"}])
    mock_cache_obj.get_cache_key.return_value = "test_cache_key"
    mock_check_cache.return_value = None
    mock_valid_prompt.return_value = True
    mock_get_token_url.return_value = ("token", "https://test-url.com")
    mock_transform.return_value = {"model": "gemini-2.5-flash", "contents": []}
    response: Final = MagicMock()
    response.json.return_value = _CACHE_RESPONSE
    client: Final = MagicMock(spec=HTTPHandler)
    client.post.return_value = response
    logging_obj: Final = MagicMock(spec=Logging)
    logging_obj.model_call_details = {}

    ContextCachingEndpoints().check_and_create_cache(
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        api_key="test_key",
        api_base=None,
        model="gemini-2.5-flash",
        client=client,
        timeout=30.0,
        logging_obj=logging_obj,
        custom_llm_provider="vertex_ai",
        vertex_project="test_project",
        vertex_location="us-central1",
        vertex_auth_header="vertex_test_token",
    )

    assert logging_obj.model_call_details["vertex_ai_cached_content"] == _EXPECTED_CREATION


@pytest.mark.asyncio
@patch(
    "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
)
@patch("litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.local_cache_obj")
@patch(
    "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.transform_openai_messages_to_gemini_context_caching"
)
@patch(
    "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.is_prompt_caching_valid_prompt"
)
@patch.object(ContextCachingEndpoints, "async_check_cache")
@patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching")
async def test_async_check_and_create_cache_stores_cached_content_creation_on_logging(
    mock_get_token_url, mock_async_check_cache, mock_valid_prompt, mock_transform, mock_cache_obj, mock_separate
) -> None:
    mock_separate.return_value = ([{"role": "user", "content": "cached"}], [{"role": "user", "content": "fresh"}])
    mock_cache_obj.get_cache_key.return_value = "test_cache_key"
    mock_async_check_cache.return_value = None
    mock_valid_prompt.return_value = True
    mock_get_token_url.return_value = ("token", "https://test-url.com")
    mock_transform.return_value = {"model": "gemini-2.5-flash", "contents": []}
    response: Final = MagicMock()
    response.json.return_value = _CACHE_RESPONSE
    client: Final = MagicMock(spec=AsyncHTTPHandler)
    client.post = AsyncMock(return_value=response)
    logging_obj: Final = MagicMock(spec=Logging)
    logging_obj.model_call_details = {}

    await ContextCachingEndpoints().async_check_and_create_cache(
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        api_key="test_key",
        api_base=None,
        model="gemini-2.5-flash",
        client=client,
        timeout=30.0,
        logging_obj=logging_obj,
        custom_llm_provider="vertex_ai",
        vertex_project="test_project",
        vertex_location="us-central1",
        vertex_auth_header="vertex_test_token",
    )

    assert logging_obj.model_call_details["vertex_ai_cached_content"] == _EXPECTED_CREATION
