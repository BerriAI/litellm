import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from litellm.llms.anthropic.messages.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching import (
    ContextCachingEndpoints,
)


def test_target_consumes_cache_control_skips_when_model_disables_prompt_caching(monkeypatch):
    """LiteLLMAnthropicMessagesAdapter should not consume cache_control for models explicitly disabling prompt caching."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    assert LiteLLMAnthropicMessagesAdapter.target_consumes_cache_control("claude-3-5-sonnet-20241022") is True
    assert LiteLLMAnthropicMessagesAdapter.target_consumes_cache_control("gemini/gemini-2.5-flash") is True
    assert LiteLLMAnthropicMessagesAdapter.target_consumes_cache_control("gemini/gemini-4-argon") is False
    assert LiteLLMAnthropicMessagesAdapter.target_consumes_cache_control("vertex_ai/gemini-4-argon") is False


def test_check_and_create_cache_skips_when_model_disables_prompt_caching(monkeypatch):
    """ContextCachingEndpoints should skip explicit cache creation when model has supports_prompt_caching: false."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    endpoints = ContextCachingEndpoints()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "hello world test token count " * 1000,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
    optional_params = {"tools": [{"type": "function", "function": {"name": "test"}}]}

    res_messages, res_params, cache_id = endpoints.check_and_create_cache(
        messages=messages,
        optional_params=optional_params,
        api_key="test-key",
        api_base=None,
        model="gemini/gemini-4-argon",
        client=MagicMock(),
        timeout=10,
        logging_obj=MagicMock(),
        custom_llm_provider="gemini",
        vertex_project=None,
        vertex_location=None,
        vertex_auth_header=None,
    )
    assert cache_id is None
    # Tools should remain in params and not be popped
    assert "tools" in res_params


@pytest.mark.asyncio
async def test_async_check_and_create_cache_fallback_on_400_explicit_caching_unsupported(monkeypatch):
    """When Vertex AI returns 400 'does not support explicit context caching', gracefully fallback to non-cached request."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    endpoints = ContextCachingEndpoints()
    endpoints.async_check_cache = AsyncMock(return_value=None)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "hello world test token count " * 1000,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
    optional_params = {"tools": [{"type": "function", "function": {"name": "test"}}]}

    mock_client = MagicMock(spec=AsyncHTTPHandler)
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = json.dumps({
        "error": {
            "code": 400,
            "message": "Model gemini-4-argon does not support explicit context caching.",
            "status": "INVALID_ARGUMENT",
        }
    })
    mock_client.post = AsyncMock(
        side_effect=httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=mock_response)
    )

    res_messages, res_params, cache_id = await endpoints.async_check_and_create_cache(
        messages=messages,
        optional_params=optional_params,
        api_key="test-key",
        api_base=None,
        model="gemini/gemini-2.5-flash",
        client=mock_client,
        timeout=10,
        logging_obj=MagicMock(),
        custom_llm_provider="gemini",
        vertex_project=None,
        vertex_location=None,
        vertex_auth_header=None,
    )
    assert cache_id is None
    assert mock_client.post.called


def test_sync_check_and_create_cache_fallback_on_400_explicit_caching_unsupported(monkeypatch):
    """Sync check_and_create_cache fallback on 400 explicit caching unsupported."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    endpoints = ContextCachingEndpoints()
    endpoints.check_cache = MagicMock(return_value=None)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "hello world test token count " * 1000,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
    optional_params = {"tools": [{"type": "function", "function": {"name": "test"}}]}

    mock_client = MagicMock(spec=HTTPHandler)
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = json.dumps({
        "error": {
            "code": 400,
            "message": "Model gemini-4-argon does not support explicit context caching.",
            "status": "INVALID_ARGUMENT",
        }
    })
    mock_client.post.side_effect = httpx.HTTPStatusError(
        "Bad Request", request=MagicMock(), response=mock_response
    )

    res_messages, res_params, cache_id = endpoints.check_and_create_cache(
        messages=messages,
        optional_params=optional_params,
        api_key="test-key",
        api_base=None,
        model="gemini/gemini-2.5-flash",
        client=mock_client,
        timeout=10,
        logging_obj=MagicMock(),
        custom_llm_provider="gemini",
        vertex_project=None,
        vertex_location=None,
        vertex_auth_header=None,
    )
    assert cache_id is None
    assert mock_client.post.called
