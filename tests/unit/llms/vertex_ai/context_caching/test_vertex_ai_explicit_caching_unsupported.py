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
from litellm.llms.vertex_ai.gemini.transformation import (
    sync_transform_request_body,
)


def test_target_consumes_cache_control_skips_when_model_disables_explicit_caching(local_model_cost_map):
    """LiteLLMAnthropicMessagesAdapter should not consume cache_control for gemini-4 models that disable explicit caching."""
    assert LiteLLMAnthropicMessagesAdapter.target_consumes_cache_control("claude-3-5-sonnet-20241022") is True
    assert LiteLLMAnthropicMessagesAdapter.target_consumes_cache_control("gemini/gemini-2.5-flash") is True
    assert LiteLLMAnthropicMessagesAdapter.target_consumes_cache_control("gemini/gemini-4-argon") is False
    assert LiteLLMAnthropicMessagesAdapter.target_consumes_cache_control("vertex_ai/gemini-4-argon") is False
    assert LiteLLMAnthropicMessagesAdapter.target_consumes_cache_control("gemini/gemini-4-carbon") is False
    assert LiteLLMAnthropicMessagesAdapter.target_consumes_cache_control("vertex_ai/gemini-4-carbon") is False
    assert LiteLLMAnthropicMessagesAdapter.target_consumes_cache_control("vertex_ai/gemini-4-carbon-cyber-permissive") is False
    # Bare model name without provider prefix should also be recognized
    assert LiteLLMAnthropicMessagesAdapter.target_consumes_cache_control("gemini-4-argon") is False
    assert LiteLLMAnthropicMessagesAdapter.target_consumes_cache_control("gemini-4-carbon") is False


def test_check_and_create_cache_skips_when_model_disables_explicit_caching(local_model_cost_map):
    """ContextCachingEndpoints should skip explicit cache creation when model has supports_explicit_prompt_caching: false."""
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
    # All messages and tools should remain intact
    assert len(res_messages) == len(messages)
    assert "tools" in res_params


@pytest.mark.asyncio
async def test_async_check_and_create_cache_fallback_on_400_explicit_caching_unsupported(local_model_cost_map):
    """When Vertex AI returns 400 'does not support explicit context caching', gracefully fallback to non-cached request."""
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
    # Critical checks: messages must not be dropped, tools must not be stripped!
    assert len(res_messages) == len(messages)
    assert "tools" in res_params


def test_sync_check_and_create_cache_fallback_on_400_explicit_caching_unsupported(local_model_cost_map):
    """Sync check_and_create_cache fallback on 400 explicit caching unsupported."""
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
    # Critical checks: messages must not be dropped, tools must not be stripped!
    assert len(res_messages) == len(messages)
    assert "tools" in res_params


def test_sync_transform_request_body_skips_caching_on_model_info_override(monkeypatch):
    """sync_transform_request_body should respect supports_prompt_caching: false in model_info."""
    mock_endpoints = MagicMock()
    mock_endpoints.check_and_create_cache.return_value = ([], {}, None)
    monkeypatch.setattr(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.ContextCachingEndpoints",
        MagicMock(return_value=mock_endpoints),
    )

    messages = [{"role": "user", "content": "hello"}]
    sync_transform_request_body(
        messages=messages,
        model="gemini-2.5-flash",
        custom_llm_provider="gemini",
        litellm_params={"model_info": {"supports_explicit_prompt_caching": False}},
        optional_params={},
        gemini_api_key="test-key",
        api_base=None,
        client=None,
        timeout=None,
        extra_headers=None,
        logging_obj=MagicMock(),
        vertex_project=None,
        vertex_location=None,
        vertex_auth_header=None,
    )
    # context_caching=False must be passed to check_and_create_cache
    assert mock_endpoints.check_and_create_cache.called
    _, kwargs = mock_endpoints.check_and_create_cache.call_args
    assert kwargs.get("context_caching") is False


def test_gemini_4_rows_keep_implicit_prompt_caching(local_model_cost_map):
    """supports_prompt_caching must stay true (implicit caching) while explicit caching is disabled."""
    import litellm

    for model in (
        "gemini/gemini-4-argon",
        "vertex_ai/gemini-4-argon",
        "vertex_ai/gemini-4-argon-cyber-permissive",
        "gemini/gemini-4-carbon",
        "vertex_ai/gemini-4-carbon",
        "vertex_ai/gemini-4-carbon-cyber-permissive",
    ):
        info = litellm.get_model_info(model)
        assert info.get("supports_prompt_caching") is True, model
        assert info.get("supports_explicit_prompt_caching") is False, model
