"""
Unit tests for the CommentLLM OpenAI-like provider.
"""

import json
import os
import sys
from typing import Final

import pytest
import respx

import litellm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

COMMENTLLM_BASE_URL: Final = "https://api.commentllm.com/v1"
COMMENTLLM_MODELS: Final = (
    "commentllm/deepseek-v4.1-flash-abliterated",
    "commentllm/deepseek-v4.1-flash-abliterated-fast",
    "commentllm/glm-5.3-abliterated",
    "commentllm/glm-5.3-flash-abliterated",
)


def _get_config():
    provider = JSONProviderRegistry.get("commentllm")
    assert provider is not None
    config_class = create_config_class(provider)
    return config_class()


def test_commentllm_provider_registered():
    provider = JSONProviderRegistry.get("commentllm")
    assert provider is not None
    assert provider.base_url == COMMENTLLM_BASE_URL
    assert provider.api_key_env == "COMMENTLLM_API_KEY"


def test_commentllm_resolves_env_api_key(monkeypatch):
    config = _get_config()
    monkeypatch.setenv("COMMENTLLM_API_KEY", "test-key")
    api_base, api_key = config.get_openai_compatible_provider_info(None, None)
    assert api_base == COMMENTLLM_BASE_URL
    assert api_key == "test-key"


def test_commentllm_complete_url_appends_endpoint():
    config = _get_config()
    url = config.get_complete_url(
        api_base=COMMENTLLM_BASE_URL,
        api_key="test-key",
        model="commentllm/deepseek-v4.1-flash-abliterated",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{COMMENTLLM_BASE_URL}/chat/completions"


def test_commentllm_models_have_prices_and_limits(local_model_cost_map: None) -> None:
    for model in COMMENTLLM_MODELS:
        row = litellm.model_cost[model]
        assert row["litellm_provider"] == "commentllm"
        assert row["mode"] == "chat"
        assert row["max_output_tokens"] <= row["max_input_tokens"]
        assert row["cache_read_input_token_cost"] < row["input_cost_per_token"] < row["output_cost_per_token"]
        assert row["supports_function_calling"] is True
        # The API refuses response_format and reasoning controls: https://commentllm.com/docs
        assert "supports_response_schema" not in row
        assert "supports_reasoning" not in row


def test_commentllm_chat_sends_the_key_and_accounts_for_cache(local_model_cost_map: None) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{COMMENTLLM_BASE_URL}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-v4.1-flash-abliterated",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "total_tokens": 105,
                    "prompt_tokens_details": {"cached_tokens": 80},
                },
            },
        )
        response: Final = litellm.completion(
            model="commentllm/deepseek-v4.1-flash-abliterated",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "Say hello"}],
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert body["model"] == "deepseek-v4.1-flash-abliterated"
    assert response.choices[0].message.content == "hello"
    rates: Final = litellm.model_cost["commentllm/deepseek-v4.1-flash-abliterated"]
    expected: Final = (
        20 * rates["input_cost_per_token"]
        + 80 * rates["cache_read_input_token_cost"]
        + 5 * rates["output_cost_per_token"]
    )
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)
