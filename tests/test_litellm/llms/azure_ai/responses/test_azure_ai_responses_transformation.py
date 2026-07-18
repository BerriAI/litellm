"""
Regression tests for native Azure AI Foundry Responses API routing (LIT-4427).

Before the fix, `azure_ai` had no native Responses config, so `litellm.responses()`
fell back to the chat-completions bridge and sent `reasoning_effort` + function tools
to `/chat/completions`, which Azure rejects for GPT-5 models. These tests assert the
request now goes to the native `/openai/v1/responses` endpoint in Responses shape.
"""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import litellm
from litellm.llms.azure_ai.responses.transformation import AzureAIResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager


class MockResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data)
        self.headers = httpx.Headers({})

    def json(self):
        return self._json_data


def _minimal_responses_payload(model: str) -> dict:
    return {
        "id": "resp_123",
        "object": "response",
        "created_at": 1741369938,
        "status": "completed",
        "model": model,
        "output": [],
        "parallel_tool_calls": False,
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "error": None,
        "tool_choice": "auto",
        "tools": [],
        "metadata": None,
        "temperature": None,
        "top_p": None,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": None,
        "truncation": None,
        "instructions": None,
        "incomplete_details": None,
        "user": None,
    }


@pytest.mark.parametrize(
    "model",
    ["gpt-5.6-luna-20260710154139", "gpt-5.5-20260504143601", "DeepSeek-R1-0528"],
)
def test_azure_ai_resolves_native_responses_config(model):
    config = ProviderConfigManager.get_provider_responses_api_config(provider="azure_ai", model=model)
    assert isinstance(config, AzureAIResponsesAPIConfig)


@pytest.mark.parametrize("model", ["claude-3-5-sonnet", "model_router/gpt-5", "agents/my-agent"])
def test_azure_ai_non_responses_models_keep_bridge(model):
    """Claude / model-router / agents routes have their own surfaces, so they must
    keep returning None (chat-completions bridge)."""
    config = ProviderConfigManager.get_provider_responses_api_config(provider="azure_ai", model=model)
    assert config is None


@pytest.mark.parametrize(
    "api_base,expected",
    [
        (
            "https://res.services.ai.azure.com/api/projects/proj",
            "https://res.services.ai.azure.com/api/projects/proj/openai/v1/responses",
        ),
        (
            "https://res.services.ai.azure.com/api/projects/proj/",
            "https://res.services.ai.azure.com/api/projects/proj/openai/v1/responses",
        ),
        (
            "https://res.services.ai.azure.com",
            "https://res.services.ai.azure.com/openai/v1/responses",
        ),
        (
            "https://res.openai.azure.com",
            "https://res.openai.azure.com/openai/v1/responses",
        ),
        (
            "https://res.services.ai.azure.com/api/projects/proj/openai/v1/responses",
            "https://res.services.ai.azure.com/api/projects/proj/openai/v1/responses",
        ),
    ],
)
def test_get_complete_url(api_base, expected):
    config = AzureAIResponsesAPIConfig()
    assert config.get_complete_url(api_base=api_base, litellm_params={}) == expected


def test_validate_environment_api_key_header_for_foundry_host():
    config = AzureAIResponsesAPIConfig()
    headers = config.validate_environment(
        headers={},
        model="gpt-5.6-luna",
        litellm_params=GenericLiteLLMParams(
            api_key="secret", api_base="https://res.services.ai.azure.com/api/projects/proj"
        ),
    )
    assert headers["api-key"] == "secret"
    assert "Authorization" not in headers


def test_validate_environment_bearer_for_serverless_host():
    config = AzureAIResponsesAPIConfig()
    headers = config.validate_environment(
        headers={},
        model="gpt-5.6-luna",
        litellm_params=GenericLiteLLMParams(
            api_key="secret", api_base="https://endpoint.eastus.models.ai.azure.com"
        ),
    )
    assert headers["Authorization"] == "Bearer secret"
    assert "api-key" not in headers


@pytest.mark.asyncio
async def test_aresponses_routes_to_native_endpoint_with_reasoning_and_tools():
    """Core LIT-4427 regression: reasoning_effort + function tools must be sent to the
    native /openai/v1/responses endpoint in Responses shape, not bridged to /chat/completions."""
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(_minimal_responses_payload("gpt-5.6-luna"), 200)

        await litellm.aresponses(
            model="azure_ai/gpt-5.6-luna-20260710154139",
            input="What is the weather in SF?",
            reasoning_effort="high",
            tools=tools,
            api_base="https://res.services.ai.azure.com/api/projects/proj",
            api_key="fake-key",
        )

    mock_post.assert_called_once()
    url = str(mock_post.call_args.kwargs["url"])
    body = mock_post.call_args.kwargs["json"]

    assert url == "https://res.services.ai.azure.com/api/projects/proj/openai/v1/responses"
    assert "/chat/completions" not in url
    assert "input" in body
    assert "messages" not in body
    assert body["reasoning"] == {"effort": "high"}
    assert body["tools"] == tools
