import json

import pytest

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.api_route.chat.transformation import APIRouteChatConfig


def test_api_route_provider_routing():
    model, provider, api_key, api_base = get_llm_provider(
        model="api_route/model-name",
        custom_llm_provider=None,
        api_key="test-key",
        api_base="https://example.com/v1",
    )

    assert model == "model-name"
    assert provider == "api_route"
    assert api_key == "test-key"
    assert api_base == "https://example.com/v1"


def test_api_route_credentials_from_environment(monkeypatch):
    monkeypatch.setenv("API_ROUTE_API_KEY", "env-key")
    monkeypatch.setenv("API_ROUTE_BASE_URL", "https://env.example.com/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="api_route/model-name",
        custom_llm_provider=None,
        api_key=None,
        api_base=None,
    )

    assert provider == "api_route"
    assert api_key == "env-key"
    assert api_base == "https://env.example.com/v1"


def test_api_route_requires_base_url(monkeypatch):
    monkeypatch.delenv("API_ROUTE_BASE_URL", raising=False)

    with pytest.raises(
        ValueError,
        match=r"API Route requires API_ROUTE_BASE_URL or api_base parameter\.",
    ):
        APIRouteChatConfig()._get_openai_compatible_provider_info(
            api_base=None,
            api_key="test-key",
        )


def test_api_route_chat_completion_request(monkeypatch, respx_mock):
    monkeypatch.setenv("API_ROUTE_BASE_URL", "https://env.example.com/v1")
    route = respx_mock.post("https://env.example.com/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "model-name",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    response = litellm.completion(
        model="api_route/model-name",
        messages=[{"role": "user", "content": "Hello!"}],
        api_key="test-key",
        tools=tools,
    )

    request = route.calls[0].request
    body = json.loads(request.content)
    assert str(request.url) == "https://env.example.com/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert body["model"] == "model-name"
    assert body["messages"] == [{"role": "user", "content": "Hello!"}]
    assert body["tools"] == tools
    assert response.choices[0].message.content == "Hello!"
