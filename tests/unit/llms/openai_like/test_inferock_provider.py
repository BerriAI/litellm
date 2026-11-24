import json
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.utils import trim_messages


def test_inferock_resolves_and_normalizes_gateway_base_url(
    monkeypatch: pytest.MonkeyPatch, local_model_cost_map: None
) -> None:
    monkeypatch.delenv("INFEROCK_API_BASE", raising=False)
    monkeypatch.delenv("GATEWAY_BASE_URL", raising=False)
    monkeypatch.setenv("INFEROCK_API_KEY", "fake-environment-key")

    # Without any base URL set, provider resolution returns None so metadata lookups still work.
    assert get_llm_provider(model="inferock/dane") == (
        "dane",
        "inferock",
        "fake-environment-key",
        None,
    )
    assert litellm.supports_function_calling("inferock/dane") is True
    assert litellm.supports_reasoning("inferock/dane") is True
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="inferock/dane", prompt_tokens=100, completion_tokens=50
    )
    assert prompt_cost > 0 and completion_cost > 0

    # Raw Console GATEWAY_BASE_URL (without /v1) normalizes to /v1.
    monkeypatch.setenv("GATEWAY_BASE_URL", "https://tenant-gateway.inferock.example/")
    assert get_llm_provider(model="inferock/dane") == (
        "dane",
        "inferock",
        "fake-environment-key",
        "https://tenant-gateway.inferock.example/v1",
    )

    # Scoped INFEROCK_API_BASE takes precedence over generic GATEWAY_BASE_URL.
    monkeypatch.setenv("INFEROCK_API_BASE", "https://scoped-gateway.inferock.example/v1")
    assert get_llm_provider(model="inferock/dane") == (
        "dane",
        "inferock",
        "fake-environment-key",
        "https://scoped-gateway.inferock.example/v1",
    )

    # Explicit api_base takes precedence over env and normalizes missing /v1.
    assert get_llm_provider(
        model="inferock/dane",
        api_key="fake-explicit-key",
        api_base="https://explicit-gateway.example",
    ) == ("dane", "inferock", "fake-explicit-key", "https://explicit-gateway.example/v1")


@pytest.mark.parametrize("use_base_handler", ["false", "true"])
def test_inferock_errors_clearly_when_base_url_is_missing(
    monkeypatch: pytest.MonkeyPatch, use_base_handler: str
) -> None:
    monkeypatch.delenv("INFEROCK_API_BASE", raising=False)
    monkeypatch.delenv("GATEWAY_BASE_URL", raising=False)
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", use_base_handler)
    with pytest.raises(Exception, match="INFEROCK_API_BASE"):
        litellm.completion(
            model="inferock/dane",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "Hello"}],
        )


def test_inferock_dane_chat_sends_governance_header_drops_unsupported_and_calculates_cost(
    local_model_cost_map: None,
) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://tenant-gateway.inferock.example/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-dane-test",
                "object": "chat.completion",
                "created": 1,
                "model": "dane",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 200,
                    "total_tokens": 1200,
                },
            },
        )
        response: Final = litellm.completion(
            model="inferock/dane",
            api_key="fake-test-key",
            api_base="https://tenant-gateway.inferock.example",
            messages=[{"role": "user", "content": "Reply only OK."}],
            reasoning_effort="low",
            max_completion_tokens=64,
            temperature=0.2,
            top_p=0.9,
            drop_params=True,
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert request.headers["x-governance-provider"] == "inferock-serve"
    assert body["model"] == "dane"
    assert body["reasoning_effort"] == "low"
    assert body["max_completion_tokens"] == 64
    assert "temperature" not in body
    assert "top_p" not in body
    assert response.choices[0].message.content == "OK"
    rates: Final = litellm.model_cost["inferock/dane"]
    expected: Final = 1000 * rates["input_cost_per_token"] + 200 * rates["output_cost_per_token"]
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


def test_inferock_dane_rejects_unsupported_sampling_params_without_drop_params() -> None:
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.completion(
            model="inferock/dane",
            api_key="fake-test-key",
            api_base="https://tenant-gateway.inferock.example/v1",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.5,
            drop_params=False,
        )


def test_inferock_dane_strips_tool_strict_and_forwards_tools() -> None:
    tools: Final = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "strict": False,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    with respx.mock() as upstream:
        route: Final = upstream.post("https://tenant-gateway.inferock.example/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-dane-tool",
                "object": "chat.completion",
                "created": 1,
                "model": "dane",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            },
        )
        response: Final = litellm.completion(
            model="inferock/dane",
            api_key="fake-test-key",
            api_base="https://tenant-gateway.inferock.example",
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            tools=tools,
            tool_choice="required",
            parallel_tool_calls=True,
            drop_params=True,
        )
    body: Final = json.loads(route.calls.last.request.content)
    assert body["tools"][0]["function"]["name"] == "get_weather"
    assert "strict" not in body["tools"][0]["function"]
    assert body["tool_choice"] == "required"
    assert "parallel_tool_calls" not in body
    assert response.choices[0].message.tool_calls[0].function.name == "get_weather"


def test_inferock_dane_trims_to_context_window(local_model_cost_map: None) -> None:
    assert "inferock/dane" in litellm.model_cost
    messages: Final = [{"role": "user", "content": "hello " * 500}]
    budget: Final = 100
    trimmed: Final = trim_messages(
        messages,
        model="inferock/dane",
        trim_ratio=budget / litellm.model_cost["inferock/dane"]["max_input_tokens"],
    )
    assert litellm.token_counter(model="inferock/dane", messages=messages) > budget
    assert litellm.token_counter(model="inferock/dane", messages=trimmed) <= budget


@pytest.mark.parametrize("use_base_handler", ["false", "true"])
def test_inferock_dane_filters_extra_body_in_both_transports(
    monkeypatch: pytest.MonkeyPatch, use_base_handler: str
) -> None:
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", use_base_handler)
    extra_tools: Final = [
        {
            "type": "function",
            "function": {
                "name": "ping",
                "description": "Ping",
                "strict": False,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.completion(
            model="inferock/dane",
            api_key="fake-test-key",
            api_base="https://tenant-gateway.inferock.example",
            messages=[{"role": "user", "content": "Hello"}],
            extra_body={"temperature": 0.7},
            drop_params=False,
        )

    with respx.mock() as upstream:
        route: Final = upstream.post("https://tenant-gateway.inferock.example/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-dane-extra",
                "object": "chat.completion",
                "created": 1,
                "model": "dane",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )
        litellm.completion(
            model="inferock/dane",
            api_key="fake-test-key",
            api_base="https://tenant-gateway.inferock.example",
            messages=[{"role": "user", "content": "Hello"}],
            extra_body={"temperature": 0.7, "tools": extra_tools},
            drop_params=True,
        )
    body: Final = json.loads(route.calls.last.request.content)
    assert "temperature" not in body
    assert body["tools"][0]["function"]["name"] == "ping"
    assert "strict" not in body["tools"][0]["function"]
