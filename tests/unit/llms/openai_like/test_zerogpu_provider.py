import json
from typing import Final

import httpx
import pytest
from openai import OpenAI

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

ZEROGPU_API_BASE: Final = "https://api.zerogpu.ai/v1"

ZEROGPU_MODELS: Final = (
    "deepseek-v4.1-flash",
    "glm-5.3-flash",
    "gpt-4.1-mini",
    "gpt-5.4-nano",
    "gpt-5.6-luna",
    "gpt-oss-120b",
    "LFM2.5-1.2B-Instruct",
    "llama-3.1-8b-instruct-fast",
    "qwen3-30b-a3b-fp8",
)


WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def _zerogpu_client(requests: list[httpx.Request], usage: dict[str, object]) -> OpenAI:
    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-zerogpu",
                "object": "chat.completion",
                "created": 1234567890,
                "model": json.loads(request.content)["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": usage,
            },
        )

    return OpenAI(
        api_key="zerogpu-test-key",
        base_url=ZEROGPU_API_BASE,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )


def test_prefixed_model_reaches_zerogpu_chat_completions_without_the_prefix():
    requests: Final[list[httpx.Request]] = []
    client: Final = _zerogpu_client(requests, {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})

    with client:
        litellm.completion(
            model="zerogpu/gpt-oss-120b",
            messages=[{"role": "user", "content": "hello"}],
            client=client,
        )

    assert len(requests) == 1
    assert str(requests[0].url) == f"{ZEROGPU_API_BASE}/chat/completions"
    assert json.loads(requests[0].content) == {
        "model": "gpt-oss-120b",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_responses_request_is_bridged_to_zerogpu_chat_completions():
    requests: Final[list[httpx.Request]] = []
    client: Final = _zerogpu_client(requests, {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})

    with client:
        response: Final = litellm.responses(model="zerogpu/gpt-oss-120b", input="hello", client=client)

    assert [str(request.url) for request in requests] == [f"{ZEROGPU_API_BASE}/chat/completions"]
    assert json.loads(requests[0].content) == {
        "model": "gpt-oss-120b",
        "messages": [{"role": "user", "content": "hello"}],
    }
    assert response.output[0].content[0].text == "hi"


def test_anthropic_messages_request_is_bridged_to_zerogpu_chat_completions():
    requests: Final[list[httpx.Request]] = []
    client: Final = _zerogpu_client(requests, {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})

    with client:
        response: Final = litellm.anthropic.messages.create(
            model="zerogpu/gpt-oss-120b",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=10,
            client=client,
        )

    assert [str(request.url) for request in requests] == [f"{ZEROGPU_API_BASE}/chat/completions"]
    assert json.loads(requests[0].content) == {
        "model": "gpt-oss-120b",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
    }
    assert response["content"] == [{"type": "text", "text": "hi"}]


@pytest.mark.parametrize("model", ZEROGPU_MODELS)
def test_zerogpu_completion_is_charged_at_the_cost_map_rates(model: str):
    requests: Final[list[httpx.Request]] = []
    usage: Final = {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "total_tokens": 1200,
        "prompt_tokens_details": {"cached_tokens": 400},
    }
    client: Final = _zerogpu_client(requests, usage)

    with client:
        response: Final = litellm.completion(
            model=f"zerogpu/{model}",
            messages=[{"role": "user", "content": "hello"}],
            client=client,
        )

    rates: Final = litellm.model_cost[f"zerogpu/{model}"]
    cached_rate: Final = rates.get("cache_read_input_token_cost", rates["input_cost_per_token"])
    expected_cost: Final = (
        600 * rates["input_cost_per_token"] + 400 * cached_rate + 200 * rates["output_cost_per_token"]
    )
    assert response._hidden_params["response_cost"] == pytest.approx(expected_cost)
    assert expected_cost > 0


@pytest.mark.parametrize(
    ("model", "forwards_tools", "forwards_reasoning_effort"),
    [("gpt-5.6-luna", True, True), ("gpt-oss-120b", True, False), ("LFM2.5-1.2B-Instruct", False, False)],
)
def test_tools_and_reasoning_effort_reach_zerogpu_only_for_models_that_support_them(
    model: str, forwards_tools: bool, forwards_reasoning_effort: bool
):
    requests: Final[list[httpx.Request]] = []
    client: Final = _zerogpu_client(requests, {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})

    with client:
        litellm.completion(
            model=f"zerogpu/{model}",
            messages=[{"role": "user", "content": "weather in Paris?"}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            reasoning_effort="high",
            drop_params=True,
            client=client,
        )

    tools: Final = {"tools": [WEATHER_TOOL], "tool_choice": "auto"} if forwards_tools else {}
    reasoning: Final = {"reasoning_effort": "high"} if forwards_reasoning_effort else {}
    assert json.loads(requests[0].content) == {
        "model": model,
        "messages": [{"role": "user", "content": "weather in Paris?"}],
        **tools,
        **reasoning,
    }


def test_zerogpu_key_and_base_come_from_zerogpu_env_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROGPU_API_KEY", "zerogpu-env-key")
    monkeypatch.delenv("ZEROGPU_API_BASE", raising=False)

    assert get_llm_provider(model="zerogpu/gpt-oss-120b") == (
        "gpt-oss-120b",
        "zerogpu",
        "zerogpu-env-key",
        ZEROGPU_API_BASE,
    )

    monkeypatch.setenv("ZEROGPU_API_BASE", "https://zerogpu.internal.example/v1")

    assert get_llm_provider(model="zerogpu/gpt-oss-120b")[3] == "https://zerogpu.internal.example/v1"


def test_zerogpu_api_base_alone_selects_the_zerogpu_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROGPU_API_KEY", "zerogpu-env-key")

    assert get_llm_provider(model="gpt-oss-120b", api_base=ZEROGPU_API_BASE) == (
        "gpt-oss-120b",
        "zerogpu",
        "zerogpu-env-key",
        ZEROGPU_API_BASE,
    )
