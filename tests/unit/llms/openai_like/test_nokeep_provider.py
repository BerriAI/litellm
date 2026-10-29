import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.proxy.auth.model_checks import get_provider_models
from litellm.utils import trim_messages

NOKEEP_API_BASE: Final = "https://nokeep.ai/v1"

NOKEEP_MODELS: Final = (
    "deepseek-v4.1-flash-uncensored",
    "deepseek-v4.1-flash-1017-uncensored",
    "glm-5.3-flash-uncensored",
)

WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def _chat_completion(model: str, usage: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "id": "chatcmpl-nokeep",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": dict(usage),
    }


def test_nokeep_key_and_base_come_from_nokeep_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOKEEP_API_KEY", "fake-environment-key")
    monkeypatch.delenv("NOKEEP_API_BASE", raising=False)

    assert get_llm_provider(model="nokeep/glm-5.3-flash-uncensored") == (
        "glm-5.3-flash-uncensored",
        "nokeep",
        "fake-environment-key",
        NOKEEP_API_BASE,
    )

    monkeypatch.setenv("NOKEEP_API_BASE", "https://gateway.example/v1")

    assert get_llm_provider(model="nokeep/glm-5.3-flash-uncensored")[3] == "https://gateway.example/v1"
    assert get_llm_provider(
        model="nokeep/glm-5.3-flash-uncensored", api_key="fake-explicit-key", api_base="https://other.example/v1"
    ) == ("glm-5.3-flash-uncensored", "nokeep", "fake-explicit-key", "https://other.example/v1")


def test_nokeep_chat_strips_the_prefix_and_forwards_tools_and_reasoning_effort() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{NOKEEP_API_BASE}/chat/completions").respond(
            200,
            json=_chat_completion(
                "deepseek-v4.1-flash-uncensored", {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
            ),
        )
        response: Final = litellm.completion(
            model="nokeep/deepseek-v4.1-flash-uncensored",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "weather in Paris?"}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            reasoning_effort="low",
        )

    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert json.loads(request.content) == {
        "model": "deepseek-v4.1-flash-uncensored",
        "messages": [{"role": "user", "content": "weather in Paris?"}],
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
        "reasoning_effort": "low",
    }
    assert response.choices[0].message.content == "hello"


def test_nokeep_streaming_reports_usage_from_the_final_chunk() -> None:
    chunks: Final = (
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "hel"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}},
    )
    envelope: Final = {"id": "chatcmpl-nokeep", "object": "chat.completion.chunk", "created": 1}
    events: Final = tuple(
        f"data: {json.dumps({**envelope, 'model': 'glm-5.3-flash-uncensored', **chunk})}\n\n" for chunk in chunks
    )
    body: Final = "".join(events) + "data: [DONE]\n\n"

    with respx.mock() as upstream:
        route: Final = upstream.post(f"{NOKEEP_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)
        )
        stream: Final = litellm.completion(
            model="nokeep/glm-5.3-flash-uncensored",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        received: Final = tuple(stream)

    sent: Final = json.loads(route.calls.last.request.content)
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    assert "".join(chunk.choices[0].delta.content or "" for chunk in received if chunk.choices) == "hello"
    usage: Final = next(chunk.usage for chunk in received if getattr(chunk, "usage", None))
    assert (usage.prompt_tokens, usage.completion_tokens) == (7, 2)


def test_nokeep_responses_requests_are_bridged_to_chat_completions() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{NOKEEP_API_BASE}/chat/completions").respond(
            200,
            json=_chat_completion(
                "glm-5.3-flash-uncensored", {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
            ),
        )
        response: Final = litellm.responses(
            model="nokeep/glm-5.3-flash-uncensored",
            api_key="fake-test-key",
            input="Say hello",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["model"] == "glm-5.3-flash-uncensored"
    assert body["messages"][-1]["role"] == "user"
    assert response.output[0].content[0].text == "hello"


@pytest.mark.parametrize("model", NOKEEP_MODELS)
def test_nokeep_completion_is_charged_at_the_cost_map_rates(model: str) -> None:
    usage: Final = {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "total_tokens": 1200,
        "prompt_tokens_details": {"cached_tokens": 400},
    }
    with respx.mock() as upstream:
        upstream.post(f"{NOKEEP_API_BASE}/chat/completions").respond(200, json=_chat_completion(model, usage))
        response: Final = litellm.completion(
            model=f"nokeep/{model}",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "hello"}],
        )

    rates: Final = litellm.model_cost[f"nokeep/{model}"]
    expected_cost: Final = (
        600 * rates["input_cost_per_token"]
        + 400 * rates["cache_read_input_token_cost"]
        + 200 * rates["output_cost_per_token"]
    )
    assert expected_cost > 0
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected_cost)


@pytest.mark.parametrize(
    ("model", "vision", "response_schema"),
    [
        ("deepseek-v4.1-flash-uncensored", True, True),
        ("deepseek-v4.1-flash-1017-uncensored", True, True),
        ("glm-5.3-flash-uncensored", False, False),
    ],
)
def test_nokeep_model_info_reports_each_model_capabilities(model: str, vision: bool, response_schema: bool) -> None:
    info: Final = litellm.get_model_info(f"nokeep/{model}")

    assert info["litellm_provider"] == "nokeep"
    assert info["max_output_tokens"] <= info["max_input_tokens"]
    assert litellm.supports_function_calling(f"nokeep/{model}") is True
    assert litellm.supports_reasoning(f"nokeep/{model}") is True
    assert litellm.supports_vision(f"nokeep/{model}") is vision
    assert litellm.supports_response_schema(f"nokeep/{model}") is response_schema


@pytest.mark.parametrize("model", NOKEEP_MODELS)
def test_trim_messages_fits_nokeep_history_into_the_model_context_window(model: str) -> None:
    budget: Final = 20
    trim_ratio: Final = budget / litellm.model_cost[f"nokeep/{model}"]["max_input_tokens"]
    messages: Final = [{"role": "user", "content": "lorem ipsum " * 300}]

    trimmed: Final = trim_messages(messages, model=f"nokeep/{model}", trim_ratio=trim_ratio)

    assert litellm.token_counter(model=f"nokeep/{model}", messages=messages) > budget
    assert litellm.token_counter(model=f"nokeep/{model}", messages=trimmed) <= budget


def test_nokeep_wildcard_lists_every_nokeep_catalog_model() -> None:
    catalog_models: Final = sorted(
        key for key, row in litellm.model_cost.items() if row.get("litellm_provider") == "nokeep"
    )

    assert catalog_models == sorted(f"nokeep/{model}" for model in NOKEEP_MODELS)
    assert sorted(get_provider_models("nokeep") or []) == catalog_models
