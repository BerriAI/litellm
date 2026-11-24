import json
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.proxy.auth.model_checks import get_provider_models

MODEL: Final = "fortytwo/qwen/qwen3.8-27b"
API_BASE: Final = "https://api.fortytwo.network/v1"
MESSAGES: Final = [{"role": "user", "content": "hello"}]
USAGE: Final = {
    "prompt_tokens": 1000,
    "completion_tokens": 200,
    "total_tokens": 1200,
    "prompt_tokens_details": {"cached_tokens": 400},
}


@pytest.fixture(autouse=True)
def fortytwo_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("FORTYTWO_API_KEY", "fortytwo-unit-test-key")
    monkeypatch.delenv("FORTYTWO_API_BASE", raising=False)
    yield


def _completion_body() -> dict[str, object]:
    return {
        "id": "chatcmpl-fortytwo-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "qwen/qwen3.8-27b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "answer", "reasoning": "reasoning text"},
                "finish_reason": "stop",
            }
        ],
        "usage": USAGE,
    }


@pytest.mark.parametrize("override", [None, "https://fortytwo-unit.invalid/v1"])
def test_completion_uses_fortytwo_environment_and_strips_only_provider_prefix(
    monkeypatch: pytest.MonkeyPatch, override: str | None
) -> None:
    if override is not None:
        monkeypatch.setenv("FORTYTWO_API_BASE", override)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-unit-key")
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{override or API_BASE}/chat/completions").respond(200, json=_completion_body())
        response: Final = litellm.completion(model=MODEL, messages=MESSAGES, num_retries=0)

    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer fortytwo-unit-test-key"
    assert json.loads(request.content) == {"model": "qwen/qwen3.8-27b", "messages": MESSAGES}
    assert response.choices[0].message.content == "answer"
    assert response.choices[0].message.reasoning_content == "reasoning text"


def test_fortytwo_key_in_generic_openai_environment_is_used_when_own_key_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FORTYTWO_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "fortytwo-key-in-openai-env")
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{API_BASE}/chat/completions").respond(200, json=_completion_body())
        response: Final = litellm.completion(model=MODEL, messages=MESSAGES, num_retries=0)

    assert len(route.calls) == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer fortytwo-key-in-openai-env"
    assert json.loads(route.calls.last.request.content) == {"model": "qwen/qwen3.8-27b", "messages": MESSAGES}
    assert response.choices[0].message.content == "answer"


def test_rejected_key_preserves_authentication_failure() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{API_BASE}/chat/completions").respond(
            401,
            json={"error": {"message": "Invalid API key", "type": "authentication_error", "code": "invalid_api_key"}},
        )
        with pytest.raises(litellm.AuthenticationError):
            litellm.completion(model=MODEL, messages=MESSAGES, num_retries=0)
    assert len(route.calls) == 1


@pytest.mark.parametrize("limit_name", ["max_tokens", "max_completion_tokens"])
def test_output_limit_reaches_upstream(limit_name: str) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{API_BASE}/chat/completions").respond(200, json=_completion_body())
        litellm.completion(model=MODEL, messages=MESSAGES, num_retries=0, **{limit_name: 17})
    assert json.loads(route.calls.last.request.content) == {
        "model": "qwen/qwen3.8-27b",
        "messages": MESSAGES,
        limit_name: 17,
    }


def test_stream_preserves_reasoning_answer_and_final_usage() -> None:
    events: Final = (
        {"choices": [{"index": 0, "delta": {"role": "assistant", "reasoning": "think"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": USAGE},
    )
    wire: Final = (
        "".join(
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-fortytwo-test",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": "qwen/qwen3.8-27b",
                    **event,
                }
            )
            + "\n\n"
            for event in events
        )
        + "data: [DONE]\n\n"
    )
    with respx.mock() as upstream:
        upstream.post(f"{API_BASE}/chat/completions").respond(
            200, text=wire, headers={"content-type": "text/event-stream"}
        )
        chunks: Final = tuple(
            litellm.completion(
                model=MODEL, messages=MESSAGES, stream=True, stream_options={"include_usage": True}, num_retries=0
            )
        )
    assert (
        "".join(getattr(chunk.choices[0].delta, "reasoning_content", "") or "" for chunk in chunks if chunk.choices)
        == "think"
    )
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "answer"
    assert chunks[-1].usage.prompt_tokens == USAGE["prompt_tokens"]
    assert chunks[-1].usage.completion_tokens == USAGE["completion_tokens"]


def test_tool_call_request_and_response_preserve_function_contract() -> None:
    tool: Final = {
        "type": "function",
        "function": {"name": "city", "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}},
    }
    tool_call: Final = {
        "id": "call-city",
        "type": "function",
        "function": {"name": "city", "arguments": '{"name":"London"}'},
    }
    body: Final = {
        **_completion_body(),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                "finish_reason": "tool_calls",
            }
        ],
    }
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{API_BASE}/chat/completions").respond(200, json=body)
        response: Final = litellm.completion(
            model=MODEL, messages=MESSAGES, tools=[tool], tool_choice="required", num_retries=0
        )
    assert json.loads(route.calls.last.request.content) == {
        "model": "qwen/qwen3.8-27b",
        "messages": MESSAGES,
        "tools": [tool],
        "tool_choice": "required",
    }
    assert response.choices[0].message.tool_calls[0].model_dump() == tool_call


def test_response_cost_uses_cached_and_uncached_catalogue_rates() -> None:
    with respx.mock() as upstream:
        upstream.post(f"{API_BASE}/chat/completions").respond(200, json=_completion_body())
        response: Final = litellm.completion(model=MODEL, messages=MESSAGES, num_retries=0)
    rates: Final = litellm.model_cost[MODEL]
    expected: Final = (
        600 * rates["input_cost_per_token"]
        + 400 * rates["cache_read_input_token_cost"]
        + 200 * rates["output_cost_per_token"]
    )
    assert response._hidden_params["response_cost"] == pytest.approx(expected)


def test_responses_call_uses_chat_bridge_and_keeps_responses_shape() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{API_BASE}/chat/completions").respond(200, json=_completion_body())
        response: Final = litellm.responses(model=MODEL, input="hello", num_retries=0)
    assert json.loads(route.calls.last.request.content)["model"] == "qwen/qwen3.8-27b"
    assert response.output[-1].content[0].text == "answer"
    assert response.usage.input_tokens == USAGE["prompt_tokens"]
    assert response.usage.output_tokens == USAGE["completion_tokens"]


def test_messages_call_uses_chat_bridge_and_keeps_messages_shape() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{API_BASE}/chat/completions").respond(200, json=_completion_body())
        response: Final = litellm.anthropic.messages.create(model=MODEL, messages=MESSAGES, max_tokens=17)
    assert json.loads(route.calls.last.request.content)["model"] == "qwen/qwen3.8-27b"
    assert response["content"][-1] == {"type": "text", "text": "answer"}
    assert response["usage"]["input_tokens"] + response["usage"]["cache_read_input_tokens"] == USAGE["prompt_tokens"]
    assert response["usage"]["output_tokens"] == USAGE["completion_tokens"]


def test_discovery_and_bundled_catalogue_match_runtime_model() -> None:
    root: Final = Path(__file__).resolve().parents[4]
    cost_map: Final = json.loads((root / "model_prices_and_context_window.json").read_text())
    bundled: Final = json.loads((root / "litellm/model_prices_and_context_window_backup.json").read_text())
    assert cost_map[MODEL] == bundled[MODEL] == litellm.model_cost[MODEL]
    assert MODEL in get_provider_models(provider="fortytwo")
    assert MODEL in litellm.model_list
    assert MODEL in litellm.models_by_provider["fortytwo"]
    assert get_llm_provider("publicai/a-test-model")[0:2] == ("a-test-model", "publicai")


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_reasoning_effort_is_forwarded_without_remapping(effort: str) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{API_BASE}/chat/completions").respond(200, json=_completion_body())
        litellm.completion(model=MODEL, messages=MESSAGES, reasoning_effort=effort, num_retries=0)
    assert json.loads(route.calls.last.request.content)["reasoning_effort"] == effort
