import json
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.utils import trim_messages


def test_outrun_resolves_environment_and_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTRUN_API_KEY", "fake-environment-key")
    assert get_llm_provider(model="outrun/claude-sonnet-5-5") == (
        "claude-sonnet-5-5",
        "outrun",
        "fake-environment-key",
        "https://api.outrun.to/v1",
    )
    assert get_llm_provider(
        model="outrun/claude-sonnet-5-5", api_key="fake-explicit-key", api_base="https://gateway.example/v1"
    ) == ("claude-sonnet-5-5", "outrun", "fake-explicit-key", "https://gateway.example/v1")


def test_outrun_chat_forwards_reasoning_and_accounts_for_cache(local_model_cost_map: None) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.outrun.to/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "claude-sonnet-5-5",
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
            model="outrun/claude-sonnet-5-5",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "Say hello"}],
            reasoning_effort="low",
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert body["model"] == "claude-sonnet-5-5"
    assert body["reasoning_effort"] == "low"
    assert response.choices[0].message.content == "hello"
    rates: Final = litellm.model_cost["outrun/claude-sonnet-5-5"]
    expected: Final = (
        20 * rates["input_cost_per_token"]
        + 80 * rates["cache_read_input_token_cost"]
        + 5 * rates["output_cost_per_token"]
    )
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


def test_outrun_uses_native_responses_endpoint() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.outrun.to/v1/responses").respond(
            200,
            json={
                "id": "resp-test",
                "object": "response",
                "created_at": 1,
                "model": "claude-sonnet-5-5",
                "status": "completed",
                "output": [
                    {
                        "id": "msg-test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="outrun/claude-sonnet-5-5",
            api_key="fake-test-key",
            input="Say hello",
            store=False,
        )
    assert route.calls.last.request.headers["authorization"] == "Bearer fake-test-key"
    assert json.loads(route.calls.last.request.content)["model"] == "claude-sonnet-5-5"
    assert response.output[0].content[0].text == "hello"


def test_outrun_models_trim_to_their_input_budget(local_model_cost_map: None) -> None:
    model_names: Final = tuple(model for model in litellm.model_cost if model.startswith("outrun/"))
    assert model_names
    messages: Final = [{"role": "user", "content": "hello " * 500}]
    budget: Final = 100
    for model in model_names:
        trimmed: Final = trim_messages(
            messages,
            model=model,
            trim_ratio=budget / litellm.model_cost[model]["max_input_tokens"],
        )
        assert litellm.token_counter(model=model, messages=messages) > budget
        assert litellm.token_counter(model=model, messages=trimmed) <= budget
