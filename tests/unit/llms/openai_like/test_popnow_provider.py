import json
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.utils import trim_messages

MODEL: Final = "popnow/google/gemini-3.7-flash/chat"
UPSTREAM_MODEL: Final = "google/gemini-3.7-flash/chat"


def test_popnow_resolves_environment_and_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POPNOW_API_KEY", "fake-environment-key")
    monkeypatch.delenv("POPNOW_API_BASE", raising=False)
    assert get_llm_provider(model=MODEL) == (
        UPSTREAM_MODEL,
        "popnow",
        "fake-environment-key",
        "https://api.popnow.ai/v1",
    )
    assert get_llm_provider(model=MODEL, api_key="fake-explicit-key", api_base="https://gateway.example/v1") == (
        UPSTREAM_MODEL,
        "popnow",
        "fake-explicit-key",
        "https://gateway.example/v1",
    )


def test_popnow_api_base_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POPNOW_API_KEY", "fake-environment-key")
    monkeypatch.setenv("POPNOW_API_BASE", "https://proxy.example/v1")
    assert get_llm_provider(model=MODEL)[3] == "https://proxy.example/v1"


def test_popnow_chat_keeps_nested_model_id_and_accounts_for_cache(local_model_cost_map: None) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.popnow.ai/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": UPSTREAM_MODEL,
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
            model=MODEL,
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "Say hello"}],
            reasoning_effort="low",
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert body["model"] == UPSTREAM_MODEL
    assert body["reasoning_effort"] == "low"
    assert response.choices[0].message.content == "hello"
    rates: Final = litellm.model_cost[MODEL]
    expected: Final = (
        20 * rates["input_cost_per_token"]
        + 80 * rates["cache_read_input_token_cost"]
        + 5 * rates["output_cost_per_token"]
    )
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


def test_popnow_uses_native_responses_endpoint() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.popnow.ai/v1/responses").respond(
            200,
            json={
                "id": "resp-test",
                "object": "response",
                "created_at": 1,
                "model": UPSTREAM_MODEL,
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
            model=MODEL,
            api_key="fake-test-key",
            input="Say hello",
            store=False,
        )
    assert route.calls.last.request.headers["authorization"] == "Bearer fake-test-key"
    assert json.loads(route.calls.last.request.content)["model"] == UPSTREAM_MODEL
    assert response.output[0].content[0].text == "hello"


def test_popnow_models_trim_to_their_input_budget(local_model_cost_map: None) -> None:
    model_names: Final = tuple(model for model in litellm.model_cost if model.startswith("popnow/"))
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
