import json
from typing import Final

import pytest
import respx

import litellm
from litellm.cost_calculator import cost_per_token
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.utils import trim_messages

_API_BASE: Final = "https://api.abliteratedllm.dev/v1"
_MODEL: Final = "abliteratedllm/glm-5.3-flash-abliterated"
ABLITERATEDLLM_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("abliteratedllm/")))
PRIORITY_TIER_MODELS: Final = tuple(
    name for name in ABLITERATEDLLM_MODELS if "input_cost_per_token_priority" in litellm.model_cost[name]
)


def test_abliteratedllm_parametrized_rows_are_present() -> None:
    assert _MODEL in ABLITERATEDLLM_MODELS
    assert PRIORITY_TIER_MODELS


def test_abliteratedllm_resolves_environment_and_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ABLITERATEDLLM_API_KEY", "fake-environment-key")
    assert get_llm_provider(model=_MODEL) == (
        "glm-5.3-flash-abliterated",
        "abliteratedllm",
        "fake-environment-key",
        _API_BASE,
    )
    assert get_llm_provider(model=_MODEL, api_key="fake-explicit-key", api_base="https://gateway.example/v1") == (
        "glm-5.3-flash-abliterated",
        "abliteratedllm",
        "fake-explicit-key",
        "https://gateway.example/v1",
    )


def test_abliteratedllm_chat_forwards_reasoning_and_accounts_for_cache(local_model_cost_map: None) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_API_BASE}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "glm-5.3-flash-abliterated",
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
            model=_MODEL,
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "Say hello"}],
            reasoning_effort="low",
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert body["model"] == "glm-5.3-flash-abliterated"
    assert body["reasoning_effort"] == "low"
    assert response.choices[0].message.content == "hello"
    rates: Final = litellm.model_cost[_MODEL]
    expected: Final = (
        20 * rates["input_cost_per_token"]
        + 80 * rates["cache_read_input_token_cost"]
        + 5 * rates["output_cost_per_token"]
    )
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


def test_abliteratedllm_uses_native_responses_endpoint() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_API_BASE}/responses").respond(
            200,
            json={
                "id": "resp-test",
                "object": "response",
                "created_at": 1,
                "model": "glm-5.3-flash-abliterated",
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
            model=_MODEL,
            api_key="fake-test-key",
            input="Say hello",
            reasoning={"effort": "low"},
            store=False,
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert body["model"] == "glm-5.3-flash-abliterated"
    assert body["reasoning"]["effort"] == "low"
    assert response.output[0].content[0].text == "hello"


def test_abliteratedllm_never_sends_the_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ABLITERATEDLLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key-must-not-leak")
    monkeypatch.setenv("OPENAI_ORGANIZATION", "org-must-not-leak")
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(f"{_API_BASE}/chat/completions").respond(200, json={})
        with pytest.raises(litellm.AuthenticationError, match="ABLITERATEDLLM_API_KEY"):
            _ = litellm.completion(model=_MODEL, messages=[{"role": "user", "content": "hi"}])
    assert not route.called


def test_abliteratedllm_never_sends_the_openai_organization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_ORGANIZATION", "org-must-not-leak")
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_API_BASE}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "glm-5.3-flash-abliterated",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
        _ = litellm.completion(model=_MODEL, api_key="fake-test-key", messages=[{"role": "user", "content": "hi"}])
    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert "openai-organization" not in request.headers


@pytest.mark.parametrize("model", PRIORITY_TIER_MODELS)
def test_abliteratedllm_priority_requests_bill_at_priority_rates(model: str, local_model_cost_map: None) -> None:
    rates: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="abliteratedllm",
        service_tier="priority",
    )
    cached_prompt_cost, _ = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cache_read_input_tokens=1_000_000,
        custom_llm_provider="abliteratedllm",
        service_tier="priority",
    )
    assert prompt_cost == pytest.approx(rates["input_cost_per_token_priority"] * 1_000_000)
    assert completion_cost == pytest.approx(rates["output_cost_per_token_priority"] * 1_000_000)
    assert cached_prompt_cost == pytest.approx(rates["cache_read_input_token_cost_priority"] * 1_000_000)


@pytest.mark.parametrize("model", ABLITERATEDLLM_MODELS)
def test_abliteratedllm_models_trim_to_their_input_budget(model: str, local_model_cost_map: None) -> None:
    messages: Final = [{"role": "user", "content": "hello " * 500}]
    budget: Final = 100
    trimmed: Final = trim_messages(
        messages,
        model=model,
        trim_ratio=budget / litellm.model_cost[model]["max_input_tokens"],
    )
    assert litellm.token_counter(model=model, messages=messages) > budget
    assert litellm.token_counter(model=model, messages=trimmed) <= budget
