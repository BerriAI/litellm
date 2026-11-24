import json
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_MODEL: Final = "llmtr/llmtr/qwen3-8-27b"


def test_llmtr_resolves_environment_and_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMTR_API_KEY", "fake-environment-key")
    monkeypatch.delenv("LLMTR_API_BASE", raising=False)
    assert get_llm_provider(model=_MODEL) == (
        "llmtr/qwen3-8-27b",
        "llmtr",
        "fake-environment-key",
        "https://llmtr.com/v1",
    )
    assert get_llm_provider(model=_MODEL, api_key="fake-explicit-key", api_base="https://gateway.example/v1") == (
        "llmtr/qwen3-8-27b",
        "llmtr",
        "fake-explicit-key",
        "https://gateway.example/v1",
    )


def test_llmtr_chat_keeps_the_upstream_model_namespace_and_accounts_for_cache(local_model_cost_map: None) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://llmtr.com/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "llmtr/qwen3-8-27b",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "merhaba"}, "finish_reason": "stop"}
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
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert body["model"] == "llmtr/qwen3-8-27b"
    assert response.choices[0].message.content == "merhaba"
    rates: Final = litellm.model_cost[_MODEL]
    expected: Final = (
        20 * rates["input_cost_per_token"]
        + 80 * rates["cache_read_input_token_cost"]
        + 5 * rates["output_cost_per_token"]
    )
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


def test_llmtr_catalog_models_route_to_llmtr(local_model_cost_map: None) -> None:
    model_names: Final = tuple(model for model in litellm.model_cost if model.startswith("llmtr/"))
    assert model_names
    for model in model_names:
        upstream_model, provider, _, api_base = get_llm_provider(model=model, api_key="fake-test-key")
        assert provider == "llmtr" == litellm.model_cost[model]["litellm_provider"]
        assert f"llmtr/{upstream_model}" == model
        assert api_base == "https://llmtr.com/v1"
