import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_MODEL_ID: Final = "thinkingmachines/Inkling-2-Small:peft:262144"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_tinker",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": _MODEL_ID,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Tinker", "reasoning_content": "Say hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "total_tokens": 1100,
        "prompt_tokens_details": {"cached_tokens": 400},
    },
}
TINKER_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("tinker/")))


def test_tinker_model_resolves_to_tinker_endpoint_and_env_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TINKER_API_KEY", "tinker-env-key")
    monkeypatch.delenv("TINKER_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(model=f"tinker/{_MODEL_ID}")

    assert (model, provider, api_key, api_base) == (_MODEL_ID, "tinker", "tinker-env-key", _DEFAULT_API_BASE)


def test_tinker_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TINKER_API_KEY", "tinker-env-key")
    monkeypatch.setenv("TINKER_API_BASE", "https://tinker.env.example/v1")

    _, provider, _, api_base = get_llm_provider(model=f"tinker/{_MODEL_ID}")

    assert (provider, api_base) == ("tinker", "https://tinker.env.example/v1")


def test_tinker_explicit_credentials_win_over_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TINKER_API_KEY", "tinker-env-key")
    monkeypatch.setenv("TINKER_API_BASE", "https://tinker.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model=f"tinker/{_MODEL_ID}",
        api_base="https://tinker.internal.example/v1",
        api_key="tinker-explicit-key",
    )

    assert (provider, api_key, api_base) == ("tinker", "tinker-explicit-key", "https://tinker.internal.example/v1")


def test_tinker_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TINKER_API_KEY", "tinker-env-key")

    model, provider, api_key, _ = get_llm_provider(model=_MODEL_ID, api_base=_DEFAULT_API_BASE)

    assert (model, provider, api_key) == (_MODEL_ID, "tinker", "tinker-env-key")


def test_tinker_chat_completion_sends_bearer_key_and_full_model_id_and_prices_the_response():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model=f"tinker/{_MODEL_ID}",
            messages=[{"role": "user", "content": "Say hello"}],
            reasoning_effort="low",
            api_key="tinker-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    entry: Final = litellm.model_cost[f"tinker/{_MODEL_ID}"]
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer tinker-test-key"
    assert body == {
        "model": _MODEL_ID,
        "messages": [{"role": "user", "content": "Say hello"}],
        "reasoning_effort": "low",
    }
    assert response.choices[0].message.content == "Hello from Tinker"
    assert response.choices[0].message.reasoning_content == "Say hi"
    assert litellm.completion_cost(completion_response=response) == pytest.approx(
        600 * entry["input_cost_per_token"]
        + 400 * entry["cache_read_input_token_cost"]
        + 100 * entry["output_cost_per_token"]
    )


def test_tinker_serverless_models_are_listed_for_the_provider():
    assert len(TINKER_MODELS) > 0
    assert set(TINKER_MODELS) == set(litellm.models_by_provider["tinker"])
    assert all(litellm.get_model_info(model)["litellm_provider"] == "tinker" for model in TINKER_MODELS)


def test_tinker_default_model_in_add_model_form_is_a_priced_model():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    tinker: Final = next(p for p in json.loads(fields_path.read_text()) if p["litellm_provider"] == "tinker")

    assert tinker["default_model_placeholder"] in TINKER_MODELS
    assert {field["key"]: field["required"] for field in tinker["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }
