import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.utils import trim_messages

AUGURE_API_BASE: Final = "https://api.augureai.ca/v1"


def test_augure_is_available_in_add_model_form() -> None:
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    augure: Final = next(provider for provider in providers if provider["litellm_provider"] == "augure")
    assert augure["provider"] == "AUGURE"
    assert augure["provider_display_name"] == "Augure"
    assert augure["default_model_placeholder"] == "augure/ossington-5"
    assert {field["key"]: field["required"] for field in augure["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_augure_resolves_environment_and_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUGURE_API_KEY", "fake-environment-key")
    assert get_llm_provider(model="augure/tofino-3") == (
        "tofino-3",
        "augure",
        "fake-environment-key",
        AUGURE_API_BASE,
    )
    assert get_llm_provider(
        model="augure/tofino-3", api_key="fake-explicit-key", api_base="https://gateway.example/v1"
    ) == ("tofino-3", "augure", "fake-explicit-key", "https://gateway.example/v1")


def test_augure_api_base_env_overrides_the_default_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUGURE_API_KEY", "fake-environment-key")
    monkeypatch.setenv("AUGURE_API_BASE", "https://augure.internal.example/v1")
    assert get_llm_provider(model="augure/ossington-5")[3] == "https://augure.internal.example/v1"


def test_augure_chat_forwards_reasoning_effort_and_accounts_for_cached_input(local_model_cost_map: None) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{AUGURE_API_BASE}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "ossington-5",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "total_tokens": 105,
                    "prompt_tokens_details": {"cached_tokens": 80},
                },
            },
        )
        response: Final = litellm.completion(
            model="augure/ossington-5",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            reasoning_effort="low",
            max_completion_tokens=64,
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert body["model"] == "ossington-5"
    assert body["reasoning_effort"] == "low"
    assert body["max_tokens"] == 64
    assert "max_completion_tokens" not in body
    assert response.choices[0].message.content == "ok"
    rates: Final = litellm.model_cost["augure/ossington-5"]
    expected: Final = (
        20 * rates["input_cost_per_token"]
        + 80 * rates["cache_read_input_token_cost"]
        + 5 * rates["output_cost_per_token"]
    )
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


def test_augure_cost_map_rows_only_list_registered_endpoints(local_model_cost_map: None) -> None:
    provider: Final = JSONProviderRegistry.get("augure")
    assert provider is not None
    rows: Final = {name: info for name, info in litellm.model_cost.items() if info.get("litellm_provider") == "augure"}
    assert rows
    for name, info in rows.items():
        assert name.startswith("augure/")
        assert set(info["supported_endpoints"]) <= set(provider.supported_endpoints), name


def test_augure_models_trim_to_their_input_budget(local_model_cost_map: None) -> None:
    model_names: Final = tuple(model for model in litellm.model_cost if model.startswith("augure/"))
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
