import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm


def test_cheapaimodel_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("CHEAPAIMODEL_API_KEY", "cheapaimodel-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="cheapaimodel/deepseek-v4-1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek-v4-1-flash"
    assert provider == "cheapaimodel"
    assert api_key == "cheapaimodel-test-key"
    assert api_base == "https://api.cheapaimodel.ai/v1"


def test_cheapaimodel_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("CHEAPAIMODEL_API_KEY", "cheapaimodel-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="cheapaimodel/kimi-k3",
        custom_llm_provider=None,
        api_base="https://cheapaimodel.internal.example/v1",
        api_key="cheapaimodel-explicit-key",
    )

    assert provider == "cheapaimodel"
    assert api_key == "cheapaimodel-explicit-key"
    assert api_base == "https://cheapaimodel.internal.example/v1"


CHEAPAIMODEL_MODELS = tuple(sorted(name for name in litellm.model_cost if name.startswith("cheapaimodel/")))


def test_cheapaimodel_models_registered():
    assert CHEAPAIMODEL_MODELS == (
        "cheapaimodel/deepseek-v4-1-flash",
        "cheapaimodel/glm-5.3",
        "cheapaimodel/glm-5.3-flash",
        "cheapaimodel/grok-4.7",
        "cheapaimodel/kimi-k3",
    )


@pytest.mark.parametrize("model", CHEAPAIMODEL_MODELS)
def test_cheapaimodel_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token

    model_info = litellm.get_model_info(model)

    assert 0 < model_info["cache_read_input_token_cost"] < model_info["input_cost_per_token"]
    assert model_info["output_cost_per_token"] > 0
    assert 0 < model_info["max_output_tokens"] <= model_info["max_input_tokens"]
    assert model_info["litellm_provider"] == "cheapaimodel"
    assert model_info["mode"] == "chat"
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_reasoning"] is True
    assert model_info["supports_response_schema"] is True
    assert litellm.supports_vision(model) is model_info["supports_vision"]

    if model == "cheapaimodel/deepseek-v4-1-flash":
        return  # time-of-day pricing, covered below
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="cheapaimodel",
    )
    assert prompt_cost == pytest.approx(model_info["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(model_info["output_cost_per_token"] * 1_000_000)


def test_cheapaimodel_deepseek_has_off_peak_pricing():
    entry = litellm.model_cost["cheapaimodel/deepseek-v4-1-flash"]
    off_peak = entry["off_peak_pricing"]

    # Off-peak is half the peak rate, as on DeepSeek's own API.
    assert off_peak["input_cost_per_token"] == pytest.approx(entry["input_cost_per_token"] / 2)
    assert off_peak["output_cost_per_token"] == pytest.approx(entry["output_cost_per_token"] / 2)
    assert off_peak["cache_read_input_token_cost"] == pytest.approx(entry["cache_read_input_token_cost"] / 2)
    assert off_peak["windows"] == litellm.model_cost["deepseek/deepseek-v4-flash"]["off_peak_pricing"]["windows"]


def test_cheapaimodel_backup_registry_mirrors_cost_map():
    package_root = Path(litellm.__file__).parent
    cost_map = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    entries = {name: entry for name, entry in cost_map.items() if name.startswith("cheapaimodel/")}

    assert tuple(sorted(entries)) == CHEAPAIMODEL_MODELS
    assert entries == {name: backup[name] for name in entries}


def test_cheapaimodel_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    entry = next(provider for provider in providers if provider["litellm_provider"] == "cheapaimodel")

    assert entry["provider"] == "CHEAPAIMODEL"
    assert entry["provider_display_name"] == "cheapaimodel"
    assert entry["default_model_placeholder"] == "cheapaimodel/deepseek-v4-1-flash"
    assert {field["key"]: field["required"] for field in entry["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_cheapaimodel_supported_endpoints():
    matrix_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    endpoints = json.loads(matrix_path.read_text())["providers"]["cheapaimodel"]["endpoints"]

    assert endpoints["chat_completions"] is True
    assert [name for name, supported in endpoints.items() if supported] == ["chat_completions"]


def test_cheapaimodel_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.cheapaimodel.ai/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-cheapaimodel",
                "object": "chat.completion",
                "created": 1_793_000_000,
                "model": "deepseek-v4-1-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello from cheapaimodel",
                            "reasoning_content": "Say hello.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
            },
        )
        response: Final = litellm.completion(
            model="cheapaimodel/deepseek-v4-1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="cheapaimodel-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.cheapaimodel.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer cheapaimodel-test-key"
    assert body["model"] == "deepseek-v4-1-flash"
    assert response.choices[0].message.content == "Hello from cheapaimodel"
    assert response.choices[0].message.reasoning_content == "Say hello."
