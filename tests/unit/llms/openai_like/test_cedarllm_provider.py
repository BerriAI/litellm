import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm


def test_cedarllm_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("CEDARLLM_API_KEY", "cedarllm-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="cedarllm/qwen3.8-27b",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "qwen3.8-27b"
    assert provider == "cedarllm"
    assert api_key == "cedarllm-test-key"
    assert api_base == "https://api.cedarllm.com/v1"


def test_cedarllm_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("CEDARLLM_API_KEY", "cedarllm-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="cedarllm/qwen3-coder-next",
        custom_llm_provider=None,
        api_base="https://cedarllm.internal.example/v1",
        api_key="cedarllm-explicit-key",
    )

    assert provider == "cedarllm"
    assert api_key == "cedarllm-explicit-key"
    assert api_base == "https://cedarllm.internal.example/v1"


CEDARLLM_MODELS = tuple(sorted(name for name in litellm.model_cost if name.startswith("cedarllm/")))


@pytest.mark.parametrize("model", CEDARLLM_MODELS)
def test_cedarllm_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token

    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="cedarllm",
    )
    model_info = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(model_info["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(model_info["output_cost_per_token"] * 1_000_000)
    assert model_info["output_cost_per_token"] > model_info["input_cost_per_token"] > 0
    assert model_info["max_tokens"] == model_info["max_output_tokens"] <= model_info["max_input_tokens"]
    assert model_info["litellm_provider"] == "cedarllm"
    assert model_info["mode"] == "chat"
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_response_schema"] is True
    assert litellm.supports_vision(model) is model_info["supports_vision"]


def test_cedarllm_backup_registry_mirrors_cost_map():
    package_root = Path(litellm.__file__).parent
    cost_map = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    entries = {name: entry for name, entry in cost_map.items() if name.startswith("cedarllm/")}

    assert tuple(sorted(entries)) == CEDARLLM_MODELS
    assert entries
    assert entries == {name: backup[name] for name in entries}


def test_cedarllm_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    cedarllm = next(provider for provider in providers if provider["litellm_provider"] == "cedarllm")

    assert cedarllm["provider"] == "CEDARLLM"
    assert cedarllm["provider_display_name"] == "CedarLLM"
    assert cedarllm["default_model_placeholder"] == "cedarllm/qwen3.8-27b"
    assert {field["key"]: field["required"] for field in cedarllm["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_cedarllm_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.cedarllm.com/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-cedarllm",
                "object": "chat.completion",
                "created": 1_794_199_800,
                "model": "qwen3.8-27b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from CedarLLM"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8},
            },
        )
        response: Final = litellm.completion(
            model="cedarllm/qwen3.8-27b",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=32,
            api_key="cedarllm-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer cedarllm-test-key"
    assert body["model"] == "qwen3.8-27b"
    assert body["max_tokens"] == 32
    assert "max_completion_tokens" not in body
    assert response.choices[0].message.content == "Hello from CedarLLM"
