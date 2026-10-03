import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm


def test_alpha_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("ALPHA_API_KEY", "alpha-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="alpha/Qwen/Qwen3.8-27B-FP8",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "Qwen/Qwen3.8-27B-FP8"
    assert provider == "alpha"
    assert api_key == "alpha-test-key"
    assert api_base == "https://alpha.sh/v1"


def test_alpha_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("ALPHA_API_KEY", "alpha-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="alpha/Qwen/Qwen3.8-27B-FP8",
        custom_llm_provider=None,
        api_base="https://alpha.internal.example/v1",
        api_key="alpha-explicit-key",
    )

    assert provider == "alpha"
    assert api_key == "alpha-explicit-key"
    assert api_base == "https://alpha.internal.example/v1"


ALPHA_MODELS = tuple(sorted(name for name in litellm.model_cost if name.startswith("alpha/")))


@pytest.mark.parametrize("model", ALPHA_MODELS)
def test_alpha_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token

    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="alpha",
    )
    model_info = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(model_info["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(model_info["output_cost_per_token"] * 1_000_000)
    assert model_info["output_cost_per_token"] > model_info["input_cost_per_token"] > 0
    assert model_info["max_tokens"] == model_info["max_output_tokens"] <= model_info["max_input_tokens"]
    assert model_info["litellm_provider"] == "alpha"
    assert model_info["mode"] == "chat"
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_reasoning"] is True
    assert litellm.supports_vision(model) is model_info["supports_vision"]


def test_alpha_backup_registry_mirrors_cost_map():
    package_root = Path(litellm.__file__).parent
    cost_map = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    alpha_entries = {name: entry for name, entry in cost_map.items() if name.startswith("alpha/")}

    assert tuple(sorted(alpha_entries)) == ALPHA_MODELS
    assert alpha_entries
    assert alpha_entries == {name: backup[name] for name in alpha_entries}


def test_alpha_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    alpha = next(provider for provider in providers if provider["litellm_provider"] == "alpha")

    assert alpha["provider"] == "ALPHA"
    assert alpha["provider_display_name"] == "Alpha.sh"
    assert alpha["default_model_placeholder"] == "alpha/Qwen/Qwen3.8-27B-FP8"
    assert {field["key"]: field["required"] for field in alpha["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_alpha_supported_endpoints():
    matrix_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    providers = json.loads(matrix_path.read_text())["providers"]

    assert providers["alpha"]["endpoints"] == {
        "chat_completions": True,
        "messages": False,
        "responses": False,
        "embeddings": False,
        "image_generations": False,
        "audio_transcriptions": False,
        "audio_speech": False,
        "moderations": False,
        "batches": False,
        "rerank": False,
        "a2a": False,
    }


def test_alpha_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://alpha.sh/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-alpha",
                "object": "chat.completion",
                "created": 1_791_050_944,
                "model": "Qwen/Qwen3.8-27B-FP8",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Alpha.sh"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8},
            },
        )
        response: Final = litellm.completion(
            model="alpha/Qwen/Qwen3.8-27B-FP8",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="alpha-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://alpha.sh/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer alpha-test-key"
    assert body["model"] == "Qwen/Qwen3.8-27B-FP8"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Alpha.sh"
