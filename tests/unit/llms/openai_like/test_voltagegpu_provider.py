import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm

VOLTAGEGPU_API_BASE: Final = "https://api.voltagegpu.com/v1"
VOLTAGEGPU_MODEL: Final = "Qwen/Qwen3.8-27B-TEE"


def test_voltagegpu_json_registry():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert JSONProviderRegistry.exists("voltagegpu")
    config = JSONProviderRegistry.get("voltagegpu")
    assert config is not None
    assert config.base_url == VOLTAGEGPU_API_BASE
    assert config.api_key_env == "VOLTAGEGPU_API_KEY"
    assert config.api_base_env == "VOLTAGEGPU_API_BASE"
    assert config.param_mappings.get("max_completion_tokens") == "max_tokens"


def test_voltagegpu_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    """Model ids on VoltageGPU contain a `/` (org/model-TEE); only the first segment is the provider."""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("VOLTAGEGPU_API_KEY", "voltagegpu-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model=f"voltagegpu/{VOLTAGEGPU_MODEL}",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == VOLTAGEGPU_MODEL
    assert provider == "voltagegpu"
    assert api_key == "voltagegpu-test-key"
    assert api_base == VOLTAGEGPU_API_BASE


def test_voltagegpu_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("VOLTAGEGPU_API_KEY", "voltagegpu-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model=f"voltagegpu/{VOLTAGEGPU_MODEL}",
        custom_llm_provider=None,
        api_base="https://voltagegpu.internal.example/v1",
        api_key="voltagegpu-explicit-key",
    )

    assert provider == "voltagegpu"
    assert api_key == "voltagegpu-explicit-key"
    assert api_base == "https://voltagegpu.internal.example/v1"


def test_voltagegpu_max_completion_tokens_is_mapped_to_max_tokens():
    """VoltageGPU documents `max_tokens` only, so `max_completion_tokens` is renamed on the way out."""
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("voltagegpu"))()

    result = config.map_openai_params(
        {"max_completion_tokens": 64, "temperature": 0.2}, {}, VOLTAGEGPU_MODEL, False
    )

    assert result["max_tokens"] == 64
    assert "max_completion_tokens" not in result
    assert result["temperature"] == 0.2


def test_voltagegpu_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    voltagegpu = next(provider for provider in providers if provider["litellm_provider"] == "voltagegpu")

    assert voltagegpu["provider"] == "VOLTAGEGPU"
    assert voltagegpu["provider_display_name"] == "VoltageGPU"
    assert voltagegpu["default_model_placeholder"] == f"voltagegpu/{VOLTAGEGPU_MODEL}"
    assert {field["key"]: field["required"] for field in voltagegpu["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_voltagegpu_supported_endpoints():
    matrix_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    providers = json.loads(matrix_path.read_text())["providers"]

    assert providers["voltagegpu"]["endpoints"] == {
        "chat_completions": True,
        "messages": True,
        "responses": True,
        "embeddings": False,
        "image_generations": False,
        "audio_transcriptions": False,
        "audio_speech": False,
        "moderations": False,
        "batches": False,
        "rerank": False,
        "a2a": False,
        "interactions": False,
    }


def test_voltagegpu_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{VOLTAGEGPU_API_BASE}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_voltagegpu",
                "object": "chat.completion",
                "created": 1_792_752_740,
                "model": VOLTAGEGPU_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from VoltageGPU"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8},
            },
        )
        response: Final = litellm.completion(
            model=f"voltagegpu/{VOLTAGEGPU_MODEL}",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="voltagegpu-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{VOLTAGEGPU_API_BASE}/chat/completions"
    assert request.headers["authorization"] == "Bearer voltagegpu-test-key"
    assert body["model"] == VOLTAGEGPU_MODEL
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from VoltageGPU"
