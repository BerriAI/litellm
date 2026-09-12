import json
from pathlib import Path

import pytest

import litellm


def test_prism_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("PRISM_API_KEY", "prism-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="prism/deepseek-v4-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek-v4-flash"
    assert provider == "prism"
    assert api_key == "prism-test-key"
    assert api_base == "https://api.prisminference.com/v1"


def test_prism_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("PRISM_API_KEY", "prism-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="prism/deepseek-v4-flash",
        custom_llm_provider=None,
        api_base="https://prism.internal.example/v1",
        api_key="prism-explicit-key",
    )

    assert provider == "prism"
    assert api_key == "prism-explicit-key"
    assert api_base == "https://prism.internal.example/v1"


def test_prism_model_cost_and_capabilities():
    from litellm.cost_calculator import cost_per_token

    prompt_cost, completion_cost = cost_per_token(
        model="prism/deepseek-v4-flash",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="prism",
    )
    model_info = litellm.get_model_info("prism/deepseek-v4-flash")

    assert prompt_cost == pytest.approx(0.14)
    assert completion_cost == pytest.approx(0.28)
    assert model_info["cache_read_input_token_cost"] == pytest.approx(7e-08)
    assert model_info["max_input_tokens"] == 1_000_000
    assert model_info["max_output_tokens"] == 393_216
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_native_streaming"] is True
    assert model_info["supports_reasoning"] is True
    assert model_info["supports_response_schema"] is True


def test_prism_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    prism = next(provider for provider in providers if provider["litellm_provider"] == "prism")

    assert prism["provider"] == "PRISM"
    assert prism["provider_display_name"] == "Prism"
    assert prism["default_model_placeholder"] == "prism/deepseek-v4-flash"
    assert {field["key"]: field["required"] for field in prism["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_prism_supported_endpoints():
    matrix_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    providers = json.loads(matrix_path.read_text())["providers"]

    assert providers["prism"]["endpoints"] == {
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
