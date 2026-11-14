import json
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://api.kitani.ai/v1"


def test_kitani_is_a_registered_provider():
    assert litellm.LlmProviders.KITANI.value == "kitani"
    assert "kitani" in litellm.provider_list
    assert "kitani" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_kitani_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KITANI_API_KEY", "kitani-test-key")
    monkeypatch.delenv("KITANI_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="kitani/z-ai/glm-5.3",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "z-ai/glm-5.3"
    assert provider == "kitani"
    assert api_key == "kitani-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_kitani_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KITANI_API_KEY", "kitani-test-key")
    monkeypatch.setenv("KITANI_API_BASE", "https://kitani.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="kitani/z-ai/glm-5.3",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "kitani"
    assert api_base == "https://kitani.example/v1"


def test_kitani_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    entries = json.loads(fields_path.read_text())
    kitani = next(entry for entry in entries if entry["litellm_provider"] == "kitani")
    assert kitani["provider_display_name"] == "Kitani"
    assert kitani["default_model_placeholder"] == "kitani/z-ai/glm-5.3"
