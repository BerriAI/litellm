import json
from pathlib import Path

import pytest

import litellm


def test_vynaris_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("VYNARIS_API_KEY", "vynaris-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="vynaris/auto",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "auto"
    assert provider == "vynaris"
    assert api_key == "vynaris-test-key"
    assert api_base == "https://api.vynaris.com/v1"


def test_vynaris_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("VYNARIS_API_KEY", "vynaris-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="vynaris/auto",
        custom_llm_provider=None,
        api_base="https://vynaris.internal.example/v1",
        api_key="vynaris-explicit-key",
    )

    assert provider == "vynaris"
    assert api_key == "vynaris-explicit-key"
    assert api_base == "https://vynaris.internal.example/v1"


def test_vynaris_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    entry = next(p for p in providers if p["litellm_provider"] == "vynaris")

    assert entry["provider_display_name"] == "Vynaris"
    assert {f["key"] for f in entry["credential_fields"]} == {"api_base", "api_key"}
