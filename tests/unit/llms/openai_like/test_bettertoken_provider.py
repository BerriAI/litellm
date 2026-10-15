import json
from pathlib import Path

import pytest

import litellm


def test_bettertoken_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("BETTERTOKEN_API_KEY", "bt-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="bettertoken/claude-sonnet-5-5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "claude-sonnet-5-5"
    assert provider == "bettertoken"
    assert api_key == "bt-test-key"
    assert api_base == "https://bettertoken.ai/v1"


def test_bettertoken_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("BETTERTOKEN_API_KEY", "bt-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="bettertoken/claude-sonnet-5-5",
        custom_llm_provider=None,
        api_base="https://bettertoken.internal.example/v1",
        api_key="bt-explicit-key",
    )

    assert provider == "bettertoken"
    assert api_key == "bt-explicit-key"
    assert api_base == "https://bettertoken.internal.example/v1"


def test_bettertoken_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    bt = next(p for p in providers if p["litellm_provider"] == "bettertoken")

    assert bt["provider_display_name"] == "BetterToken"
    assert bt["default_model_placeholder"] == "bettertoken/claude-sonnet-5-5"
    assert {f["key"]: f["required"] for f in bt["credential_fields"]} == {"api_base": False, "api_key": True}


def test_bettertoken_cost_map_entries_exist():
    entry = litellm.model_cost["bettertoken/claude-sonnet-5-5"]
    assert entry["litellm_provider"] == "bettertoken"
    assert entry["input_cost_per_token"] > 0
