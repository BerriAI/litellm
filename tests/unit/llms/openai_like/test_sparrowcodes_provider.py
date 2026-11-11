"""
Unit tests for the SparrowCodes OpenAI-like provider.
"""

import json
from pathlib import Path

import pytest

import litellm
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

SPARROWCODES_BASE_URL = "https://api.sparrowcodes.com/v1"


def _get_config():
    provider = JSONProviderRegistry.get("sparrowcodes")
    assert provider is not None
    return create_config_class(provider)()


def test_sparrowcodes_provider_registered():
    provider = JSONProviderRegistry.get("sparrowcodes")
    assert provider is not None
    assert provider.base_url == SPARROWCODES_BASE_URL
    assert provider.api_key_env == "SPARROWCODES_API_KEY"


def test_sparrowcodes_resolves_env_api_key(monkeypatch: pytest.MonkeyPatch):
    config = _get_config()
    monkeypatch.setenv("SPARROWCODES_API_KEY", "sc-test-key")
    api_base, api_key = config.get_openai_compatible_provider_info(None, None)
    assert api_base == SPARROWCODES_BASE_URL
    assert api_key == "sc-test-key"


def test_sparrowcodes_complete_url_appends_endpoint():
    config = _get_config()
    url = config.get_complete_url(
        api_base=SPARROWCODES_BASE_URL,
        api_key="sc-test-key",
        model="sparrowcodes/sparrow/auto",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{SPARROWCODES_BASE_URL}/chat/completions"


def test_sparrowcodes_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SPARROWCODES_API_KEY", "sc-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="sparrowcodes/anthropic/claude-sonnet-5.5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "anthropic/claude-sonnet-5.5"
    assert provider == "sparrowcodes"
    assert api_key == "sc-test-key"
    assert api_base == SPARROWCODES_BASE_URL


def test_sparrowcodes_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SPARROWCODES_API_KEY", "sc-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="sparrowcodes/sparrow/auto",
        custom_llm_provider=None,
        api_base="https://proxy.example.com/v1",
        api_key="sc-explicit-key",
    )

    assert provider == "sparrowcodes"
    assert api_key == "sc-explicit-key"
    assert api_base == "https://proxy.example.com/v1"


def test_sparrowcodes_provider_config_manager():
    from litellm import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_chat_config(
        model="sparrow/auto", provider=LlmProviders.SPARROWCODES
    )

    assert config is not None
    assert config.custom_llm_provider == "sparrowcodes"


def test_sparrowcodes_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    entry = next(p for p in providers if p["litellm_provider"] == "sparrowcodes")

    assert entry["provider"] == "SPARROWCODES"
    assert entry["provider_display_name"] == "SparrowCodes"
    assert entry["default_model_placeholder"] == "sparrowcodes/sparrow/auto"
    assert {field["key"]: field["required"] for field in entry["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }
