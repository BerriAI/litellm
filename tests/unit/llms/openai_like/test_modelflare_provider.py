"""
Unit tests for the Modelflare OpenAI-compatible provider (JSON-configured).
"""

import pytest

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

MODELFLARE_BASE_URL = "https://modelflare.dev/v1"


def test_modelflare_is_a_registered_provider():
    assert litellm.LlmProviders.MODELFLARE.value == "modelflare"
    assert "modelflare" in litellm.provider_list
    assert "modelflare" in litellm.constants.openai_compatible_providers

    provider = JSONProviderRegistry.get("modelflare")
    assert provider is not None
    assert provider.base_url == MODELFLARE_BASE_URL
    assert provider.api_key_env == "MODELFLARE_API_KEY"
    assert provider.api_base_env == "MODELFLARE_API_BASE"


def test_modelflare_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODELFLARE_API_KEY", "modelflare-test-key")
    monkeypatch.delenv("MODELFLARE_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="modelflare/some-model",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "some-model"
    assert provider == "modelflare"
    assert api_key == "modelflare-test-key"
    assert api_base == MODELFLARE_BASE_URL


def test_modelflare_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODELFLARE_API_KEY", "modelflare-test-key")
    monkeypatch.setenv("MODELFLARE_API_BASE", "https://modelflare.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="modelflare/some-model",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "modelflare"
    assert api_base == "https://modelflare.env.example/v1"
