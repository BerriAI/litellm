"""Tests for the Requesty provider (OpenAI-compatible gateway)."""

import litellm
from litellm import get_llm_provider
from litellm.llms.requesty.chat.transformation import RequestyConfig


def test_get_llm_provider_resolves_requesty():
    """requesty/<provider>/<model> routes to the requesty provider + base URL."""
    model, custom_llm_provider, _dynamic_api_key, api_base = get_llm_provider(
        model="requesty/openai/gpt-4o-mini"
    )
    assert custom_llm_provider == "requesty"
    assert model == "openai/gpt-4o-mini"
    assert api_base == "https://router.requesty.ai/v1"


def test_requesty_in_provider_registries():
    """requesty is registered in the provider list and openai-compatible sets."""
    assert "requesty" in litellm.provider_list
    assert "requesty" in litellm.openai_compatible_providers
    assert "https://router.requesty.ai/v1" in litellm.openai_compatible_endpoints


def test_requesty_config_default_base_url():
    """RequestyConfig exposes the fixed Requesty router base URL."""
    api_base, _dynamic_api_key = RequestyConfig()._get_openai_compatible_provider_info(
        api_base=None, api_key="test-key"
    )
    assert api_base == "https://router.requesty.ai/v1"
