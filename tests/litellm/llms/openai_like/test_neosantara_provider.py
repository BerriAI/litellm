"""
Unit tests for the Neosantara OpenAI-like provider.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

NEOSANTARA_BASE_URL = "https://api.neosantara.xyz/v1"


def _get_config():
    provider = JSONProviderRegistry.get("neosantara")
    assert provider is not None
    config_class = create_config_class(provider)
    return config_class()


def test_neosantara_provider_registered():
    provider = JSONProviderRegistry.get("neosantara")
    assert provider is not None
    assert provider.base_url == NEOSANTARA_BASE_URL
    assert provider.api_key_env == "NEOSANTARA_API_KEY"
    assert provider.api_base_env == "NEOSANTARA_API_BASE"
    assert "/v1/responses" in provider.supported_endpoints


def test_neosantara_resolves_env_api_key(monkeypatch):
    config = _get_config()
    monkeypatch.setenv("NEOSANTARA_API_KEY", "test-key")
    api_base, api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_base == NEOSANTARA_BASE_URL
    assert api_key == "test-key"


def test_neosantara_complete_url_appends_endpoint():
    config = _get_config()
    url = config.get_complete_url(
        api_base=NEOSANTARA_BASE_URL,
        api_key="test-key",
        model="neosantara/claude-4.5-sonnet",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{NEOSANTARA_BASE_URL}/chat/completions"


def test_neosantara_provider_resolution(monkeypatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.delenv("NEOSANTARA_API_KEY", raising=False)
    model, provider, api_key, api_base = get_llm_provider(
        model="neosantara/claude-4.5-sonnet",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "claude-4.5-sonnet"
    assert provider == "neosantara"
    assert api_base == NEOSANTARA_BASE_URL
    assert api_key is None


def test_neosantara_provider_config_manager():
    from litellm import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_chat_config(
        model="claude-4.5-sonnet", provider=LlmProviders.NEOSANTARA
    )

    assert config is not None
    assert config.custom_llm_provider == "neosantara"
