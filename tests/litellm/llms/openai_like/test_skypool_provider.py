"""
Unit tests for the Skypool OpenAI-compatible provider.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

SKYPOOL_BASE_URL = "https://a.skypool.xyz/v1"


def _get_config():
    provider = JSONProviderRegistry.get("skypool")
    assert provider is not None
    config_class = create_config_class(provider)
    return config_class()


def test_skypool_provider_registered():
    provider = JSONProviderRegistry.get("skypool")
    assert provider is not None
    assert provider.base_url == SKYPOOL_BASE_URL
    assert provider.api_key_env == "SKYPOOL_API_KEY"
    assert provider.api_base_env == "SKYPOOL_API_BASE"
    assert provider.param_mappings["max_completion_tokens"] == "max_tokens"


def test_skypool_resolves_env_api_key(monkeypatch):
    config = _get_config()
    monkeypatch.setenv("SKYPOOL_API_KEY", "test-key")
    api_base, api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_base == SKYPOOL_BASE_URL
    assert api_key == "test-key"


def test_skypool_resolves_env_api_base(monkeypatch):
    config = _get_config()
    monkeypatch.setenv("SKYPOOL_API_BASE", "https://example.com/v1")
    api_base, api_key = config._get_openai_compatible_provider_info(None, "test-key")
    assert api_base == "https://example.com/v1"
    assert api_key == "test-key"


def test_skypool_complete_url_appends_endpoint():
    config = _get_config()
    url = config.get_complete_url(
        api_base=SKYPOOL_BASE_URL,
        api_key="test-key",
        model="skypool/gemma4:26b",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{SKYPOOL_BASE_URL}/chat/completions"


def test_skypool_provider_resolution():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider(
        model="skypool/gemma4:26b",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "gemma4:26b"
    assert provider == "skypool"
    assert api_base == SKYPOOL_BASE_URL


def test_skypool_provider_config_manager():
    from litellm import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_chat_config(
        model="gemma4:26b", provider=LlmProviders.SKYPOOL
    )

    assert config is not None
    assert config.custom_llm_provider == "skypool"
