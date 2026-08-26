"""
Unit tests for the AssemblyAI LLM Gateway OpenAI-like provider.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

ASSEMBLYAI_BASE_URL = "https://llm-gateway.assemblyai.com/v1"


def _get_config():
    provider = JSONProviderRegistry.get("assemblyai")
    assert provider is not None
    config_class = create_config_class(provider)
    return config_class()


def test_assemblyai_provider_registered():
    provider = JSONProviderRegistry.get("assemblyai")
    assert provider is not None
    assert provider.base_url == ASSEMBLYAI_BASE_URL
    assert provider.api_key_env == "ASSEMBLYAI_API_KEY"


def test_assemblyai_resolves_env_api_key(monkeypatch):
    config = _get_config()
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-key")
    api_base, api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_base == ASSEMBLYAI_BASE_URL
    assert api_key == "test-key"


def test_assemblyai_complete_url_appends_endpoint():
    config = _get_config()
    url = config.get_complete_url(
        api_base=ASSEMBLYAI_BASE_URL,
        api_key="test-key",
        model="assemblyai/claude-sonnet-4-5-20250929",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{ASSEMBLYAI_BASE_URL}/chat/completions"


def test_assemblyai_provider_resolution():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider(
        model="assemblyai/claude-sonnet-4-5-20250929",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "claude-sonnet-4-5-20250929"
    assert provider == "assemblyai"
    assert api_base == ASSEMBLYAI_BASE_URL


def test_assemblyai_provider_config_manager():
    from litellm import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_chat_config(
        model="claude-sonnet-4-5-20250929", provider=LlmProviders.ASSEMBLYAI
    )

    assert config is not None
    assert config.custom_llm_provider == "assemblyai"
