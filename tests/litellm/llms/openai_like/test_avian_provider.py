"""
Unit tests for the Avian OpenAI-like provider.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

AVIAN_BASE_URL = "https://api.avian.io/v1"


def _get_config():
    provider = JSONProviderRegistry.get("avian")
    assert provider is not None
    config_class = create_config_class(provider)
    return config_class()


def test_avian_provider_registered():
    provider = JSONProviderRegistry.get("avian")
    assert provider is not None
    assert provider.base_url == AVIAN_BASE_URL
    assert provider.api_key_env == "AVIAN_API_KEY"


def test_avian_resolves_env_api_key(monkeypatch):
    config = _get_config()
    monkeypatch.setenv("AVIAN_API_KEY", "test-key")
    api_base, api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_base == AVIAN_BASE_URL
    assert api_key == "test-key"


def test_avian_complete_url_appends_endpoint():
    config = _get_config()
    url = config.get_complete_url(
        api_base=AVIAN_BASE_URL,
        api_key="test-key",
        model="avian/deepseek/deepseek-v3.2",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{AVIAN_BASE_URL}/chat/completions"
