"""
Unit tests for the Abliteration OpenAI-like provider.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

ABLITERATION_BASE_URL = "https://api.abliteration.ai/v1"


def _get_config():
    provider = JSONProviderRegistry.get("abliteration")
    assert provider is not None
    config_class = create_config_class(provider)
    return config_class()


def test_abliteration_provider_registered():
    provider = JSONProviderRegistry.get("abliteration")
    assert provider is not None
    assert provider.base_url == ABLITERATION_BASE_URL
    assert provider.api_key_env == "ABLITERATION_API_KEY"


def test_abliteration_resolves_env_api_key(monkeypatch):
    config = _get_config()
    monkeypatch.setenv("ABLITERATION_API_KEY", "test-key")
    api_base, api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_base == ABLITERATION_BASE_URL
    assert api_key == "test-key"


def test_abliteration_complete_url_appends_endpoint():
    config = _get_config()
    url = config.get_complete_url(
        api_base=ABLITERATION_BASE_URL,
        api_key="test-key",
        model="abliteration/abliterated-model",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{ABLITERATION_BASE_URL}/chat/completions"
