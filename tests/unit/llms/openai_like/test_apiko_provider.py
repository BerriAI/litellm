"""
Unit tests for the APIKO OpenAI-compatible provider.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import pytest

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

APIKO_BASE_URL = "https://api.apiko.app/v1"


def _get_config():
    provider = JSONProviderRegistry.get("apiko")
    assert provider is not None
    return create_config_class(provider)()


def test_apiko_provider_registered():
    provider = JSONProviderRegistry.get("apiko")
    assert provider is not None
    assert provider.base_url == APIKO_BASE_URL
    assert provider.api_key_env == "APIKO_API_KEY"
    assert provider.api_base_env == "APIKO_API_BASE"
    assert litellm.LlmProviders.APIKO.value == "apiko"


def test_apiko_resolves_env_api_key(monkeypatch):
    monkeypatch.delenv("APIKO_API_BASE", raising=False)
    monkeypatch.setenv("APIKO_API_KEY", "test-key")
    api_base, api_key = _get_config().get_openai_compatible_provider_info(None, None)
    assert api_base == APIKO_BASE_URL
    assert api_key == "test-key"


def test_apiko_model_prefix_routes_to_provider(monkeypatch):
    monkeypatch.delenv("APIKO_API_BASE", raising=False)
    monkeypatch.setenv("APIKO_API_KEY", "test-key")
    model, provider, api_key, api_base = get_llm_provider("apiko/deepseek-v4-pro")
    assert model == "deepseek-v4-pro"
    assert provider == "apiko"
    assert api_key == "test-key"
    assert api_base == APIKO_BASE_URL


def test_apiko_complete_url_appends_endpoint():
    url = _get_config().get_complete_url(
        api_base=APIKO_BASE_URL,
        api_key="test-key",
        model="apiko/deepseek-v4-pro",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{APIKO_BASE_URL}/chat/completions"


def test_apiko_does_not_fall_back_to_openai_key(monkeypatch):
    monkeypatch.delenv("APIKO_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unrelated-openai-key")
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "openai_key", None)
    with pytest.raises(litellm.AuthenticationError, match="Set APIKO_API_KEY"):
        litellm.completion(
            model="apiko/deepseek-v4-pro",
            messages=[{"role": "user", "content": "hi"}],
        )
