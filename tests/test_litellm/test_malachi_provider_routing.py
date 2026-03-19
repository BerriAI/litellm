"""
Focused tests for LiteLLM Malachi provider routing and config registration.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)

import litellm
from litellm.constants import LITELLM_CHAT_PROVIDERS, openai_compatible_providers
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_malachi_in_provider_allowlists():
    assert "malachi" in LITELLM_CHAT_PROVIDERS
    assert "malachi" in openai_compatible_providers


def test_malachi_provider_routing():
    model, provider, dynamic_api_key, api_base = litellm.get_llm_provider(
        model="malachi/gpt-5.4",
        api_base=None,
        api_key=None,
    )

    assert model == "gpt-5.4"
    assert provider == "malachi"
    assert dynamic_api_key is None
    assert api_base is None


def test_malachi_resolves_env_api_base_and_key(monkeypatch):
    monkeypatch.setenv("MALACHI_API_BASE", "https://malachi.example.com/v1")
    monkeypatch.setenv("MALACHI_API_KEY", "malachi-test-key")

    model, provider, dynamic_api_key, api_base = litellm.get_llm_provider(
        model="malachi/gpt-5.4",
        api_base=None,
        api_key=None,
    )

    assert model == "gpt-5.4"
    assert provider == "malachi"
    assert dynamic_api_key == "malachi-test-key"
    assert api_base == "https://malachi.example.com/v1"


def test_malachi_api_base_auto_detects_provider(monkeypatch):
    monkeypatch.setenv("MALACHI_API_BASE", "https://malachi.example.com/v1")
    monkeypatch.setenv("MALACHI_API_KEY", "malachi-test-key")

    model, provider, dynamic_api_key, api_base = litellm.get_llm_provider(
        model="gpt-5.4",
        api_base="https://malachi.example.com/v1",
        api_key=None,
    )

    assert model == "gpt-5.4"
    assert provider == "malachi"
    assert dynamic_api_key == "malachi-test-key"
    assert api_base == "https://malachi.example.com/v1"


def test_malachi_provider_config_manager_chat():
    config = ProviderConfigManager.get_provider_chat_config(
        model="gpt-5.4",
        provider=LlmProviders.MALACHI,
    )

    assert config is not None
    assert config.custom_llm_provider == LlmProviders.MALACHI


def test_malachi_chat_config_reads_provider_env(monkeypatch):
    monkeypatch.setenv("MALACHI_API_BASE", "https://malachi.example.com/v1")
    monkeypatch.setenv("MALACHI_API_KEY", "malachi-test-key")

    config = ProviderConfigManager.get_provider_chat_config(
        model="gpt-5.4",
        provider=LlmProviders.MALACHI,
    )

    api_base, api_key = config._get_openai_compatible_provider_info(
        api_base=None,
        api_key=None,
    )

    assert api_base == "https://malachi.example.com/v1"
    assert api_key == "malachi-test-key"


def test_malachi_provider_config_manager_responses():
    config = ProviderConfigManager.get_provider_responses_api_config(
        model="gpt-5.4",
        provider=LlmProviders.MALACHI,
    )

    assert config is not None
    assert config.custom_llm_provider == LlmProviders.MALACHI


def test_malachi_responses_config_reads_provider_env(monkeypatch):
    monkeypatch.setenv("MALACHI_API_BASE", "https://malachi.example.com/v1")
    monkeypatch.setenv("MALACHI_API_KEY", "malachi-test-key")

    config = ProviderConfigManager.get_provider_responses_api_config(
        model="gpt-5.4",
        provider=LlmProviders.MALACHI,
    )

    headers = config.validate_environment(
        headers={},
        model="gpt-5.4",
        litellm_params=None,
    )
    url = config.get_complete_url(
        api_base=None,
        litellm_params={},
    )

    assert headers["Authorization"] == "Bearer malachi-test-key"
    assert url == "https://malachi.example.com/v1/responses"
