"""
Test apiToken.sale Anthropic-compatible API support (messages route)
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../")
)  # Adds the parent directory to the system path

from litellm.llms.apitoken.messages.transformation import ApiTokenMessagesConfig


def test_apitoken_anthropic_config():
    """Test that ApiTokenMessagesConfig is properly configured"""
    config = ApiTokenMessagesConfig()

    # Test custom_llm_provider
    assert config.custom_llm_provider == "apitoken"

    # Test get_api_base default
    api_base = config.get_api_base()
    assert api_base == "https://api.apitoken.sale/v1/messages"

    # Test get_api_base with custom value
    custom_base = config.get_api_base(api_base="https://api.apitoken.sale/v1/messages")
    assert custom_base == "https://api.apitoken.sale/v1/messages"


def test_apitoken_get_complete_url():
    """Test that the /v1/messages suffix is appended exactly once"""
    config = ApiTokenMessagesConfig()

    url = config.get_complete_url(
        api_base="https://api.apitoken.sale",
        api_key="sk-pool-test",
        model="claude-opus-4-8",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.apitoken.sale/v1/messages"

    url = config.get_complete_url(
        api_base="https://api.apitoken.sale/v1/messages",
        api_key="sk-pool-test",
        model="claude-opus-4-8",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.apitoken.sale/v1/messages"


def test_apitoken_provider_routing():
    """Test that apitoken provider is properly routed"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    # Test with apitoken/ prefix
    model, provider, api_key, api_base = get_llm_provider(
        model="apitoken/claude-opus-4-8",
        api_base="https://api.apitoken.sale/v1/messages",
    )
    assert provider == "apitoken"
    assert model == "claude-opus-4-8"


def test_apitoken_provider_config_manager():
    """Test that ProviderConfigManager returns ApiTokenMessagesConfig"""
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="claude-opus-4-8", provider=LlmProviders.APITOKEN
    )

    assert config is not None
    assert isinstance(config, ApiTokenMessagesConfig)
    assert config.custom_llm_provider == "apitoken"
