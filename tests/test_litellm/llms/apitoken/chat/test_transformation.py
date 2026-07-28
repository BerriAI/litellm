"""
Test apiToken.sale chat completions support (Anthropic-format upstream)
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.apitoken.chat.transformation import ApiTokenChatConfig


def test_apitoken_chat_config():
    """Test that ApiTokenChatConfig is properly configured"""
    config = ApiTokenChatConfig()

    assert config.custom_llm_provider == "apitoken"
    assert config.get_api_base() == "https://api.apitoken.sale"
    assert config.get_api_base("https://example.com") == "https://example.com"


def test_apitoken_chat_complete_url():
    """Test that the chat config targets /v1/messages"""
    config = ApiTokenChatConfig()

    url = config.get_complete_url(
        api_base=None,
        api_key="sk-pool-test",
        model="claude-opus-4-8",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.apitoken.sale/v1/messages"


def test_apitoken_validate_environment_headers():
    """Test that validate_environment produces Anthropic-style headers with the apitoken key"""
    config = ApiTokenChatConfig()

    headers = config.validate_environment(
        headers={},
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key="sk-pool-test",
        api_base="https://api.apitoken.sale",
    )

    assert headers["x-api-key"] == "sk-pool-test"
    assert headers["anthropic-version"] == "2023-06-01"


def test_apitoken_validate_environment_missing_key():
    """Test that a missing key raises AuthenticationError mentioning APITOKEN_API_KEY"""
    config = ApiTokenChatConfig()

    with patch.dict(os.environ, {}, clear=True):
        litellm.api_key = None
        with pytest.raises(litellm.AuthenticationError) as excinfo:
            config.validate_environment(
                headers={},
                model="claude-opus-4-8",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
            )
        assert "APITOKEN_API_KEY" in str(excinfo.value)


def test_apitoken_chat_provider_config_manager():
    """Test that ProviderConfigManager returns ApiTokenChatConfig for chat"""
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_chat_config(
        model="claude-opus-4-8", provider=LlmProviders.APITOKEN
    )

    assert config is not None
    assert isinstance(config, ApiTokenChatConfig)


@pytest.mark.skip(reason="Requires actual apiToken.sale API key")
def test_apitoken_completion_basic():
    """Test basic completion with the apiToken.sale Anthropic-compatible API"""
    response = litellm.completion(
        model="apitoken/claude-haiku-4-5",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        api_key=os.getenv("APITOKEN_API_KEY"),
    )

    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
