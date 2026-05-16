import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.llms.github_copilot.messages.transformation import (
    GithubCopilotAnthropicMessagesConfig,
)
from litellm.llms.github_copilot.common_utils import GetAPIKeyError


def test_github_copilot_anthropic_messages_config_init():
    """Test GithubCopilotAnthropicMessagesConfig initialization."""
    config = GithubCopilotAnthropicMessagesConfig()
    assert config is not None
    assert hasattr(config, "authenticator")


def test_github_copilot_anthropic_messages_get_complete_url():
    """Test URL construction for GitHub Copilot messages endpoint."""
    config = GithubCopilotAnthropicMessagesConfig()

    # Test with default api_base
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="github_copilot/claude-haiku-4.5",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.githubcopilot.com/v1/messages"

    # Test with custom api_base
    url = config.get_complete_url(
        api_base="https://custom.api.com",
        api_key=None,
        model="github_copilot/claude-haiku-4.5",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://custom.api.com/v1/messages"

    # Test with api_base already ending with /v1/messages
    url = config.get_complete_url(
        api_base="https://custom.api.com/v1/messages",
        api_key=None,
        model="github_copilot/claude-haiku-4.5",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://custom.api.com/v1/messages"


def test_github_copilot_anthropic_messages_validate_environment():
    """Test environment validation and header injection."""
    config = GithubCopilotAnthropicMessagesConfig()

    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    headers = {}
    validated_headers, api_base = config.validate_anthropic_messages_environment(
        headers=headers,
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    # Check that Copilot headers were added
    assert "copilot-integration-id" in validated_headers
    assert validated_headers["copilot-integration-id"] == "vscode-chat"
    assert "Authorization" in validated_headers
    assert "anthropic-version" in validated_headers
    assert validated_headers["anthropic-version"] == "2023-06-01"
    assert api_base == "https://api.githubcopilot.com"


def test_github_copilot_anthropic_messages_validate_environment_auth_error():
    """Test error handling when authentication fails."""
    config = GithubCopilotAnthropicMessagesConfig()

    # Mock the authenticator to raise an error
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.side_effect = GetAPIKeyError(
        status_code=401, message="No valid API key found"
    )

    with pytest.raises(Exception):  # AuthenticationError
        config.validate_anthropic_messages_environment(
            headers={},
            model="github_copilot/claude-haiku-4.5",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )


def test_github_copilot_anthropic_messages_supported_params():
    """Test supported parameters list."""
    config = GithubCopilotAnthropicMessagesConfig()
    params = config.get_supported_anthropic_messages_params("github_copilot/claude-haiku-4.5")

    # Should inherit from AnthropicMessagesConfig
    assert "messages" in params
    assert "model" in params
    assert "max_tokens" in params
    assert "thinking" in params
