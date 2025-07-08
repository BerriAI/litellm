import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import httpx
import pytest
from respx import MockRouter

import litellm

# Import at the top to make the patch work correctly
import litellm.llms.github_copilot.chat.transformation
from litellm import Choices, Message, ModelResponse, Usage, acompletion, completion
from litellm.exceptions import AuthenticationError
from litellm.llms.github_copilot.authenticator import Authenticator
from litellm.llms.github_copilot.chat.transformation import GithubCopilotConfig
from litellm.llms.github_copilot.common_utils import (
    APIKeyExpiredError,
    GetAccessTokenError,
    GetAPIKeyError,
    GetDeviceCodeError,
    RefreshAPIKeyError,
)


def test_github_copilot_config_get_openai_compatible_provider_info():
    """Test the GitHub Copilot configuration provider info retrieval."""

    config = GithubCopilotConfig()

    # Mock the authenticator to avoid actual API calls
    mock_api_key = "gh.test-key-123456789"
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = mock_api_key

    # Test with default values
    model = "github_copilot/gpt-4"
    (
        api_base,
        dynamic_api_key,
        custom_llm_provider,
    ) = config._get_openai_compatible_provider_info(
        model=model,
        api_base=None,
        api_key=None,
        custom_llm_provider="github_copilot",
    )

    assert api_base == "https://api.githubcopilot.com/"
    assert dynamic_api_key == mock_api_key
    assert custom_llm_provider == "github_copilot"

    # Test with authentication failure
    config.authenticator.get_api_key.side_effect = GetAPIKeyError(
        message="Failed to get API key",
        status_code=401,
    )

    with pytest.raises(AuthenticationError) as excinfo:
        config._get_openai_compatible_provider_info(
            model=model,
            api_base=None,
            api_key=None,
            custom_llm_provider="github_copilot",
        )

    assert "Failed to get API key" in str(excinfo.value)


@patch("litellm.llms.github_copilot.authenticator.Authenticator.get_api_key")
@patch("litellm.llms.openai.openai.OpenAIChatCompletion.completion")
def test_completion_github_copilot_mock_response(mock_completion, mock_get_api_key):
    """Test the completion function with GitHub Copilot provider."""

    # Mock the API key return value
    mock_api_key = "gh.test-key-123456789"
    mock_get_api_key.return_value = mock_api_key

    # Mock completion response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello, I'm GitHub Copilot!"
    mock_completion.return_value = mock_response

    # Test non-streaming completion
    messages = [
        {"role": "system", "content": "You're GitHub Copilot, an AI assistant."},
        {"role": "user", "content": "Hello, who are you?"},
    ]

    # Create a properly formatted headers dictionary
    headers = {
        "editor-version": "Neovim/0.9.0",
        "Copilot-Integration-Id": "vscode-chat",
    }

    response = completion(
        model="github_copilot/gpt-4",
        messages=messages,
        extra_headers=headers,
    )

    assert response is not None

    # Verify the get_api_key call was made (can be called multiple times)
    assert mock_get_api_key.call_count >= 1

    # Verify the completion call was made with the expected params
    mock_completion.assert_called_once()
    args, kwargs = mock_completion.call_args

    # Check that the proper authorization header is set
    assert "headers" in kwargs
    # Check that the model name is correctly formatted
    assert (
        kwargs.get("model") == "gpt-4"
    )  # Model name should be without provider prefix
    assert kwargs.get("messages") == messages


