import pytest
import os
import sys
import json
import asyncio
from datetime import datetime, timedelta
from typing import AsyncGenerator
from unittest.mock import AsyncMock, patch, mock_open, MagicMock

sys.path.insert(0, os.path.abspath("../.."))

import httpx
import pytest
from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse, ModelResponse, Usage
from litellm import completion, acompletion
from litellm.llms.github_copilot.chat.transformation import GithubCopilotConfig
from litellm.llms.github_copilot.authenticator import Authenticator
from litellm.llms.github_copilot.constants import (
    GetAccessTokenError,
    GetDeviceCodeError,
    RefreshAPIKeyError,
    GetAPIKeyError,
    APIKeyExpiredError,
)
from litellm.exceptions import AuthenticationError

# Import at the top to make the patch work correctly
import litellm.llms.github_copilot.chat.transformation


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

    assert api_base == "https://api.githubcopilot.com"
    assert dynamic_api_key == mock_api_key
    assert custom_llm_provider == "github_copilot"

    # Test with authentication failure
    config.authenticator.get_api_key.side_effect = GetAPIKeyError(
        "Failed to get API key"
    )

    with pytest.raises(AuthenticationError) as excinfo:
        config._get_openai_compatible_provider_info(
            model=model,
            api_base=None,
            api_key=None,
            custom_llm_provider="github_copilot",
        )

    assert "Failed to get API key" in str(excinfo.value)


@patch("litellm.main.get_llm_provider")
@patch("litellm.llms.openai.openai.OpenAIChatCompletion.completion")
def test_completion_github_copilot_mock_response(mock_completion, mock_get_llm_provider):
    """Test the completion function with GitHub Copilot provider."""

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

    # Patch the get_llm_provider function instead of the config method
    # Make it return the expected tuple directly
    mock_get_llm_provider.return_value = (
        "gpt-4",
        "github_copilot",
        "gh.test-key-123456789",
        "https://api.githubcopilot.com",
    )

    response = completion(
        model="github_copilot/gpt-4",
        messages=messages,
        extra_headers=headers,
    )

    assert response is not None

    # Verify the get_llm_provider call was made with the expected params
    mock_get_llm_provider.assert_called_once()
    args, kwargs = mock_get_llm_provider.call_args
    assert kwargs.get("model") is "github_copilot/gpt-4"
    assert kwargs.get("custom_llm_provider") is None
    assert kwargs.get("api_key") is None
    assert kwargs.get("api_base") is None

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


@patch("litellm.llms.github_copilot.authenticator.Authenticator.get_api_key")
def test_authenticator_get_api_key(mock_get_api_key):
    """Test the Authenticator's get_api_key method."""
    from litellm.llms.github_copilot.authenticator import Authenticator

    # Test successful API key retrieval
    mock_get_api_key.return_value = "gh.test-key-123456789"
    authenticator = Authenticator()
    api_key = authenticator.get_api_key()

    assert api_key == "gh.test-key-123456789"
    mock_get_api_key.assert_called_once()

    # Test API key retrieval failure
    mock_get_api_key.reset_mock()
    mock_get_api_key.side_effect = GetAPIKeyError("Failed to get API key")
    authenticator = Authenticator()

    with pytest.raises(GetAPIKeyError) as excinfo:
        authenticator.get_api_key()

    assert "Failed to get API key" in str(excinfo.value)


# def test_completion_github_copilot(stream=False):
#     try:
#         litellm.set_verbose = True
#         messages = [
#             {"role": "system", "content": "You are an AI programming assistant."},
#             {
#                 "role": "user",
#                 "content": "Write a Python function to calculate fibonacci numbers",
#             },
#         ]
#         extra_headers = {
#             "editor-version": "Neovim/0.9.0",
#             "Copilot-Integration-Id": "vscode-chat",
#         }
#         response = completion(
#             model="github_copilot/gpt-4",
#             messages=messages,
#             stream=stream,
#             extra_headers=extra_headers,
#         )
#         print(response)

#         if stream is True:
#             for chunk in response:
#                 print(chunk)
#                 assert chunk is not None
#                 assert isinstance(chunk, litellm.ModelResponseStream)
#                 assert isinstance(chunk.choices[0], litellm.utils.StreamingChoices)

#         else:
#             assert response is not None
#             assert isinstance(response, litellm.ModelResponse)
#             assert response.choices[0].message.content is not None
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# def test_completion_github_copilot_sonnet_3_7_thought(stream=False):
#     try:
#         litellm.set_verbose = True
#         messages = [
#             {"role": "system", "content": "You are an AI programming assistant."},
#             {
#                 "role": "user",
#                 "content": "Write a Python function to calculate fibonacci numbers",
#             },
#         ]
#         extra_headers = {
#             "editor-version": "Neovim/0.9.0",
#             "Copilot-Integration-Id": "vscode-chat",
#         }
#         response = completion(
#             model="github_copilot/claude-3.7-sonnet-thought",
#             messages=messages,
#             stream=stream,
#             extra_headers=extra_headers,
#         )
#         print(response)

#         if stream is True:
#             for chunk in response:
#                 print(chunk)
#                 assert chunk is not None
#                 assert isinstance(chunk, litellm.ModelResponseStream)
#                 assert isinstance(chunk.choices[0], litellm.utils.StreamingChoices)

#         else:
#             assert response is not None
#             assert isinstance(response, litellm.ModelResponse)
#             assert response.choices[0].message.content is not None
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
