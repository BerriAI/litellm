import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../../.."))

from unittest.mock import MagicMock, patch

from litellm.types.utils import Delta, ModelResponse, StreamingChoices


def test_anthropic_experimental_pass_through_messages_handler():
    """
    Test that api key is passed to litellm.completion
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    with patch("litellm.completion", return_value="test-response") as mock_completion:
        try:
            anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                model="openai/claude-3-5-sonnet-20240620",
                api_key="test-api-key",
            )
        except Exception as e:
            print(f"Error: {e}")
        mock_completion.assert_called_once()
        mock_completion.call_args.kwargs["api_key"] == "test-api-key"


def test_anthropic_experimental_pass_through_messages_handler_dynamic_api_key_and_api_base_and_custom_values():
    """
    Test that api key is passed to litellm.completion
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    with patch("litellm.completion", return_value="test-response") as mock_completion:
        try:
            anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                model="azure/o1",
                api_key="test-api-key",
                api_base="test-api-base",
                custom_key="custom_value",
            )
        except Exception as e:
            print(f"Error: {e}")
        mock_completion.assert_called_once()
        mock_completion.call_args.kwargs["api_key"] == "test-api-key"
        mock_completion.call_args.kwargs["api_base"] == "test-api-base"
        mock_completion.call_args.kwargs["custom_key"] == "custom_value"


def test_anthropic_experimental_pass_through_messages_handler_custom_llm_provider():
    """
    Test that litellm.completion is called when a custom LLM provider is given
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    with patch("litellm.completion", return_value="test-response") as mock_completion:
        try:
            anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                model="my-custom-model",
                custom_llm_provider="my-custom-llm",
                api_key="test-api-key",
            )
        except Exception as e:
            print(f"Error: {e}")

        # Assert that litellm.completion was called when using a custom LLM provider
        mock_completion.assert_called_once()

        # Verify that the custom provider was passed through
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["custom_llm_provider"] == "my-custom-llm"
        assert call_kwargs["model"] == "my-custom-llm/my-custom-model"
        assert call_kwargs["api_key"] == "test-api-key"
