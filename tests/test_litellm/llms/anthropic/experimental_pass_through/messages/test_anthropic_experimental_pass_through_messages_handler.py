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


def test_anthropic_messages_image_url_handling_anthropic_provider():
    """
    Test that Anthropic provider keeps image URLs as-is (no conversion)
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
        AnthropicMessagesConfig,
    )
    
    config = AnthropicMessagesConfig()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/image.jpg"}
                }
            ]
        }
    ]
    
    # Mock the image handling function to avoid actual network calls
    with patch('litellm.litellm_core_utils.prompt_templates.image_handling.convert_url_to_base64') as mock_convert:
        result = config.transform_anthropic_messages_request(
            model="claude-3-5-sonnet-20241022",
            messages=messages,
            anthropic_messages_optional_request_params={"max_tokens": 100},
            litellm_params={},
            headers={}
        )
    
    # Verify that the image URL was kept as-is (no conversion called)
    mock_convert.assert_not_called()
    
    # Verify the request structure
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"
    assert len(result["messages"][0]["content"]) == 2
    assert result["messages"][0]["content"][1]["type"] == "image"
    assert result["messages"][0]["content"][1]["source"]["type"] == "url"
    assert result["messages"][0]["content"][1]["source"]["url"] == "https://example.com/image.jpg"


def test_anthropic_messages_image_url_handling_bedrock_provider():
    """
    Test that Bedrock provider converts image URLs to base64
    """
    from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
        AmazonAnthropicClaudeMessagesConfig,
    )
    
    config = AmazonAnthropicClaudeMessagesConfig()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/image.jpg"}
                }
            ]
        }
    ]
    
    # Mock the image handling function to return mock base64 data
    with patch('litellm.litellm_core_utils.prompt_templates.image_handling.convert_url_to_base64') as mock_convert:
        mock_convert.return_value = "data:image/jpeg;base64,mock_base64_data"
        
        result = config.transform_anthropic_messages_request(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=messages,
            anthropic_messages_optional_request_params={"max_tokens": 100},
            litellm_params={},
            headers={}
        )
    
    # Verify that the image URL was converted to base64
    mock_convert.assert_called_once_with("https://example.com/image.jpg")
    
    # Verify the request structure
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"
    assert len(result["messages"][0]["content"]) == 2
    assert result["messages"][0]["content"][1]["type"] == "image"
    assert result["messages"][0]["content"][1]["source"]["type"] == "base64"
    assert result["messages"][0]["content"][1]["source"]["media_type"] == "image/jpeg"
    assert result["messages"][0]["content"][1]["source"]["data"] == "mock_base64_data"


def test_anthropic_messages_image_url_handling_vertex_ai_provider():
    """
    Test that Vertex AI provider converts image URLs to base64
    """
    from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.experimental_pass_through.transformation import (
        VertexAIPartnerModelsAnthropicMessagesConfig,
    )
    
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/image.jpg"}
                }
            ]
        }
    ]
    
    # Mock the image handling function to return mock base64 data
    with patch('litellm.litellm_core_utils.prompt_templates.image_handling.convert_url_to_base64') as mock_convert:
        mock_convert.return_value = "data:image/jpeg;base64,mock_base64_data"
        
        result = config.transform_anthropic_messages_request(
            model="claude-3-5-sonnet-20241022",
            messages=messages,
            anthropic_messages_optional_request_params={"max_tokens": 100},
            litellm_params={},
            headers={}
        )
    
    # Verify that the image URL was converted to base64
    mock_convert.assert_called_once_with("https://example.com/image.jpg")
    
    # Verify the request structure
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"
    assert len(result["messages"][0]["content"]) == 2
    assert result["messages"][0]["content"][1]["type"] == "image"
    assert result["messages"][0]["content"][1]["source"]["type"] == "base64"
    assert result["messages"][0]["content"][1]["source"]["media_type"] == "image/jpeg"
    assert result["messages"][0]["content"][1]["source"]["data"] == "mock_base64_data"
