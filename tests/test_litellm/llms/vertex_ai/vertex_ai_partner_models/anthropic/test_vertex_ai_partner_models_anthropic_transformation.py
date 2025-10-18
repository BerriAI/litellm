import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../../..")
)  # Adds the parent directory to the system path
from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import (
    VertexAIAnthropicConfig,
)
from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.experimental_pass_through.transformation import (
    VertexAIPartnerModelsAnthropicMessagesConfig,
)


@pytest.mark.parametrize(
    "model, expected_thinking",
    [
        ("claude-sonnet-4@20250514", True),
    ],
)
def test_vertex_ai_anthropic_thinking_param(model, expected_thinking):
    supported_openai_params = VertexAIAnthropicConfig().get_supported_openai_params(
        model=model
    )

    if expected_thinking:
        assert "thinking" in supported_openai_params
    else:
        assert "thinking" not in supported_openai_params


def test_get_supported_params_thinking():
    config = VertexAIAnthropicConfig()
    params = config.get_supported_openai_params(model="claude-sonnet-4")
    assert "thinking" in params


class TestVertexAIAnthropicMessagesImageProcessing:
    """Test async image processing for Vertex AI Anthropic Messages."""

    @pytest.mark.asyncio
    async def test_async_process_image_content_with_urls(self):
        """Test that async_anthropic_provider_process_image_content correctly converts URLs to base64."""
        from litellm.litellm_core_utils.prompt_templates.image_handling import async_anthropic_provider_process_image_content
        
        # Test with image URL in string format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    },
                    {
                        "type": "image",
                        "source": "https://example.com/image.jpg"
                    }
                ]
            }
        ]
        
        # Mock the async_convert_url_to_base64 function
        with patch('litellm.litellm_core_utils.prompt_templates.image_handling.async_convert_url_to_base64') as mock_convert:
            mock_convert.return_value = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD"
            
            result = await async_anthropic_provider_process_image_content(messages)
            
            # Verify the URL was converted to base64 format
            image_content = result[0]["content"][1]
            assert image_content["source"]["type"] == "base64"
            assert image_content["source"]["media_type"] == "image/jpeg"
            assert image_content["source"]["data"] == "/9j/4AAQSkZJRgABAQAAAQABAAD"
            mock_convert.assert_called_once_with("https://example.com/image.jpg")

    @pytest.mark.asyncio
    async def test_async_process_image_content_with_url_objects(self):
        """Test that async_anthropic_provider_process_image_content correctly handles URL objects."""
        from litellm.litellm_core_utils.prompt_templates.image_handling import async_anthropic_provider_process_image_content
        
        # Test with image URL in object format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "https://example.com/image.png"
                        }
                    }
                ]
            }
        ]
        
        # Mock the async_convert_url_to_base64 function
        with patch('litellm.litellm_core_utils.prompt_templates.image_handling.async_convert_url_to_base64') as mock_convert:
            mock_convert.return_value = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
            
            result = await async_anthropic_provider_process_image_content(messages)
            
            # Verify the URL was converted to base64 format
            image_content = result[0]["content"][1]
            assert image_content["source"]["type"] == "base64"
            assert image_content["source"]["media_type"] == "image/png"
            assert image_content["source"]["data"] == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
            mock_convert.assert_called_once_with("https://example.com/image.png")

    @pytest.mark.asyncio
    async def test_vertex_ai_transform_anthropic_messages_request_with_images(self):
        """Test that Vertex AI transform_anthropic_messages_request uses async image processing."""
        config = VertexAIPartnerModelsAnthropicMessagesConfig()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    },
                    {
                        "type": "image",
                        "source": "https://example.com/image.jpg"
                    }
                ]
            }
        ]
        
        # Mock the async_anthropic_provider_process_image_content function
        with patch('litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.experimental_pass_through.transformation.async_anthropic_provider_process_image_content') as mock_async_process:
            mock_async_process.return_value = messages  # Return messages unchanged for simplicity
            
            result = await config.transform_anthropic_messages_request(
                model="claude-3-5-sonnet-20241022-v2:0",
                messages=messages,
                anthropic_messages_optional_request_params={"max_tokens": 100},
                litellm_params={},
                headers={}
            )
            
            # Verify async_anthropic_provider_process_image_content was called
            mock_async_process.assert_called_once_with(messages)
            
            # Verify the result contains expected fields
            assert "anthropic_version" in result
            assert result["anthropic_version"] == "vertex-2023-10-16"
            assert "model" not in result  # Should be removed for Vertex AI
