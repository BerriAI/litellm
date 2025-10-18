"""
Test anthropic_beta header support for AWS Bedrock.

Tests that anthropic-beta headers are correctly processed and passed to AWS Bedrock
for enabling beta features like 1M context window, computer use tools, etc.
"""

import pytest
from unittest.mock import patch, MagicMock
import json

from litellm.llms.bedrock.common_utils import get_anthropic_beta_from_headers
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import AmazonAnthropicClaudeConfig
from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import AmazonAnthropicClaudeMessagesConfig


class TestAnthropicBetaHeaderSupport:
    """Test anthropic_beta header functionality across Bedrock APIs."""

    def test_get_anthropic_beta_from_headers_empty(self):
        """Test header extraction with no headers."""
        headers = {}
        result = get_anthropic_beta_from_headers(headers)
        assert result == []

    def test_get_anthropic_beta_from_headers_single(self):
        """Test header extraction with single beta header."""
        headers = {"anthropic-beta": "context-1m-2025-08-07"}
        result = get_anthropic_beta_from_headers(headers)
        assert result == ["context-1m-2025-08-07"]

    def test_get_anthropic_beta_from_headers_multiple(self):
        """Test header extraction with multiple comma-separated beta headers."""
        headers = {"anthropic-beta": "context-1m-2025-08-07,computer-use-2024-10-22"}
        result = get_anthropic_beta_from_headers(headers)
        assert result == ["context-1m-2025-08-07", "computer-use-2024-10-22"]

    def test_get_anthropic_beta_from_headers_whitespace(self):
        """Test header extraction handles whitespace correctly."""
        headers = {"anthropic-beta": " context-1m-2025-08-07 , computer-use-2024-10-22 "}
        result = get_anthropic_beta_from_headers(headers)
        assert result == ["context-1m-2025-08-07", "computer-use-2024-10-22"]

    def test_invoke_transformation_anthropic_beta(self):
        """Test that Invoke API transformation includes anthropic_beta in request."""
        config = AmazonAnthropicClaudeConfig()
        headers = {"anthropic-beta": "context-1m-2025-08-07,computer-use-2024-10-22"}
        
        result = config.transform_request(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": "Test"}],
            optional_params={},
            litellm_params={},
            headers=headers
        )
        
        assert "anthropic_beta" in result
        assert result["anthropic_beta"] == ["context-1m-2025-08-07", "computer-use-2024-10-22"]

    def test_converse_transformation_anthropic_beta(self):
        """Test that Converse API transformation includes anthropic_beta in additionalModelRequestFields."""
        config = AmazonConverseConfig()
        headers = {"anthropic-beta": "context-1m-2025-08-07,interleaved-thinking-2025-05-14"}
        
        result = config._transform_request_helper(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            system_content_blocks=[],
            optional_params={},
            messages=[{"role": "user", "content": "Test"}],
            headers=headers
        )
        
        assert "additionalModelRequestFields" in result
        additional_fields = result["additionalModelRequestFields"]
        assert "anthropic_beta" in additional_fields
        assert additional_fields["anthropic_beta"] == ["context-1m-2025-08-07", "interleaved-thinking-2025-05-14"]

    @pytest.mark.asyncio
    async def test_messages_transformation_anthropic_beta(self):
        """Test that Messages API transformation includes anthropic_beta in request."""
        config = AmazonAnthropicClaudeMessagesConfig()
        headers = {"anthropic-beta": "output-128k-2025-02-19"}
        
        result = await config.transform_anthropic_messages_request(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": "Test"}],
            anthropic_messages_optional_request_params={"max_tokens": 100},
            litellm_params={},
            headers=headers
        )
        
        assert "anthropic_beta" in result
        assert result["anthropic_beta"] == ["output-128k-2025-02-19"]

    def test_converse_computer_use_compatibility(self):
        """Test that user anthropic_beta headers work with computer use tools."""
        config = AmazonConverseConfig()
        headers = {"anthropic-beta": "context-1m-2025-08-07"}
        
        # Computer use tools should automatically add computer-use-2024-10-22
        tools = [
            {
                "type": "computer_20241022",
                "name": "computer",
                "display_width_px": 1024,
                "display_height_px": 768
            }
        ]
        
        result = config._transform_request_helper(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            system_content_blocks=[],
            optional_params={"tools": tools},
            messages=[{"role": "user", "content": "Test"}],
            headers=headers
        )
        
        additional_fields = result["additionalModelRequestFields"]
        betas = additional_fields["anthropic_beta"]
        
        # Should contain both user-provided and auto-added beta headers
        assert "context-1m-2025-08-07" in betas
        assert "computer-use-2024-10-22" in betas
        assert len(betas) == 2  # No duplicates

    def test_no_anthropic_beta_headers(self):
        """Test that transformations work correctly when no anthropic_beta headers are provided."""
        config = AmazonConverseConfig()
        headers = {}
        
        result = config._transform_request_helper(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            system_content_blocks=[],
            optional_params={},
            messages=[{"role": "user", "content": "Test"}],
            headers=headers
        )
        
        additional_fields = result.get("additionalModelRequestFields", {})
        assert "anthropic_beta" not in additional_fields

    def test_anthropic_beta_all_supported_features(self):
        """Test that all documented beta features are properly handled."""
        supported_features = [
            "context-1m-2025-08-07",
            "computer-use-2025-01-24",
            "computer-use-2024-10-22",
            "token-efficient-tools-2025-02-19",
            "interleaved-thinking-2025-05-14",
            "output-128k-2025-02-19",
            "dev-full-thinking-2025-05-14"
        ]
        
        config = AmazonAnthropicClaudeConfig()
        headers = {"anthropic-beta": ",".join(supported_features)}
        
        result = config.transform_request(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": "Test"}],
            optional_params={},
            litellm_params={},
            headers=headers
        )
        
        assert "anthropic_beta" in result
        assert result["anthropic_beta"] == supported_features

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
    async def test_async_transform_anthropic_messages_request(self):
        """Test that async_transform_anthropic_messages_request uses async image processing."""
        config = AmazonAnthropicClaudeMessagesConfig()
        
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
        
        headers = {"anthropic-beta": "context-1m-2025-08-07"}
        
        # Mock the async_anthropic_provider_process_image_content function
        with patch('litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation.async_anthropic_provider_process_image_content') as mock_async_process:
            mock_async_process.return_value = messages  # Return messages unchanged for simplicity
            
            result = await config.transform_anthropic_messages_request(
                model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                messages=messages,
                anthropic_messages_optional_request_params={"max_tokens": 100},
                litellm_params={},
                headers=headers
            )
            
            # Verify async_anthropic_provider_process_image_content was called
            mock_async_process.assert_called_once_with(messages)
            
            # Verify the result contains expected fields
            assert "anthropic_version" in result
            assert "anthropic_beta" in result
            assert result["anthropic_beta"] == ["context-1m-2025-08-07"]