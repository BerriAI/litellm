"""
Tests for Vertex AI Anthropic image URL handling.

Issue: https://github.com/BerriAI/litellm/issues/18430
Vertex AI Anthropic models don't support URL sources for images.
LiteLLM should convert image URLs to base64 when using Vertex AI Anthropic.
"""
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.prompt_templates.factory import (
    anthropic_messages_pt,
    convert_to_anthropic_tool_result,
    create_anthropic_image_param,
)


class TestVertexAIAnthropicImageURLHandling:
    """Test that Vertex AI Anthropic converts image URLs to base64."""

    @patch("litellm.litellm_core_utils.prompt_templates.factory.convert_url_to_base64")
    def test_vertex_ai_anthropic_converts_https_url_to_base64(
        self, mock_convert_url: MagicMock
    ):
        """
        Test that HTTPS image URLs are converted to base64 for Vertex AI Anthropic.

        For regular Anthropic, HTTPS URLs are passed through as URL type.
        For Vertex AI Anthropic, HTTPS URLs should be converted to base64.
        """
        mock_convert_url.return_value = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ=="

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.jpg"},
                    },
                ],
            }
        ]

        # For Vertex AI, image URLs should be converted to base64
        result = anthropic_messages_pt(
            messages=messages,
            model="claude-sonnet-4",
            llm_provider="vertex_ai",
        )

        # Verify convert_url_to_base64 was called
        mock_convert_url.assert_called_once_with(url="https://example.com/image.jpg")

        # Check the result has base64 source type
        user_message = result[0]
        assert user_message["role"] == "user"
        image_content = user_message["content"][1]
        assert image_content["type"] == "image"
        assert image_content["source"]["type"] == "base64"

    @patch("litellm.litellm_core_utils.prompt_templates.factory.convert_url_to_base64")
    def test_regular_anthropic_uses_url_type_for_https(
        self, mock_convert_url: MagicMock
    ):
        """
        Test that regular Anthropic API uses URL type for HTTPS images.

        This confirms the original behavior is preserved for non-Vertex AI.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.jpg"},
                    },
                ],
            }
        ]

        # For regular Anthropic, HTTPS URLs should NOT be converted
        result = anthropic_messages_pt(
            messages=messages,
            model="claude-sonnet-4",
            llm_provider="anthropic",
        )

        # convert_url_to_base64 should NOT be called for regular Anthropic with HTTPS
        mock_convert_url.assert_not_called()

        # Check the result has URL source type
        user_message = result[0]
        assert user_message["role"] == "user"
        image_content = user_message["content"][1]
        assert image_content["type"] == "image"
        assert image_content["source"]["type"] == "url"
        assert image_content["source"]["url"] == "https://example.com/image.jpg"

    @patch("litellm.litellm_core_utils.prompt_templates.factory.convert_url_to_base64")
    def test_vertex_ai_beta_also_converts_to_base64(
        self, mock_convert_url: MagicMock
    ):
        """
        Test that vertex_ai_beta provider also converts image URLs to base64.
        """
        mock_convert_url.return_value = "data:image/png;base64,iVBORw0KGgo="

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": "https://example.com/photo.png",
                    },
                ],
            }
        ]

        result = anthropic_messages_pt(
            messages=messages,
            model="claude-3-opus",
            llm_provider="vertex_ai_beta",
        )

        # Verify convert_url_to_base64 was called
        mock_convert_url.assert_called_once()

        # Check the result has base64 source type
        user_message = result[0]
        image_content = user_message["content"][1]
        assert image_content["source"]["type"] == "base64"


class TestCreateAnthropicImageParam:
    """Test the create_anthropic_image_param function directly."""

    @patch("litellm.litellm_core_utils.prompt_templates.factory.convert_url_to_base64")
    def test_force_base64_converts_https_url(self, mock_convert_url: MagicMock):
        """
        Test that is_bedrock_invoke=True (used for both Bedrock and Vertex AI)
        forces conversion of HTTPS URLs to base64.
        """
        mock_convert_url.return_value = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="

        result = create_anthropic_image_param(
            image_url_input="https://example.com/image.jpg",
            format=None,
            is_bedrock_invoke=True,  # This flag is set for both Bedrock and Vertex AI
        )

        mock_convert_url.assert_called_once_with(url="https://example.com/image.jpg")
        assert result["source"]["type"] == "base64"

    @patch("litellm.litellm_core_utils.prompt_templates.factory.convert_url_to_base64")
    def test_no_force_uses_url_type(self, mock_convert_url: MagicMock):
        """
        Test that without force, HTTPS URLs use URL type.
        """
        result = create_anthropic_image_param(
            image_url_input="https://example.com/image.jpg",
            format=None,
            is_bedrock_invoke=False,
        )

        mock_convert_url.assert_not_called()
        assert result["source"]["type"] == "url"
        assert result["source"]["url"] == "https://example.com/image.jpg"


class TestToolMessageImageURLHandling:
    """
    Test that tool messages with image_url are converted to base64 for Vertex AI.

    Issue: https://github.com/BerriAI/litellm/issues/19891
    """

    @patch("litellm.litellm_core_utils.prompt_templates.factory.convert_url_to_base64")
    def test_convert_to_anthropic_tool_result_with_force_base64(
        self, mock_convert_url: MagicMock
    ):
        """
        Test that convert_to_anthropic_tool_result converts image URLs to base64
        when force_base64=True.
        """
        mock_convert_url.return_value = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="

        tool_message = {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/tool_result.jpg"},
                }
            ],
        }

        result = convert_to_anthropic_tool_result(tool_message, force_base64=True)

        mock_convert_url.assert_called_once_with(url="https://example.com/tool_result.jpg")
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "call_123"

        # Check the image content is base64
        content = result["content"]
        assert len(content) == 1
        assert content[0]["type"] == "image"
        assert content[0]["source"]["type"] == "base64"

    @patch("litellm.litellm_core_utils.prompt_templates.factory.convert_url_to_base64")
    def test_convert_to_anthropic_tool_result_without_force_base64(
        self, mock_convert_url: MagicMock
    ):
        """
        Test that convert_to_anthropic_tool_result uses URL type when force_base64=False.
        """
        tool_message = {
            "role": "tool",
            "tool_call_id": "call_456",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"},
                }
            ],
        }

        result = convert_to_anthropic_tool_result(tool_message, force_base64=False)

        mock_convert_url.assert_not_called()
        assert result["type"] == "tool_result"

        # Check the image content uses URL type
        content = result["content"]
        assert len(content) == 1
        assert content[0]["type"] == "image"
        assert content[0]["source"]["type"] == "url"

    @patch("litellm.litellm_core_utils.prompt_templates.factory.convert_url_to_base64")
    def test_vertex_ai_tool_message_converts_image_to_base64(
        self, mock_convert_url: MagicMock
    ):
        """
        Test full conversation with tool result containing image for Vertex AI.
        The image URL should be converted to base64.
        """
        mock_convert_url.return_value = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="

        messages = [
            {
                "role": "user",
                "content": "Get me an image and describe it",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_789",
                        "type": "function",
                        "function": {
                            "name": "get_image",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_789",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/result.jpg"},
                    }
                ],
            },
        ]

        result = anthropic_messages_pt(
            messages=messages,
            model="claude-sonnet-4",
            llm_provider="vertex_ai",
        )

        # Verify convert_url_to_base64 was called for the tool result image
        mock_convert_url.assert_called_once_with(url="https://example.com/result.jpg")

        # Find the tool_result in the converted messages
        for msg in result:
            if msg.get("role") == "user":
                for content_item in msg.get("content", []):
                    if isinstance(content_item, dict) and content_item.get("type") == "tool_result":
                        tool_content = content_item.get("content", [])
                        for item in tool_content:
                            if isinstance(item, dict) and item.get("type") == "image":
                                assert item["source"]["type"] == "base64"
                                return
        pytest.fail("Could not find image in tool result")

    @patch("litellm.litellm_core_utils.prompt_templates.factory.convert_url_to_base64")
    def test_regular_anthropic_tool_message_uses_url(
        self, mock_convert_url: MagicMock
    ):
        """
        Test that regular Anthropic API uses URL type for tool result images.
        """
        messages = [
            {
                "role": "user",
                "content": "Get me an image",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_image",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.jpg"},
                    }
                ],
            },
        ]

        result = anthropic_messages_pt(
            messages=messages,
            model="claude-sonnet-4",
            llm_provider="anthropic",
        )

        # convert_url_to_base64 should NOT be called for regular Anthropic
        mock_convert_url.assert_not_called()

        # Find the tool_result and verify URL type
        for msg in result:
            if msg.get("role") == "user":
                for content_item in msg.get("content", []):
                    if isinstance(content_item, dict) and content_item.get("type") == "tool_result":
                        tool_content = content_item.get("content", [])
                        for item in tool_content:
                            if isinstance(item, dict) and item.get("type") == "image":
                                assert item["source"]["type"] == "url"
                                return
        pytest.fail("Could not find image in tool result")
