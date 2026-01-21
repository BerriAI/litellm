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
