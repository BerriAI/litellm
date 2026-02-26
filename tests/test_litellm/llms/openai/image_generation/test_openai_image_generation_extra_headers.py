"""
Unit tests for extra_headers support in OpenAI image generation.

Verifies that extra_headers passed to litellm.image_generation() are forwarded
through to the OpenAI SDK's images.generate() call as `extra_headers`.

Ref: https://github.com/BerriAI/litellm/issues/22025
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.openai.openai import OpenAIChatCompletion
from litellm.types.utils import ImageResponse


class TestOpenAIImageGenerationExtraHeaders:
    """Test that extra_headers are passed through to the OpenAI SDK image generation call."""

    def test_image_generation_passes_extra_headers_sync(self):
        """
        Test that headers dict is injected as extra_headers into the
        data dict used for the sync images.generate() call.
        """
        openai_chat_completions = OpenAIChatCompletion()

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "created": 1589478378,
            "data": [{"url": "https://example.com/image.png", "revised_prompt": "test"}],
        }

        mock_openai_client = MagicMock()
        mock_openai_client.images.generate.return_value = mock_response
        mock_openai_client.api_key = "test-key"
        mock_openai_client._base_url = MagicMock()
        mock_openai_client._base_url._uri_reference = "https://api.openai.com"

        extra_headers = {"X-Custom-Header": "custom-value", "X-Another": "another-value"}

        with patch.object(
            openai_chat_completions,
            "_get_openai_client",
            return_value=mock_openai_client,
        ):
            openai_chat_completions.image_generation(
                model="dall-e-3",
                prompt="a white siamese cat",
                timeout=600,
                optional_params={"n": 1, "size": "1024x1024"},
                logging_obj=MagicMock(),
                api_key="test-key",
                api_base="https://api.openai.com",
                model_response=ImageResponse(),
                headers=extra_headers,
            )

        # Verify that images.generate was called with extra_headers in the kwargs
        call_kwargs = mock_openai_client.images.generate.call_args[1]
        assert call_kwargs.get("extra_headers") == extra_headers

    def test_image_generation_no_headers_no_extra_headers(self):
        """
        Test that when headers is None, extra_headers is NOT injected into
        the data dict.
        """
        openai_chat_completions = OpenAIChatCompletion()

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "created": 1589478378,
            "data": [{"url": "https://example.com/image.png", "revised_prompt": "test"}],
        }

        mock_openai_client = MagicMock()
        mock_openai_client.images.generate.return_value = mock_response
        mock_openai_client.api_key = "test-key"
        mock_openai_client._base_url = MagicMock()
        mock_openai_client._base_url._uri_reference = "https://api.openai.com"

        with patch.object(
            openai_chat_completions,
            "_get_openai_client",
            return_value=mock_openai_client,
        ):
            openai_chat_completions.image_generation(
                model="dall-e-3",
                prompt="a white siamese cat",
                timeout=600,
                optional_params={"n": 1, "size": "1024x1024"},
                logging_obj=MagicMock(),
                api_key="test-key",
                api_base="https://api.openai.com",
                model_response=ImageResponse(),
                headers=None,
            )

        # Verify that extra_headers was NOT passed
        call_kwargs = mock_openai_client.images.generate.call_args[1]
        assert "extra_headers" not in call_kwargs

    def test_image_generation_empty_headers_no_extra_headers(self):
        """
        Test that when headers is an empty dict, extra_headers is NOT
        injected into the data dict.
        """
        openai_chat_completions = OpenAIChatCompletion()

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "created": 1589478378,
            "data": [{"url": "https://example.com/image.png", "revised_prompt": "test"}],
        }

        mock_openai_client = MagicMock()
        mock_openai_client.images.generate.return_value = mock_response
        mock_openai_client.api_key = "test-key"
        mock_openai_client._base_url = MagicMock()
        mock_openai_client._base_url._uri_reference = "https://api.openai.com"

        with patch.object(
            openai_chat_completions,
            "_get_openai_client",
            return_value=mock_openai_client,
        ):
            openai_chat_completions.image_generation(
                model="dall-e-3",
                prompt="a white siamese cat",
                timeout=600,
                optional_params={"n": 1, "size": "1024x1024"},
                logging_obj=MagicMock(),
                api_key="test-key",
                api_base="https://api.openai.com",
                model_response=ImageResponse(),
                headers={},
            )

        # Verify that extra_headers was NOT passed (empty dict is falsy)
        call_kwargs = mock_openai_client.images.generate.call_args[1]
        assert "extra_headers" not in call_kwargs

    @pytest.mark.asyncio
    async def test_aimage_generation_passes_extra_headers(self):
        """
        Test that headers dict is injected as extra_headers into the
        data dict used for the async images.generate() call.
        """
        openai_chat_completions = OpenAIChatCompletion()

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "created": 1589478378,
            "data": [{"url": "https://example.com/image.png", "revised_prompt": "test"}],
        }

        mock_openai_aclient = MagicMock()
        mock_openai_aclient.images.generate = AsyncMock(return_value=mock_response)
        mock_openai_aclient.api_key = "test-key"
        mock_openai_aclient._base_url = MagicMock()
        mock_openai_aclient._base_url._uri_reference = "https://api.openai.com"

        extra_headers = {"X-Custom-Header": "custom-value"}

        with patch.object(
            openai_chat_completions,
            "_get_openai_client",
            return_value=mock_openai_aclient,
        ):
            await openai_chat_completions.aimage_generation(
                prompt="a white siamese cat",
                data={"model": "dall-e-3", "prompt": "a white siamese cat", "n": 1},
                model_response=ImageResponse(),
                timeout=600,
                logging_obj=MagicMock(),
                api_key="test-key",
                api_base="https://api.openai.com",
                headers=extra_headers,
            )

        # Verify that images.generate was called with extra_headers
        call_kwargs = mock_openai_aclient.images.generate.call_args[1]
        assert call_kwargs.get("extra_headers") == extra_headers


class TestImageGenerationMainExtraHeaders:
    """
    Test that litellm.image_generation() passes extra_headers through
    to the OpenAI provider path.
    """

    @patch("litellm.images.main.openai_chat_completions")
    def test_image_generation_main_passes_headers_to_openai(
        self, mock_openai_chat_completions
    ):
        """
        Test that the OpenAI branch in images/main.py passes the merged
        headers (including extra_headers) to the OpenAI image_generation call.
        """
        import litellm

        mock_openai_chat_completions.image_generation.return_value = ImageResponse(
            created=1589478378,
            data=[],
        )

        extra_headers = {"X-Custom-Header": "custom-value"}

        litellm.image_generation(
            model="dall-e-3",
            prompt="a white siamese cat",
            extra_headers=extra_headers,
        )

        # Verify the OpenAI provider's image_generation was called with headers
        call_kwargs = mock_openai_chat_completions.image_generation.call_args[1]
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["X-Custom-Header"] == "custom-value"
