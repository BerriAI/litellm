"""
Unit tests for extra_headers propagation in OpenAI image generation.

Verifies that extra_headers passed to litellm.image_generation() /
litellm.aimage_generation() are forwarded to the OpenAI API client as
extra_headers in the images.generate() call.
"""

import os
import sys
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.openai.openai import OpenAIChatCompletion


@pytest.fixture
def openai_chat_completions():
    return OpenAIChatCompletion()


@pytest.fixture
def mock_logging_obj():
    logging_obj = MagicMock()
    logging_obj.pre_call = MagicMock()
    logging_obj.post_call = MagicMock()
    return logging_obj


class TestImageGenerationExtraHeaders:
    """Test that extra_headers are properly injected into OpenAI image generation calls."""

    def test_sync_image_generation_with_headers(
        self, openai_chat_completions, mock_logging_obj
    ):
        """Sync image_generation should pass headers as extra_headers to images.generate()."""
        mock_image_data = MagicMock()
        mock_image_data.model_dump.return_value = {
            "created": 1700000000,
            "data": [{"url": "https://example.com/image.png"}],
        }

        mock_openai_client = MagicMock()
        mock_openai_client.images.generate.return_value = mock_image_data
        mock_openai_client.api_key = "test-key"
        mock_openai_client._base_url._uri_reference = "https://api.openai.com"

        test_headers = {"cf-aig-authorization": "Bearer custom-token"}

        openai_chat_completions.image_generation(
            model="dall-e-3",
            prompt="A white cat",
            timeout=60.0,
            optional_params={},
            logging_obj=mock_logging_obj,
            api_key="test-key",
            headers=test_headers,
            client=mock_openai_client,
        )

        _, kwargs = mock_openai_client.images.generate.call_args
        assert kwargs.get("extra_headers") == test_headers

    def test_sync_image_generation_without_headers(
        self, openai_chat_completions, mock_logging_obj
    ):
        """Sync image_generation without headers should not inject extra_headers."""
        mock_image_data = MagicMock()
        mock_image_data.model_dump.return_value = {
            "created": 1700000000,
            "data": [{"url": "https://example.com/image.png"}],
        }

        mock_openai_client = MagicMock()
        mock_openai_client.images.generate.return_value = mock_image_data
        mock_openai_client.api_key = "test-key"
        mock_openai_client._base_url._uri_reference = "https://api.openai.com"

        openai_chat_completions.image_generation(
            model="dall-e-3",
            prompt="A white cat",
            timeout=60.0,
            optional_params={},
            logging_obj=mock_logging_obj,
            api_key="test-key",
            client=mock_openai_client,
        )

        _, kwargs = mock_openai_client.images.generate.call_args
        assert "extra_headers" not in kwargs

    @pytest.mark.asyncio
    async def test_async_image_generation_with_headers(
        self, openai_chat_completions, mock_logging_obj
    ):
        """Async aimage_generation should pass headers as extra_headers to images.generate()."""
        mock_image_data = MagicMock()
        mock_image_data.model_dump.return_value = {
            "created": 1700000000,
            "data": [{"url": "https://example.com/image.png"}],
        }

        mock_openai_client = MagicMock()
        mock_openai_client.images.generate = AsyncMock(return_value=mock_image_data)
        mock_openai_client.api_key = "test-key"

        test_headers = {"cf-aig-authorization": "Bearer custom-token"}

        await openai_chat_completions.aimage_generation(
            prompt="A white cat",
            data={"model": "dall-e-3", "prompt": "A white cat"},
            model_response=MagicMock(),
            timeout=60.0,
            logging_obj=mock_logging_obj,
            api_key="test-key",
            headers=test_headers,
            client=mock_openai_client,
        )

        _, kwargs = mock_openai_client.images.generate.call_args
        assert kwargs.get("extra_headers") == test_headers

    @pytest.mark.asyncio
    async def test_async_image_generation_without_headers(
        self, openai_chat_completions, mock_logging_obj
    ):
        """Async aimage_generation without headers should not inject extra_headers."""
        mock_image_data = MagicMock()
        mock_image_data.model_dump.return_value = {
            "created": 1700000000,
            "data": [{"url": "https://example.com/image.png"}],
        }

        mock_openai_client = MagicMock()
        mock_openai_client.images.generate = AsyncMock(return_value=mock_image_data)
        mock_openai_client.api_key = "test-key"

        await openai_chat_completions.aimage_generation(
            prompt="A white cat",
            data={"model": "dall-e-3", "prompt": "A white cat"},
            model_response=MagicMock(),
            timeout=60.0,
            logging_obj=mock_logging_obj,
            api_key="test-key",
            client=mock_openai_client,
        )

        _, kwargs = mock_openai_client.images.generate.call_args
        assert "extra_headers" not in kwargs

    def test_sync_image_generation_forwards_headers_to_async(
        self, openai_chat_completions, mock_logging_obj
    ):
        """When aimg_generation=True, image_generation should forward headers to aimage_generation."""
        with patch.object(
            openai_chat_completions, "aimage_generation"
        ) as mock_aimage_gen:
            mock_aimage_gen.return_value = MagicMock()

            test_headers = {"x-custom-header": "value"}

            openai_chat_completions.image_generation(
                model="dall-e-3",
                prompt="A white cat",
                timeout=60.0,
                optional_params={},
                logging_obj=mock_logging_obj,
                api_key="test-key",
                aimg_generation=True,
                headers=test_headers,
            )

            mock_aimage_gen.assert_called_once()
            call_kwargs = mock_aimage_gen.call_args[1]
            assert call_kwargs["headers"] == test_headers


class TestImageGenerationEntryPointHeaders:
    """Test that litellm.image_generation() passes headers through to the OpenAI provider."""

    @pytest.mark.asyncio
    async def test_extra_headers_reach_openai_provider(self):
        """End-to-end: extra_headers from litellm.aimage_generation() reach OpenAI images.generate()."""
        import litellm

        mock_image_data = MagicMock()
        mock_image_data.model_dump.return_value = {
            "created": 1700000000,
            "data": [{"url": "https://example.com/image.png"}],
        }

        mock_openai_client = MagicMock()
        mock_openai_client.images.generate = AsyncMock(return_value=mock_image_data)
        mock_openai_client.api_key = "test-key"
        mock_openai_client._base_url._uri_reference = "https://api.openai.com"

        test_headers = {"cf-aig-authorization": "Bearer my-secret"}

        await litellm.aimage_generation(
            model="dall-e-3",
            prompt="A white cat",
            extra_headers=test_headers,
            client=mock_openai_client,
            api_key="test-key",
        )

        mock_openai_client.images.generate.assert_called_once()
        _, kwargs = mock_openai_client.images.generate.call_args
        assert kwargs.get("extra_headers") == test_headers
