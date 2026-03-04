import base64
import json
import os
import sys
from io import BytesIO
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.llms.openrouter.image_edit.transformation import (
    OpenRouterImageEditConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse


class TestOpenRouterImageEditTransformation:
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = OpenRouterImageEditConfig()
        self.model = "google/gemini-2.5-flash-image"
        self.logging_obj = MagicMock()
        self.sample_image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    def test_get_supported_openai_params(self):
        """Test that get_supported_openai_params returns correct parameters."""
        supported_params = self.config.get_supported_openai_params(self.model)

        assert "size" in supported_params
        assert "quality" in supported_params
        assert "n" in supported_params
        assert len(supported_params) == 3

    def test_use_multipart_form_data_returns_false(self):
        """Test that OpenRouter uses JSON, not multipart/form-data."""
        assert self.config.use_multipart_form_data() is False

    # Parameter mapping tests

    def test_map_openai_params_size(self):
        """Test that size is mapped to image_config.aspect_ratio."""
        result = self.config.map_openai_params(
            image_edit_optional_params={"size": "1024x1024"},
            model=self.model,
            drop_params=False,
        )

        assert "image_config" in result
        assert result["image_config"]["aspect_ratio"] == "1:1"

    def test_map_openai_params_quality(self):
        """Test that quality is mapped to image_config.image_size."""
        result = self.config.map_openai_params(
            image_edit_optional_params={"quality": "high"},
            model=self.model,
            drop_params=False,
        )

        assert "image_config" in result
        assert result["image_config"]["image_size"] == "4K"

    def test_map_openai_params_size_and_quality(self):
        """Test that both size and quality are mapped correctly."""
        result = self.config.map_openai_params(
            image_edit_optional_params={"size": "1792x1024", "quality": "hd"},
            model=self.model,
            drop_params=False,
        )

        assert result["image_config"]["aspect_ratio"] == "16:9"
        assert result["image_config"]["image_size"] == "4K"

    def test_map_openai_params_n_passthrough(self):
        """Test that n parameter is passed through directly."""
        result = self.config.map_openai_params(
            image_edit_optional_params={"n": 2},
            model=self.model,
            drop_params=False,
        )

        assert result["n"] == 2

    def test_map_openai_params_unknown_quality_ignored(self):
        """Test that unknown quality values produce no image_size mapping."""
        result = self.config.map_openai_params(
            image_edit_optional_params={"quality": "unknown_value"},
            model=self.model,
            drop_params=False,
        )

        assert "image_config" not in result

    # Size-to-aspect-ratio mapping tests

    def test_map_size_to_aspect_ratio_square(self):
        """Test mapping square sizes to 1:1 aspect ratio."""
        assert self.config._map_size_to_aspect_ratio("256x256") == "1:1"
        assert self.config._map_size_to_aspect_ratio("512x512") == "1:1"
        assert self.config._map_size_to_aspect_ratio("1024x1024") == "1:1"

    def test_map_size_to_aspect_ratio_landscape(self):
        """Test mapping landscape sizes to correct aspect ratios."""
        assert self.config._map_size_to_aspect_ratio("1536x1024") == "3:2"
        assert self.config._map_size_to_aspect_ratio("1792x1024") == "16:9"

    def test_map_size_to_aspect_ratio_portrait(self):
        """Test mapping portrait sizes to correct aspect ratios."""
        assert self.config._map_size_to_aspect_ratio("1024x1536") == "2:3"
        assert self.config._map_size_to_aspect_ratio("1024x1792") == "9:16"

    def test_map_size_to_aspect_ratio_unknown_defaults_to_1_1(self):
        """Test that unknown size defaults to 1:1."""
        assert self.config._map_size_to_aspect_ratio("999x999") == "1:1"

    # Quality-to-image-size mapping tests

    def test_map_quality_to_image_size(self):
        """Test quality to image size mappings."""
        assert self.config._map_quality_to_image_size("low") == "1K"
        assert self.config._map_quality_to_image_size("standard") == "1K"
        assert self.config._map_quality_to_image_size("auto") == "1K"
        assert self.config._map_quality_to_image_size("medium") == "2K"
        assert self.config._map_quality_to_image_size("high") == "4K"
        assert self.config._map_quality_to_image_size("hd") == "4K"

    def test_map_quality_to_image_size_unknown_returns_none(self):
        """Test that unknown quality returns None."""
        assert self.config._map_quality_to_image_size("unknown") is None

    # URL tests

    def test_get_complete_url_default(self):
        """Test that default URL is OpenRouter chat completions endpoint."""
        result = self.config.get_complete_url(
            model=self.model,
            api_base=None,
            litellm_params={},
        )

        assert result == "https://openrouter.ai/api/v1/chat/completions"

    def test_get_complete_url_with_custom_base(self):
        """Test that custom api_base gets /chat/completions appended."""
        result = self.config.get_complete_url(
            model=self.model,
            api_base="https://custom.openrouter.ai/api/v1",
            litellm_params={},
        )

        assert result == "https://custom.openrouter.ai/api/v1/chat/completions"

    def test_get_complete_url_with_complete_base(self):
        """Test that api_base already ending in /chat/completions is not duplicated."""
        url = "https://custom.openrouter.ai/api/v1/chat/completions"
        result = self.config.get_complete_url(
            model=self.model,
            api_base=url,
            litellm_params={},
        )

        assert result == url

    # Validate environment tests

    @patch("litellm.llms.openrouter.image_edit.transformation.get_secret_str")
    def test_validate_environment_with_api_key(self, mock_get_secret):
        """Test that validate_environment sets authorization header with provided key."""
        headers = {}
        result = self.config.validate_environment(
            headers=headers,
            model=self.model,
            api_key="test_api_key",
        )

        assert result["Authorization"] == "Bearer test_api_key"
        mock_get_secret.assert_not_called()

    @patch("litellm.llms.openrouter.image_edit.transformation.get_secret_str")
    def test_validate_environment_with_secret_key(self, mock_get_secret):
        """Test that validate_environment falls back to secret key."""
        mock_get_secret.return_value = "secret_api_key"
        headers = {}
        result = self.config.validate_environment(
            headers=headers,
            model=self.model,
            api_key=None,
        )

        assert result["Authorization"] == "Bearer secret_api_key"

    @patch("litellm.llms.openrouter.image_edit.transformation.litellm")
    @patch("litellm.llms.openrouter.image_edit.transformation.get_secret_str")
    def test_validate_environment_missing_api_key_raises(self, mock_get_secret, mock_litellm):
        """Test that validate_environment raises ValueError when no API key is available."""
        mock_get_secret.return_value = None
        mock_litellm.api_key = None

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not set"):
            self.config.validate_environment(
                headers={},
                model=self.model,
                api_key=None,
            )

    # Request transformation tests

    def test_transform_image_edit_request_basic(self):
        """Test basic request transformation with image and prompt."""
        data, files = self.config.transform_image_edit_request(
            model=self.model,
            prompt="Add a sunset to this image",
            image=self.sample_image_bytes,
            image_edit_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["model"] == self.model
        assert data["modalities"] == ["image", "text"]
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "user"

        content = data["messages"][0]["content"]
        assert len(content) == 2

        # First content part should be the image
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")

        # Second content part should be the text prompt
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "Add a sunset to this image"

        # Files should be empty (JSON mode)
        assert list(files) == []

    def test_transform_image_edit_request_with_bytesio(self):
        """Test request transformation with BytesIO image input."""
        image = BytesIO(self.sample_image_bytes)
        data, files = self.config.transform_image_edit_request(
            model=self.model,
            prompt="Edit this",
            image=image,
            image_edit_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        content = data["messages"][0]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_transform_image_edit_request_with_multiple_images(self):
        """Test request transformation with a list of images."""
        images = [self.sample_image_bytes, self.sample_image_bytes]
        data, files = self.config.transform_image_edit_request(
            model=self.model,
            prompt="Combine these images",
            image=images,
            image_edit_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        content = data["messages"][0]["content"]
        # Two image parts + one text part
        assert len(content) == 3
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "text"

    def test_transform_image_edit_request_with_optional_params(self):
        """Test that optional params are included in request body."""
        data, files = self.config.transform_image_edit_request(
            model=self.model,
            prompt="Edit this",
            image=self.sample_image_bytes,
            image_edit_optional_request_params={
                "image_config": {"aspect_ratio": "16:9"},
                "n": 2,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["image_config"]["aspect_ratio"] == "16:9"
        assert data["n"] == 2

    def test_transform_image_edit_request_base64_encoding(self):
        """Test that image bytes are correctly base64-encoded in the request."""
        raw_bytes = b"test_image_data"
        expected_b64 = base64.b64encode(raw_bytes).decode("utf-8")

        data, _ = self.config.transform_image_edit_request(
            model=self.model,
            prompt="Edit",
            image=raw_bytes,
            image_edit_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        image_url = data["messages"][0]["content"][0]["image_url"]["url"]
        # Extract the base64 part after the data URL prefix
        b64_part = image_url.split(",", 1)[1]
        assert b64_part == expected_b64

    def test_transform_image_edit_request_no_prompt(self):
        """Test request transformation with no prompt (image-only)."""
        data, _ = self.config.transform_image_edit_request(
            model=self.model,
            prompt=None,
            image=self.sample_image_bytes,
            image_edit_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        content = data["messages"][0]["content"]
        # Only image, no text part
        assert len(content) == 1
        assert content[0]["type"] == "image_url"

    # Response transformation tests

    def test_transform_image_edit_response_with_base64(self):
        """Test response transformation with base64 image data."""
        response_data = {
            "choices": [{
                "message": {
                    "content": "Here is the edited image.",
                    "role": "assistant",
                    "images": [{
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANS"},
                        "type": "image_url"
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 300,
                "completion_tokens": 1299,
                "total_tokens": 1599,
                "completion_tokens_details": {"image_tokens": 1290},
                "cost": 0.05
            },
            "model": self.model
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        result = self.config.transform_image_edit_response(
            model=self.model,
            raw_response=mock_response,
            logging_obj=self.logging_obj,
        )

        assert len(result.data) == 1
        assert result.data[0].b64_json == "iVBORw0KGgoAAAANS"
        assert result.data[0].url is None

    def test_transform_image_edit_response_with_url(self):
        """Test response transformation with URL image data."""
        response_data = {
            "choices": [{
                "message": {
                    "content": "Edited.",
                    "role": "assistant",
                    "images": [{
                        "image_url": {"url": "https://example.com/edited.png"},
                        "type": "image_url"
                    }]
                }
            }],
            "usage": {"prompt_tokens": 10, "total_tokens": 1310},
            "model": self.model
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        result = self.config.transform_image_edit_response(
            model=self.model,
            raw_response=mock_response,
            logging_obj=self.logging_obj,
        )

        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/edited.png"
        assert result.data[0].b64_json is None

    def test_transform_image_edit_response_usage_and_cost(self):
        """Test that usage and cost are correctly extracted from response."""
        response_data = {
            "choices": [{
                "message": {
                    "content": "Edited.",
                    "role": "assistant",
                    "images": [{
                        "image_url": {"url": "data:image/png;base64,abc123"},
                        "type": "image_url"
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 300,
                "completion_tokens": 1299,
                "total_tokens": 1599,
                "completion_tokens_details": {"image_tokens": 1290},
                "prompt_tokens_details": {"image_tokens": 258},
                "cost": 0.05,
                "cost_details": {"input_cost": 0.01, "output_cost": 0.04}
            },
            "model": self.model
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        result = self.config.transform_image_edit_response(
            model=self.model,
            raw_response=mock_response,
            logging_obj=self.logging_obj,
        )

        # Check usage
        assert result.usage is not None
        assert result.usage.input_tokens == 300
        assert result.usage.output_tokens == 1290
        assert result.usage.total_tokens == 1599
        assert result.usage.input_tokens_details.image_tokens == 258
        assert result.usage.input_tokens_details.text_tokens == 42

        # Check cost
        assert result._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == 0.05

        # Check cost details
        assert result._hidden_params["response_cost_details"]["input_cost"] == 0.01
        assert result._hidden_params["response_cost_details"]["output_cost"] == 0.04

        # Check model
        assert result._hidden_params["model"] == self.model

    def test_transform_image_edit_response_multiple_images(self):
        """Test response transformation with multiple output images."""
        response_data = {
            "choices": [{
                "message": {
                    "content": "Here are your edits.",
                    "role": "assistant",
                    "images": [
                        {
                            "image_url": {"url": "data:image/png;base64,img1data"},
                            "type": "image_url"
                        },
                        {
                            "image_url": {"url": "data:image/png;base64,img2data"},
                            "type": "image_url"
                        }
                    ]
                }
            }],
            "usage": {"prompt_tokens": 300, "total_tokens": 2600},
            "model": self.model
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        result = self.config.transform_image_edit_response(
            model=self.model,
            raw_response=mock_response,
            logging_obj=self.logging_obj,
        )

        assert len(result.data) == 2
        assert result.data[0].b64_json == "img1data"
        assert result.data[1].b64_json == "img2data"

    def test_transform_image_edit_response_json_error(self):
        """Test that invalid JSON response raises OpenRouterException."""
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.status_code = 500
        mock_response.headers = {}

        with pytest.raises(OpenRouterException) as exc_info:
            self.config.transform_image_edit_response(
                model=self.model,
                raw_response=mock_response,
                logging_obj=self.logging_obj,
            )

        assert "Error parsing OpenRouter response" in str(exc_info.value)
        assert exc_info.value.status_code == 500

    def test_get_error_class(self):
        """Test that get_error_class returns OpenRouterException."""
        error = self.config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={"Content-Type": "application/json"},
        )

        assert isinstance(error, OpenRouterException)
        assert error.status_code == 400

    # Read image bytes tests

    def test_read_image_bytes_from_bytes(self):
        """Test reading bytes directly."""
        result = self.config._read_image_bytes(b"raw_bytes")
        assert result == b"raw_bytes"

    def test_read_image_bytes_from_bytesio(self):
        """Test reading bytes from BytesIO."""
        bio = BytesIO(b"bytesio_data")
        bio.seek(5)  # Move position to test seek reset
        result = self.config._read_image_bytes(bio)
        assert result == b"bytesio_data"
        assert bio.tell() == 5  # Position should be restored

    def test_read_image_bytes_unsupported_type(self):
        """Test that unsupported image type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported image type"):
            self.config._read_image_bytes("not_an_image")  # type: ignore
