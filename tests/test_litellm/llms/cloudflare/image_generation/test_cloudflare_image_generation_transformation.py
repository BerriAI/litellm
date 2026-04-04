"""
Unit tests for Cloudflare Workers AI image generation transformation.

Tests the CloudflareImageGenerationConfig including request/response transformation,
URL construction, parameter mapping, and environment validation.
"""

import base64
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.cloudflare.chat.transformation import CloudflareError
from litellm.llms.cloudflare.image_generation.transformation import (
    CloudflareImageGenerationConfig,
)
from litellm.types.utils import ImageResponse


class TestCloudflareImageGenerationConfig:
    """Test CloudflareImageGenerationConfig parameter handling and transformation."""

    def setup_method(self):
        self.config = CloudflareImageGenerationConfig()
        self.model = "@cf/black-forest-labs/flux-1-schnell"

    def test_get_supported_openai_params(self):
        """Test that supported params include n and size."""
        params = self.config.get_supported_openai_params(self.model)
        assert "n" in params
        assert "size" in params

    def test_map_openai_params_size(self):
        """Test that size string is parsed into width/height."""
        result = self.config.map_openai_params(
            non_default_params={"size": "1024x768"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["width"] == 1024
        assert result["height"] == 768
        assert "size" not in result

    def test_map_openai_params_n(self):
        """Test that n maps to num_images."""
        result = self.config.map_openai_params(
            non_default_params={"n": 3},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["num_images"] == 3
        assert "n" not in result

    def test_map_openai_params_size_without_x_raises(self):
        """Test that size without 'x' separator raises ValueError."""
        with pytest.raises(ValueError, match="Invalid size format"):
            self.config.map_openai_params(
                non_default_params={"size": "large"},
                optional_params={},
                model=self.model,
                drop_params=False,
            )

    def test_map_openai_params_unsupported_raises(self):
        """Test that unsupported param raises ValueError when drop_params=False."""
        with pytest.raises(ValueError, match="not supported"):
            self.config.map_openai_params(
                non_default_params={"quality": "hd"},
                optional_params={},
                model=self.model,
                drop_params=False,
            )

    def test_map_openai_params_unsupported_dropped(self):
        """Test that unsupported param is silently dropped when drop_params=True."""
        result = self.config.map_openai_params(
            non_default_params={"quality": "hd"},
            optional_params={},
            model=self.model,
            drop_params=True,
        )
        assert "quality" not in result

    def test_transform_image_generation_request(self):
        """Test that request body includes prompt and optional params."""
        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt="a beautiful sunset",
            optional_params={"width": 1024, "height": 768},
            litellm_params={},
            headers={},
        )
        assert result["prompt"] == "a beautiful sunset"
        assert result["width"] == 1024
        assert result["height"] == 768

    def test_transform_image_generation_response_binary(self):
        """Test response transformation when Cloudflare returns raw image bytes."""
        image_bytes = b"fake-png-data"
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.headers = {"content-type": "image/png"}
        mock_response.content = image_bytes

        model_response = ImageResponse(data=[])

        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 1
        expected_b64 = base64.b64encode(image_bytes).decode("utf-8")
        assert result.data[0].b64_json == expected_b64

    def test_transform_image_generation_response_json(self):
        """Test response transformation when Cloudflare returns JSON with base64 image."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "result": {"image": "base64-image-data-here"},
            "success": True,
        }

        model_response = ImageResponse(data=[])

        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 1
        assert result.data[0].b64_json == "base64-image-data-here"

    def test_transform_image_generation_response_json_null_result(self):
        """Test response raises error when JSON result is null."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.headers = {"content-type": "application/json"}
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": None,
            "success": False,
        }

        model_response = ImageResponse(data=[])

        with pytest.raises(CloudflareError, match="No image data"):
            self.config.transform_image_generation_response(
                model=self.model,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=MagicMock(),
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

    def test_validate_environment(self):
        """Test that validate_environment sets correct headers."""
        headers = self.config.validate_environment(
            headers={},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-key-123",
        )
        assert headers["Authorization"] == "Bearer test-key-123"
        assert headers["content-type"] == "application/json"

    def test_validate_environment_missing_key(self):
        """Test that validate_environment raises error when key is missing."""
        with pytest.raises(ValueError, match="Missing Cloudflare API Key"):
            self.config.validate_environment(
                headers={},
                model=self.model,
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

    @patch(
        "litellm.llms.cloudflare.image_generation.transformation.get_secret_str"
    )
    def test_get_complete_url(self, mock_get_secret):
        """Test URL construction with default api_base."""
        mock_get_secret.return_value = "test-account-id"
        url = self.config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )
        assert (
            url
            == f"https://api.cloudflare.com/client/v4/accounts/test-account-id/ai/run/{self.model}"
        )

    def test_get_complete_url_with_api_base(self):
        """Test URL construction with explicit api_base."""
        url = self.config.get_complete_url(
            api_base="https://custom.api.com/ai/run/",
            api_key="test-key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )
        assert url == f"https://custom.api.com/ai/run/{self.model}"
