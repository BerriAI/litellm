"""
Unit tests for Black Forest Labs image edit transformation functionality.
"""

import base64
import json
import os
import sys
import time
from io import BytesIO
from typing import Dict, List
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.black_forest_labs.image_edit.transformation import (
    BlackForestLabsImageEditConfig,
)
from litellm.llms.black_forest_labs.common_utils import BlackForestLabsError
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageObject, ImageResponse


class TestBlackForestLabsImageEditTransformation:
    """
    Unit tests for Black Forest Labs image edit transformation functionality.
    """

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = BlackForestLabsImageEditConfig()
        self.model = "flux-kontext-pro"
        self.logging_obj = MagicMock()
        self.prompt = "Add a red hat to the person in the image"

    def test_get_supported_openai_params(self):
        """Test that supported OpenAI params are returned correctly."""
        params = self.config.get_supported_openai_params(self.model)

        assert "n" in params
        assert "size" in params
        assert "response_format" in params

    def test_map_openai_params_basic(self):
        """Test mapping of OpenAI params to BFL params."""
        optional_params = ImageEditOptionalRequestParams()

        result = self.config.map_openai_params(
            image_edit_optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Should have default output_format
        assert result.get("output_format") == "png"

    def test_map_openai_params_with_bfl_specific(self):
        """Test that BFL-specific params are passed through."""
        # BFL-specific params are passed as dict keys
        optional_params: ImageEditOptionalRequestParams = {
            "seed": 42,
            "safety_tolerance": 2,
            "aspect_ratio": "16:9",
        }

        result = self.config.map_openai_params(
            image_edit_optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result.get("seed") == 42
        assert result.get("safety_tolerance") == 2
        assert result.get("aspect_ratio") == "16:9"
        assert result.get("output_format") == "png"

    def test_validate_environment_with_api_key(self):
        """Test environment validation with provided API key."""
        headers = {}

        result = self.config.validate_environment(
            headers=headers,
            model=self.model,
            api_key="test-api-key",
        )

        assert result["x-key"] == "test-api-key"
        assert result["Content-Type"] == "application/json"
        assert result["Accept"] == "application/json"

    def test_validate_environment_missing_api_key(self):
        """Test that missing API key raises error."""
        headers = {}

        with patch("litellm.llms.black_forest_labs.image_edit.transformation.get_secret_str") as mock_get_secret:
            mock_get_secret.return_value = None

            with pytest.raises(BlackForestLabsError) as exc_info:
                self.config.validate_environment(
                    headers=headers,
                    model=self.model,
                    api_key=None,
                )

            assert exc_info.value.status_code == 401
            assert "BFL_API_KEY is not set" in exc_info.value.message

    def test_get_model_endpoint_kontext_pro(self):
        """Test endpoint resolution for flux-kontext-pro."""
        endpoint = self.config._get_model_endpoint("flux-kontext-pro")
        assert endpoint == "/v1/flux-kontext-pro"

    def test_get_model_endpoint_kontext_max(self):
        """Test endpoint resolution for flux-kontext-max."""
        endpoint = self.config._get_model_endpoint("flux-kontext-max")
        assert endpoint == "/v1/flux-kontext-max"

    def test_get_model_endpoint_with_provider_prefix(self):
        """Test endpoint resolution with provider prefix."""
        endpoint = self.config._get_model_endpoint("black_forest_labs/flux-kontext-pro")
        assert endpoint == "/v1/flux-kontext-pro"

    def test_get_model_endpoint_fill(self):
        """Test endpoint resolution for flux-pro-1.0-fill."""
        endpoint = self.config._get_model_endpoint("flux-pro-1.0-fill")
        assert endpoint == "/v1/flux-pro-1.0-fill"

    def test_get_complete_url(self):
        """Test complete URL generation."""
        url = self.config.get_complete_url(
            model="flux-kontext-pro",
            api_base=None,
            litellm_params={},
        )

        assert url == "https://api.bfl.ai/v1/flux-kontext-pro"

    def test_get_complete_url_custom_base(self):
        """Test complete URL generation with custom base."""
        url = self.config.get_complete_url(
            model="flux-kontext-pro",
            api_base="https://custom.api.com/",
            litellm_params={},
        )

        assert url == "https://custom.api.com/v1/flux-kontext-pro"

    def test_transform_image_edit_request(self):
        """Test request transformation to BFL format."""
        image_data = b"fake_image_data"
        image = BytesIO(image_data)

        image_edit_optional_params = {
            "seed": 123,
            "output_format": "jpeg",
        }

        litellm_params = GenericLiteLLMParams()
        headers = {}

        data, files = self.config.transform_image_edit_request(
            model=self.model,
            prompt=self.prompt,
            image=image,
            image_edit_optional_request_params=image_edit_optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Check that data contains the expected parameters
        assert data["prompt"] == self.prompt
        assert "input_image" in data
        # Verify base64 encoding
        decoded = base64.b64decode(data["input_image"])
        assert decoded == image_data
        assert data["seed"] == 123
        assert data["output_format"] == "jpeg"

        # BFL uses JSON, not multipart - files should be empty
        assert files == []

    def test_transform_image_edit_request_with_mask(self):
        """Test request transformation with mask for inpainting."""
        image_data = b"fake_image_data"
        mask_data = b"fake_mask_data"
        image = BytesIO(image_data)

        image_edit_optional_params = {
            "mask": BytesIO(mask_data),
            "output_format": "png",
        }

        litellm_params = GenericLiteLLMParams()
        headers = {}

        data, files = self.config.transform_image_edit_request(
            model="flux-pro-1.0-fill",
            prompt=self.prompt,
            image=image,
            image_edit_optional_request_params=image_edit_optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Check mask is base64 encoded
        assert "mask" in data
        decoded_mask = base64.b64decode(data["mask"])
        assert decoded_mask == mask_data

    def test_read_image_bytes_from_bytes(self):
        """Test reading image bytes from bytes input."""
        image_data = b"test_image_bytes"
        result = self.config._read_image_bytes(image_data)
        assert result == image_data

    def test_read_image_bytes_from_file_like(self):
        """Test reading image bytes from file-like object."""
        image_data = b"test_image_bytes"
        image = BytesIO(image_data)
        result = self.config._read_image_bytes(image)
        assert result == image_data

    def test_read_image_bytes_from_list(self):
        """Test reading image bytes from list (takes first)."""
        image_data = b"test_image_bytes"
        images = [BytesIO(image_data), BytesIO(b"other")]
        result = self.config._read_image_bytes(images)
        assert result == image_data

    def test_poll_for_result_success(self):
        """Test successful polling."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "Ready",
            "result": {"sample": "https://example.com/image.png"},
        }

        with patch("httpx.get", return_value=mock_response):
            result = self.config._poll_for_result(
                polling_url="https://api.bfl.ai/v1/get_result?id=123",
                api_key="test-key",
                max_wait=10,
                interval=0.1,
            )

        assert result["status"] == "Ready"
        assert result["result"]["sample"] == "https://example.com/image.png"

    def test_poll_for_result_pending_then_ready(self):
        """Test polling that starts pending then becomes ready."""
        pending_response = MagicMock()
        pending_response.status_code = 200
        pending_response.json.return_value = {"status": "Pending"}

        ready_response = MagicMock()
        ready_response.status_code = 200
        ready_response.json.return_value = {
            "status": "Ready",
            "result": {"sample": "https://example.com/image.png"},
        }

        with patch("httpx.get", side_effect=[pending_response, ready_response]):
            result = self.config._poll_for_result(
                polling_url="https://api.bfl.ai/v1/get_result?id=123",
                api_key="test-key",
                max_wait=10,
                interval=0.1,
            )

        assert result["status"] == "Ready"

    def test_poll_for_result_error_status(self):
        """Test polling with error status."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "Error"}

        with patch("httpx.get", return_value=mock_response):
            with pytest.raises(BlackForestLabsError) as exc_info:
                self.config._poll_for_result(
                    polling_url="https://api.bfl.ai/v1/get_result?id=123",
                    api_key="test-key",
                    max_wait=10,
                    interval=0.1,
                )

        assert exc_info.value.status_code == 400
        assert "Error" in exc_info.value.message

    def test_poll_for_result_content_moderated(self):
        """Test polling with content moderated status."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "Content Moderated"}

        with patch("httpx.get", return_value=mock_response):
            with pytest.raises(BlackForestLabsError) as exc_info:
                self.config._poll_for_result(
                    polling_url="https://api.bfl.ai/v1/get_result?id=123",
                    api_key="test-key",
                    max_wait=10,
                    interval=0.1,
                )

        assert exc_info.value.status_code == 400
        assert "Content Moderated" in exc_info.value.message

    def test_poll_for_result_timeout(self):
        """Test polling timeout."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "Pending"}

        with patch("httpx.get", return_value=mock_response):
            with pytest.raises(BlackForestLabsError) as exc_info:
                self.config._poll_for_result(
                    polling_url="https://api.bfl.ai/v1/get_result?id=123",
                    api_key="test-key",
                    max_wait=0.2,
                    interval=0.1,
                )

        assert exc_info.value.status_code == 408
        assert "Timeout" in exc_info.value.message

    def test_poll_for_result_http_error(self):
        """Test polling with HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("httpx.get", return_value=mock_response):
            with pytest.raises(BlackForestLabsError) as exc_info:
                self.config._poll_for_result(
                    polling_url="https://api.bfl.ai/v1/get_result?id=123",
                    api_key="test-key",
                    max_wait=10,
                    interval=0.1,
                )

        assert exc_info.value.status_code == 500

    def test_transform_image_edit_response_success(self):
        """Test successful response transformation."""
        # Create mock initial response with polling URL
        mock_request = MagicMock()
        mock_request.headers = {"x-key": "test-key"}

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "task-123",
            "polling_url": "https://api.bfl.ai/v1/get_result?id=task-123",
        }
        mock_response.request = mock_request
        mock_response.status_code = 200

        # Mock the polling result
        poll_response = MagicMock()
        poll_response.status_code = 200
        poll_response.json.return_value = {
            "status": "Ready",
            "result": {"sample": "https://example.com/edited-image.png"},
        }

        with patch("httpx.get", return_value=poll_response):
            result = self.config.transform_image_edit_response(
                model=self.model,
                raw_response=mock_response,
                logging_obj=self.logging_obj,
            )

        assert isinstance(result, ImageResponse)
        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/edited-image.png"
        assert result.created is not None

    def test_transform_image_edit_response_no_polling_url(self):
        """Test response transformation when polling URL is missing."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "task-123"}  # No polling_url
        mock_response.status_code = 200

        with pytest.raises(BlackForestLabsError) as exc_info:
            self.config.transform_image_edit_response(
                model=self.model,
                raw_response=mock_response,
                logging_obj=self.logging_obj,
            )

        assert exc_info.value.status_code == 500
        assert "No polling_url" in exc_info.value.message

    def test_transform_image_edit_response_api_error(self):
        """Test response transformation with API error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "errors": ["Invalid image format"]
        }
        mock_response.status_code = 400

        with pytest.raises(BlackForestLabsError) as exc_info:
            self.config.transform_image_edit_response(
                model=self.model,
                raw_response=mock_response,
                logging_obj=self.logging_obj,
            )

        assert "Invalid image format" in exc_info.value.message

    def test_transform_image_edit_response_json_parse_error(self):
        """Test response transformation with JSON parse error."""
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.status_code = 500

        with pytest.raises(BlackForestLabsError) as exc_info:
            self.config.transform_image_edit_response(
                model=self.model,
                raw_response=mock_response,
                logging_obj=self.logging_obj,
            )

        assert "Error parsing BFL response" in exc_info.value.message
