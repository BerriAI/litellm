"""Tests for Xinference image edit transformation."""
import io
import json
from typing import Any, Dict
from unittest.mock import Mock, patch

import httpx
import pytest

from litellm.llms.xinference.image_edit.transformation import (
    XInferenceImageEditConfig,
)
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ImageResponse


class TestXInferenceImageEditConfig:
    """Test XInferenceImageEditConfig class."""

    def test_get_supported_openai_params(self):
        """Test that supported OpenAI params are correctly returned."""
        config = XInferenceImageEditConfig()
        supported_params = config.get_supported_openai_params(model="test-model")
        
        expected_params = [
            "image",
            "prompt",
            "mask",
            "n",
            "size",
            "response_format",
        ]
        
        assert supported_params == expected_params

    def test_map_openai_params_with_drop_params_true(self):
        """Test parameter mapping with drop_params=True."""
        config = XInferenceImageEditConfig()
        
        optional_params = ImageEditOptionalRequestParams(
            n=2,
            size="1024x1024",
            response_format="url",
            quality="hd",  # Not supported by Xinference
        )
        
        mapped_params = config.map_openai_params(
            image_edit_optional_params=optional_params,
            model="test-model",
            drop_params=True,
        )
        
        # Should filter out unsupported params when drop_params=True
        assert "n" in mapped_params
        assert "size" in mapped_params
        assert "response_format" in mapped_params
        assert "quality" not in mapped_params

    def test_map_openai_params_with_drop_params_false(self):
        """Test parameter mapping with drop_params=False."""
        config = XInferenceImageEditConfig()
        
        optional_params = ImageEditOptionalRequestParams(
            n=2,
            size="1024x1024",
            response_format="url",
        )
        
        mapped_params = config.map_openai_params(
            image_edit_optional_params=optional_params,
            model="test-model",
            drop_params=False,
        )
        
        # Should include all params when drop_params=False
        assert "n" in mapped_params
        assert "size" in mapped_params
        assert "response_format" in mapped_params

    def test_transform_image_edit_request(self):
        """Test transformation of image edit request to Xinference format."""
        config = XInferenceImageEditConfig()
        
        # Create a mock image file
        image_bytes = b"fake image data"
        image_file = io.BytesIO(image_bytes)
        
        # Create request parameters
        image_edit_params = {
            "n": 1,
            "size": "1024x1024",
            "response_format": "url",
        }
        
        litellm_params = GenericLiteLLMParams()
        headers = {}
        
        # Transform the request
        data, files = config.transform_image_edit_request(
            model="xinference/test-model",
            prompt="Test prompt",
            image=image_file,
            image_edit_optional_request_params=image_edit_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        
        # Verify data parameters
        assert data["model"] == "xinference/test-model"
        assert data["prompt"] == "Test prompt"
        assert data["n"] == 1
        assert data["size"] == "1024x1024"
        assert data["response_format"] == "url"
        
        # Verify files
        assert len(files) == 1
        assert files[0][0] == "image"

    def test_transform_image_edit_request_with_mask(self):
        """Test transformation of image edit request with mask."""
        config = XInferenceImageEditConfig()
        
        # Create mock image and mask files
        image_bytes = b"fake image data"
        mask_bytes = b"fake mask data"
        image_file = io.BytesIO(image_bytes)
        mask_file = io.BytesIO(mask_bytes)
        
        # Create request parameters with mask
        image_edit_params = {
            "mask": mask_file,
            "n": 1,
        }
        
        litellm_params = GenericLiteLLMParams()
        headers = {}
        
        # Transform the request
        data, files = config.transform_image_edit_request(
            model="xinference/test-model",
            prompt="Test prompt with mask",
            image=image_file,
            image_edit_optional_request_params=image_edit_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        
        # Verify files include both image and mask
        assert len(files) == 2
        file_names = [f[0] for f in files]
        assert "image" in file_names
        assert "mask" in file_names

    def test_transform_image_edit_response_success(self):
        """Test successful transformation of image edit response."""
        config = XInferenceImageEditConfig()
        
        # Create mock response
        response_data = {
            "created": 1234567890,
            "data": [
                {
                    "url": "https://example.com/image1.png",
                    "b64_json": None,
                },
                {
                    "url": "https://example.com/image2.png",
                    "b64_json": None,
                }
            ]
        }
        
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        
        logging_obj = Mock()
        
        # Transform the response
        result = config.transform_image_edit_response(
            model="test-model",
            raw_response=mock_response,
            logging_obj=logging_obj,
        )
        
        # Verify the result
        assert isinstance(result, ImageResponse)
        assert len(result.data) == 2
        assert result.data[0].url == "https://example.com/image1.png"
        assert result.data[1].url == "https://example.com/image2.png"

    def test_transform_image_edit_response_error(self):
        """Test error handling in response transformation."""
        config = XInferenceImageEditConfig()
        
        # Create mock error response
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.side_effect = Exception("Invalid JSON")
        mock_response.text = "Error message"
        mock_response.status_code = 400
        mock_response.headers = {}
        
        logging_obj = Mock()
        
        # Should raise an error
        with pytest.raises(Exception):
            config.transform_image_edit_response(
                model="test-model",
                raw_response=mock_response,
                logging_obj=logging_obj,
            )

    def test_validate_environment_with_api_key(self):
        """Test environment validation with API key."""
        config = XInferenceImageEditConfig()
        headers = {}
        
        validated_headers = config.validate_environment(
            headers=headers,
            model="test-model",
            api_key="test-api-key",
        )
        
        assert "Authorization" in validated_headers
        assert validated_headers["Authorization"] == "Bearer test-api-key"

    def test_validate_environment_without_api_key(self):
        """Test environment validation without API key."""
        config = XInferenceImageEditConfig()
        headers = {}
        
        validated_headers = config.validate_environment(
            headers=headers,
            model="test-model",
            api_key=None,
        )
        
        # Should not add Authorization header if no API key
        assert "Authorization" not in validated_headers

    def test_validate_environment_with_anything_api_key(self):
        """Test environment validation with 'anything' API key (common for Xinference)."""
        config = XInferenceImageEditConfig()
        headers = {}
        
        validated_headers = config.validate_environment(
            headers=headers,
            model="test-model",
            api_key="anything",
        )
        
        # Should not add Authorization header for 'anything' key
        assert "Authorization" not in validated_headers

    def test_get_complete_url_default(self):
        """Test URL construction with default base."""
        config = XInferenceImageEditConfig()
        
        url = config.get_complete_url(
            model="test-model",
            api_base=None,
            litellm_params={},
        )
        
        assert url == "http://127.0.0.1:9997/v1/images/edits"

    def test_get_complete_url_custom_base(self):
        """Test URL construction with custom base."""
        config = XInferenceImageEditConfig()
        
        url = config.get_complete_url(
            model="test-model",
            api_base="http://custom-host:8000/v1",
            litellm_params={},
        )
        
        assert url == "http://custom-host:8000/v1/images/edits"

    def test_get_complete_url_trailing_slash(self):
        """Test URL construction handles trailing slashes correctly."""
        config = XInferenceImageEditConfig()
        
        url = config.get_complete_url(
            model="test-model",
            api_base="http://custom-host:8000/v1/",
            litellm_params={},
        )
        
        # Should remove trailing slash
        assert url == "http://custom-host:8000/v1/images/edits"
