import json
import os
import sys
from typing import List, Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.recraft.image_generation.transformation import (
    RecraftImageGenerationConfig,
)
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageObject, ImageResponse


class TestRecraftImageGenerationTransformation:
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = RecraftImageGenerationConfig()
        self.model = "recraft-v3"
        self.logging_obj = MagicMock()


    def test_map_openai_params_supported_params(self):
        """Test that map_openai_params correctly maps supported parameters."""
        non_default_params = {
            "n": 2,
            "response_format": "url",
            "size": "1024x1024",
            "style": "photographic"
        }
        optional_params = {}
        
        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False
        )
        
        assert result == non_default_params
        
    def test_map_openai_params_unsupported_param_drop_true(self):
        """Test that map_openai_params drops unsupported parameters when drop_params=True."""
        non_default_params = {
            "n": 2,
            "unsupported_param": "value"
        }
        optional_params = {}
        
        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=True
        )
        
        assert result == {"n": 2}
        assert "unsupported_param" not in result
        
    def test_map_openai_params_unsupported_param_drop_false(self):
        """Test that map_openai_params raises ValueError for unsupported parameters when drop_params=False."""
        non_default_params = {
            "n": 2,
            "unsupported_param": "value"
        }
        optional_params = {}
        
        with pytest.raises(ValueError) as exc_info:
            self.config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=self.model,
                drop_params=False
            )
        
        assert "unsupported_param" in str(exc_info.value)
        assert "is not supported for model" in str(exc_info.value)

    @patch("litellm.llms.recraft.image_generation.transformation.get_secret_str")
    def test_get_complete_url_with_api_base(self, mock_get_secret):
        """Test that get_complete_url returns correct URL when api_base is provided."""
        api_base = "https://custom.api.recraft.ai"
        
        result = self.config.get_complete_url(
            api_base=api_base,
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={}
        )
        
        expected_url = f"{api_base}/{self.config.IMAGE_GENERATION_ENDPOINT}"
        assert result == expected_url
        mock_get_secret.assert_not_called()

    @patch("litellm.llms.recraft.image_generation.transformation.get_secret_str")
    def test_get_complete_url_with_secret_base(self, mock_get_secret):
        """Test that get_complete_url uses secret when api_base is None."""
        mock_get_secret.return_value = "https://secret.api.recraft.ai"
        
        result = self.config.get_complete_url(
            api_base=None,
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={}
        )
        
        expected_url = f"https://secret.api.recraft.ai/{self.config.IMAGE_GENERATION_ENDPOINT}"
        assert result == expected_url
        mock_get_secret.assert_called_once_with("RECRAFT_API_BASE")

    @patch("litellm.llms.recraft.image_generation.transformation.get_secret_str")
    def test_get_complete_url_with_default_base(self, mock_get_secret):
        """Test that get_complete_url uses default base URL when no other options are available."""
        mock_get_secret.return_value = None
        
        result = self.config.get_complete_url(
            api_base=None,
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={}
        )
        
        expected_url = f"{self.config.DEFAULT_BASE_URL}/{self.config.IMAGE_GENERATION_ENDPOINT}"
        assert result == expected_url

    @patch("litellm.llms.recraft.image_generation.transformation.get_secret_str")
    def test_validate_environment_with_api_key(self, mock_get_secret):
        """Test that validate_environment correctly sets authorization header when api_key is provided."""
        headers = {}
        api_key = "test_api_key"
        
        result = self.config.validate_environment(
            headers=headers,
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=api_key
        )
        
        assert result["Authorization"] == f"Bearer {api_key}"
        mock_get_secret.assert_not_called()

    @patch("litellm.llms.recraft.image_generation.transformation.get_secret_str")
    def test_validate_environment_with_secret_key(self, mock_get_secret):
        """Test that validate_environment uses secret API key when api_key is None."""
        mock_get_secret.return_value = "secret_api_key"
        headers = {}
        
        result = self.config.validate_environment(
            headers=headers,
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None
        )
        
        assert result["Authorization"] == "Bearer secret_api_key"
        mock_get_secret.assert_called_once_with("RECRAFT_API_KEY")

    @patch("litellm.llms.recraft.image_generation.transformation.get_secret_str")
    def test_validate_environment_no_api_key_raises_error(self, mock_get_secret):
        """Test that validate_environment raises ValueError when no API key is available."""
        mock_get_secret.return_value = None
        headers = {}
        
        with pytest.raises(ValueError) as exc_info:
            self.config.validate_environment(
                headers=headers,
                model=self.model,
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None
            )
        
        assert "RECRAFT_API_KEY is not set" in str(exc_info.value)

    def test_transform_image_generation_request(self):
        """Test that transform_image_generation_request correctly transforms request parameters."""
        prompt = "A beautiful sunset over mountains"
        optional_params = {
            "n": 2,
            "size": "1024x1024",
            "style": "photographic"
        }
        litellm_params = {}
        headers = {}
        
        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers
        )
        
        assert result["prompt"] == prompt
        assert result["model"] == self.model
        assert result["n"] == 2
        assert result["size"] == "1024x1024"
        assert result["style"] == "photographic"

    def test_transform_image_generation_response_success(self):
        """Test that transform_image_generation_response correctly transforms successful response."""
        # Mock response data
        response_data = {
            "data": [
                {"url": "https://example.com/image1.jpg", "b64_json": None},
                {"url": None, "b64_json": "base64encodeddata"}
            ]
        }
        
        # Create mock response
        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        
        # Create empty model response
        model_response = ImageResponse(data=[])
        
        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None
        )
        
        assert len(result.data) == 2
        assert result.data[0].url == "https://example.com/image1.jpg"
        assert result.data[0].b64_json is None
        assert result.data[1].url is None
        assert result.data[1].b64_json == "base64encodeddata"

    def test_transform_image_generation_response_json_error(self):
        """Test that transform_image_generation_response raises error when response JSON is invalid."""
        # Create mock response that raises JSON decode error
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.status_code = 500
        mock_response.headers = {}
        
        model_response = ImageResponse(data=[])
        
        with pytest.raises(Exception) as exc_info:
            self.config.transform_image_generation_response(
                model=self.model,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None
            )
        
        assert "Error transforming image generation response" in str(exc_info.value) 