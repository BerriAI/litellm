"""
Unit tests for ModelScope image generation configuration.

These tests validate the ModelScopeImageGenerationConfig class which handles
transformation between OpenAI-compatible format and ModelScope API format.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.modelscope.image_generation.transformation import (
    ModelScopeImageGenerationConfig,
)
from litellm.types.utils import ImageResponse


class TestModelScopeImageGenerationTransformation:
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = ModelScopeImageGenerationConfig()
        self.model = "modelscope/Qwen/Qwen-Image-Edit"
        self.logging_obj = MagicMock()

    def test_get_supported_openai_params(self):
        """Test that get_supported_openai_params returns correct parameters."""
        supported_params = self.config.get_supported_openai_params(self.model)

        assert "n" in supported_params
        assert "size" in supported_params
        assert "response_format" in supported_params
        assert "user" in supported_params

    def test_map_openai_params(self):
        """Test that map_openai_params correctly passes through parameters."""
        non_default_params = {
            "n": 2,
            "size": "1024x1024",
            "response_format": "url",
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["n"] == 2
        assert result["size"] == "1024x1024"
        assert result["response_format"] == "url"

    def test_map_openai_params_with_user(self):
        """Test that map_openai_params correctly passes through user parameter."""
        non_default_params = {"user": "test-user-123"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["user"] == "test-user-123"

    def test_get_complete_url_default(self):
        """Test that get_complete_url returns default ModelScope URL."""
        result = self.config.get_complete_url(
            api_base=None,
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert result == "https://api-inference.modelscope.cn/v1/images/generations"

    def test_get_complete_url_with_custom_base(self):
        """Test that get_complete_url uses custom api_base."""
        custom_base = "https://custom.modelscope.cn/v1"

        result = self.config.get_complete_url(
            api_base=custom_base,
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert result == f"{custom_base}/images/generations"

    def test_get_complete_url_with_trailing_slash(self):
        """Test that get_complete_url strips trailing slashes from base."""
        custom_base = "https://custom.modelscope.cn/v1/"

        result = self.config.get_complete_url(
            api_base=custom_base,
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert result == "https://custom.modelscope.cn/v1/images/generations"

    @patch("litellm.llms.modelscope.image_generation.transformation.get_secret_str")
    def test_validate_environment_with_api_key(self, mock_get_secret):
        """Test that validate_environment correctly sets authorization header."""
        headers = {}
        api_key = "test_api_key"

        result = self.config.validate_environment(
            headers=headers,
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=api_key,
        )

        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"
        mock_get_secret.assert_not_called()

    @patch("litellm.llms.modelscope.image_generation.transformation.get_secret_str")
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
            api_key=None,
        )

        assert result["Authorization"] == "Bearer secret_api_key"
        mock_get_secret.assert_called_once_with("MODELSCOPE_API_KEY")

    @patch("litellm.llms.modelscope.image_generation.transformation.get_secret_str")
    def test_validate_environment_no_api_key(self, mock_get_secret):
        """Test that validate_environment raises error when no API key is available."""
        mock_get_secret.return_value = None
        headers = {}

        with pytest.raises(ValueError) as exc_info:
            self.config.validate_environment(
                headers=headers,
                model=self.model,
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

        assert "MODELSCOPE_API_KEY is not set" in str(exc_info.value)

    def test_transform_image_generation_request_basic(self):
        """Test that transform_image_generation_request creates correct request body."""
        prompt = "A beautiful sunset over mountains"
        optional_params = {}

        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert result["model"] == self.model
        assert result["prompt"] == prompt

    def test_transform_image_generation_request_with_optional_params(self):
        """Test that transform_image_generation_request includes optional params."""
        prompt = "A beautiful sunset"
        optional_params = {
            "n": 2,
            "size": "1024x1024",
            "response_format": "b64_json",
        }

        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert result["model"] == self.model
        assert result["prompt"] == prompt
        assert result["n"] == 2
        assert result["size"] == "1024x1024"
        assert result["response_format"] == "b64_json"

    def test_transform_image_generation_request_ignores_internal_params(self):
        """Test that transform_image_generation_request ignores params starting with _."""
        prompt = "A beautiful sunset"
        optional_params = {
            "n": 2,
            "_internal_param": "should_be_ignored",
        }

        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert result["model"] == self.model
        assert result["n"] == 2
        assert "_internal_param" not in result

    def test_transform_image_generation_response_with_url_images(self):
        """Test that transform_image_generation_response correctly extracts URL images."""
        response_data = {
            "created": 1234567890,
            "data": [
                {"url": "https://example.com/image1.png"},
                {"url": "https://example.com/image2.png"},
            ],
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        model_response = ImageResponse(data=[])

        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 2
        assert result.data[0].url == "https://example.com/image1.png"
        assert result.data[1].url == "https://example.com/image2.png"

    def test_transform_image_generation_response_with_b64_json(self):
        """Test that transform_image_generation_response correctly extracts base64 images."""
        response_data = {
            "created": 1234567890,
            "data": [
                {"b64_json": "iVBORw0KGgoAAAANS"},
            ],
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        model_response = ImageResponse(data=[])

        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 1
        assert result.data[0].b64_json == "iVBORw0KGgoAAAANS"
        assert result.data[0].url is None

    def test_transform_image_generation_response_with_revised_prompt(self):
        """Test that transform_image_generation_response extracts revised_prompt."""
        response_data = {
            "created": 1234567890,
            "data": [
                {
                    "url": "https://example.com/image.png",
                    "revised_prompt": "A detailed description of a beautiful sunset",
                },
            ],
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        model_response = ImageResponse(data=[])

        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 1
        assert (
            result.data[0].revised_prompt
            == "A detailed description of a beautiful sunset"
        )

    def test_transform_image_generation_response_empty_data(self):
        """Test that transform_image_generation_response handles empty data array."""
        response_data = {
            "created": 1234567890,
            "data": [],
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        model_response = ImageResponse(data=[])

        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 0

    def test_transform_image_generation_response_error_handling(self):
        """Test that transform_image_generation_response raises error on API error."""
        response_data = {
            "error": {
                "message": "Invalid prompt provided",
                "type": "invalid_request_error",
            }
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 400
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
                encoding=None,
            )

        assert "ModelScope error" in str(exc_info.value)
        assert "Invalid prompt provided" in str(exc_info.value)

    def test_transform_image_generation_response_json_error(self):
        """Test that transform_image_generation_response raises error on invalid JSON."""
        import json

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
                encoding=None,
            )

        assert "Error parsing ModelScope response" in str(exc_info.value)

    def test_get_error_class_bad_request(self):
        """Test that get_error_class returns BadRequestError for 400 status."""
        from litellm.exceptions import BadRequestError

        error = self.config.get_error_class(
            error_message="Bad request",
            status_code=400,
            headers={"Content-Type": "application/json"},
        )

        assert isinstance(error, BadRequestError)

    def test_get_error_class_authentication_error(self):
        """Test that get_error_class returns AuthenticationError for 401 status."""
        from litellm.exceptions import AuthenticationError

        error = self.config.get_error_class(
            error_message="Invalid API key",
            status_code=401,
            headers={"Content-Type": "application/json"},
        )

        assert isinstance(error, AuthenticationError)

    def test_get_error_class_internal_server_error(self):
        """Test that get_error_class returns InternalServerError for 500+ status."""
        from litellm.exceptions import InternalServerError

        error = self.config.get_error_class(
            error_message="Internal server error",
            status_code=500,
            headers={"Content-Type": "application/json"},
        )

        assert isinstance(error, InternalServerError)

    def test_get_error_class_default(self):
        """Test that get_error_class returns BadRequestError for other status codes."""
        from litellm.exceptions import BadRequestError

        error = self.config.get_error_class(
            error_message="Some error",
            status_code=404,
            headers={"Content-Type": "application/json"},
        )

        assert isinstance(error, BadRequestError)
