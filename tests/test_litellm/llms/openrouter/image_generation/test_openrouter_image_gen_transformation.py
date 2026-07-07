import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

from litellm.llms.openrouter.image_generation.transformation import (
    QUALITY_ALIASES,
    OpenRouterImageGenerationConfig,
)
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.types.utils import ImageResponse


class TestOpenRouterImageGenerationTransformation:
    def setup_method(self):
        self.config = OpenRouterImageGenerationConfig()
        self.model = "bytedance-seed/seedream-4.5"
        self.logging_obj = MagicMock()

    def test_get_supported_openai_params(self):
        supported_params = self.config.get_supported_openai_params(self.model)

        assert "n" in supported_params
        assert "quality" in supported_params
        assert "size" in supported_params
        assert "background" in supported_params
        assert "output_compression" in supported_params
        assert "output_format" in supported_params
        assert len(supported_params) == 6

    def test_map_openai_params_size_passed_directly(self):
        result = self.config.map_openai_params(
            non_default_params={"size": "1024x1024"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result["size"] == "1024x1024"
        assert "image_config" not in result

    def test_map_openai_params_quality_passed_directly(self):
        result = self.config.map_openai_params(
            non_default_params={"quality": "high"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result["quality"] == "high"
        assert "image_config" not in result

    def test_map_openai_params_quality_aliases(self):
        for alias, mapped in QUALITY_ALIASES.items():
            result = self.config.map_openai_params(
                non_default_params={"quality": alias},
                optional_params={},
                model=self.model,
                drop_params=False,
            )
            assert result["quality"] == mapped

    def test_map_openai_params_multiple(self):
        result = self.config.map_openai_params(
            non_default_params={"size": "1792x1024", "quality": "hd", "n": 2},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result["size"] == "1792x1024"
        assert result["quality"] == "high"
        assert result["n"] == 2

    def test_map_openai_params_unsupported_drop_false(self):
        result = self.config.map_openai_params(
            non_default_params={"size": "1024x1024", "unsupported_param": "value"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result["unsupported_param"] == "value"

    def test_map_openai_params_unsupported_drop_true(self):
        result = self.config.map_openai_params(
            non_default_params={"size": "1024x1024", "unsupported_param": "value"},
            optional_params={},
            model=self.model,
            drop_params=True,
        )

        assert "unsupported_param" not in result

    def test_get_complete_url_default(self):
        result = self.config.get_complete_url(
            api_base=None,
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert result == "https://openrouter.ai/api/v1/images"

    def test_get_complete_url_with_custom_base(self):
        result = self.config.get_complete_url(
            api_base="https://custom.openrouter.ai/api/v1",
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert result == "https://custom.openrouter.ai/api/v1/images"

    def test_get_complete_url_already_complete(self):
        result = self.config.get_complete_url(
            api_base="https://custom.openrouter.ai/api/v1/images",
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert result == "https://custom.openrouter.ai/api/v1/images"

    @patch("litellm.llms.openrouter.image_generation.transformation.get_secret_str")
    def test_validate_environment_with_api_key(self, mock_get_secret):
        result = self.config.validate_environment(
            headers={},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test_api_key",
        )

        assert result["Authorization"] == "Bearer test_api_key"
        mock_get_secret.assert_not_called()

    @patch("litellm.llms.openrouter.image_generation.transformation.get_secret_str")
    def test_validate_environment_with_secret_key(self, mock_get_secret):
        mock_get_secret.return_value = "secret_api_key"

        result = self.config.validate_environment(
            headers={},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )

        assert result["Authorization"] == "Bearer secret_api_key"
        mock_get_secret.assert_called_once_with("OPENROUTER_API_KEY")

    def test_transform_request_basic(self):
        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt="A beautiful sunset over mountains",
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert result["model"] == self.model
        assert result["prompt"] == "A beautiful sunset over mountains"
        assert "messages" not in result

    def test_transform_request_with_optional_params(self):
        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt="A sunset",
            optional_params={
                "size": "1024x1024",
                "quality": "high",
                "n": 2,
                "background": "transparent",
            },
            litellm_params={},
            headers={},
        )

        assert result["model"] == self.model
        assert result["prompt"] == "A sunset"
        assert result["size"] == "1024x1024"
        assert result["quality"] == "high"
        assert result["n"] == 2
        assert result["background"] == "transparent"

    def test_transform_response_with_b64_images(self):
        response_data = {
            "created": 1748372400,
            "data": [
                {"b64_json": "iVBORw0KGgoAAAANS", "media_type": "image/png"},
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4175,
                "total_tokens": 4185,
                "cost": 0.04,
            },
            "model": "bytedance-seed/seedream-4.5",
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=ImageResponse(data=[]),
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 1
        assert result.data[0].b64_json == "iVBORw0KGgoAAAANS"
        assert result.data[0].url is None

    def test_transform_response_with_url_images(self):
        response_data = {
            "created": 1748372400,
            "data": [
                {"url": "https://example.com/image.png"},
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4175,
                "total_tokens": 4185,
            },
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=ImageResponse(data=[]),
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/image.png"
        assert result.data[0].b64_json is None

    def test_transform_response_usage_and_cost(self):
        response_data = {
            "created": 1748372400,
            "data": [{"b64_json": "abc123"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4175,
                "total_tokens": 4185,
                "cost": 0.04,
                "cost_details": {"input_cost": 0.001, "output_cost": 0.039},
            },
            "model": "bytedance-seed/seedream-4.5",
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=ImageResponse(data=[]),
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 4175
        assert result.usage.total_tokens == 4185
        assert result.usage.input_tokens_details.text_tokens == 10
        assert result.usage.input_tokens_details.image_tokens == 0

        assert result._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == 0.04
        assert result._hidden_params["response_cost_details"]["input_cost"] == 0.001
        assert result._hidden_params["response_cost_details"]["output_cost"] == 0.039
        assert result._hidden_params["model"] == "bytedance-seed/seedream-4.5"

    def test_transform_response_multiple_images(self):
        response_data = {
            "created": 1748372400,
            "data": [
                {"b64_json": "image1data"},
                {"b64_json": "image2data"},
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8350,
                "total_tokens": 8360,
            },
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}

        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=ImageResponse(data=[]),
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 2
        assert result.data[0].b64_json == "image1data"
        assert result.data[1].b64_json == "image2data"

    def test_transform_response_json_parse_error(self):
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.status_code = 500
        mock_response.headers = {}

        with pytest.raises(OpenRouterException) as exc_info:
            self.config.transform_image_generation_response(
                model=self.model,
                raw_response=mock_response,
                model_response=ImageResponse(data=[]),
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

        assert "Error parsing OpenRouter response" in str(exc_info.value)
        assert exc_info.value.status_code == 500

    def test_get_error_class(self):
        error = self.config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={"Content-Type": "application/json"},
        )

        assert isinstance(error, OpenRouterException)
        assert "Test error" in str(error)
        assert error.status_code == 400
