import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.openrouter.image_generation.transformation import (
    OpenRouterImageGenerationConfig,
)
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.types.utils import ImageResponse


class TestOpenRouterImageGenerationTransformation:
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = OpenRouterImageGenerationConfig()
        self.model = "google/gemini-2.5-flash-image"
        self.logging_obj = MagicMock()

    def test_get_supported_openai_params(self):
        """Test that get_supported_openai_params returns correct parameters."""
        supported_params = self.config.get_supported_openai_params(self.model)
        
        assert "size" in supported_params
        assert "quality" in supported_params
        assert "n" in supported_params
        assert len(supported_params) == 3

    def test_map_size_to_aspect_ratio_square(self):
        """Test mapping square sizes to aspect ratio."""
        assert self.config._map_size_to_aspect_ratio("256x256") == "1:1"
        assert self.config._map_size_to_aspect_ratio("512x512") == "1:1"
        assert self.config._map_size_to_aspect_ratio("1024x1024") == "1:1"

    def test_map_size_to_aspect_ratio_landscape(self):
        """Test mapping landscape sizes to aspect ratio."""
        assert self.config._map_size_to_aspect_ratio("1536x1024") == "3:2"
        assert self.config._map_size_to_aspect_ratio("1792x1024") == "16:9"

    def test_map_size_to_aspect_ratio_portrait(self):
        """Test mapping portrait sizes to aspect ratio."""
        assert self.config._map_size_to_aspect_ratio("1024x1536") == "2:3"
        assert self.config._map_size_to_aspect_ratio("1024x1792") == "9:16"

    def test_map_size_to_aspect_ratio_auto(self):
        """Test mapping auto size to default aspect ratio."""
        assert self.config._map_size_to_aspect_ratio("auto") == "1:1"

    def test_map_size_to_aspect_ratio_unknown(self):
        """Test mapping unknown size defaults to 1:1."""
        assert self.config._map_size_to_aspect_ratio("999x999") == "1:1"

    def test_map_quality_to_image_size_low(self):
        """Test mapping low quality values to 1K."""
        assert self.config._map_quality_to_image_size("low") == "1K"
        assert self.config._map_quality_to_image_size("standard") == "1K"
        assert self.config._map_quality_to_image_size("auto") == "1K"

    def test_map_quality_to_image_size_medium(self):
        """Test mapping medium quality to 2K."""
        assert self.config._map_quality_to_image_size("medium") == "2K"

    def test_map_quality_to_image_size_high(self):
        """Test mapping high quality values to 4K."""
        assert self.config._map_quality_to_image_size("high") == "4K"
        assert self.config._map_quality_to_image_size("hd") == "4K"

    def test_map_quality_to_image_size_unknown(self):
        """Test mapping unknown quality returns None."""
        assert self.config._map_quality_to_image_size("unknown") is None

    def test_map_openai_params_size_only(self):
        """Test that map_openai_params correctly maps size parameter."""
        non_default_params = {"size": "1024x1024"}
        optional_params = {}
        
        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False
        )
        
        assert "image_config" in result
        assert result["image_config"]["aspect_ratio"] == "1:1"

    def test_map_openai_params_quality_only(self):
        """Test that map_openai_params correctly maps quality parameter."""
        non_default_params = {"quality": "high"}
        optional_params = {}
        
        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False
        )
        
        assert "image_config" in result
        assert result["image_config"]["image_size"] == "4K"

    def test_map_openai_params_size_and_quality(self):
        """Test that map_openai_params correctly maps both size and quality."""
        non_default_params = {
            "size": "1792x1024",
            "quality": "hd"
        }
        optional_params = {}
        
        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False
        )
        
        assert "image_config" in result
        assert result["image_config"]["aspect_ratio"] == "16:9"
        assert result["image_config"]["image_size"] == "4K"

    def test_map_openai_params_with_n_parameter(self):
        """Test that map_openai_params correctly passes through n parameter."""
        non_default_params = {
            "size": "1024x1024",
            "n": 2
        }
        optional_params = {}
        
        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False
        )
        
        assert "image_config" in result
        assert result["image_config"]["aspect_ratio"] == "1:1"
        assert result["n"] == 2

    def test_map_openai_params_unsupported_param_drop_false(self):
        """Test that unsupported params are passed through when drop_params=False."""
        non_default_params = {
            "size": "1024x1024",
            "unsupported_param": "value"
        }
        optional_params = {}
        
        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False
        )
        
        assert "image_config" in result
        assert result["unsupported_param"] == "value"

    def test_map_openai_params_unsupported_param_drop_true(self):
        """Test that unsupported params are dropped when drop_params=True."""
        non_default_params = {
            "size": "1024x1024",
            "unsupported_param": "value"
        }
        optional_params = {}
        
        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=True
        )
        
        assert "image_config" in result
        assert "unsupported_param" not in result

    def test_get_complete_url_default(self):
        """Test that get_complete_url returns default OpenRouter URL."""
        result = self.config.get_complete_url(
            api_base=None,
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={}
        )
        
        assert result == "https://openrouter.ai/api/v1/chat/completions"

    def test_get_complete_url_with_custom_base(self):
        """Test that get_complete_url uses custom api_base."""
        custom_base = "https://custom.openrouter.ai/api/v1"
        
        result = self.config.get_complete_url(
            api_base=custom_base,
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={}
        )
        
        assert result == f"{custom_base}/chat/completions"

    def test_get_complete_url_with_base_already_complete(self):
        """Test that get_complete_url doesn't duplicate /chat/completions."""
        custom_base = "https://custom.openrouter.ai/api/v1/chat/completions"
        
        result = self.config.get_complete_url(
            api_base=custom_base,
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={}
        )
        
        assert result == custom_base

    @patch("litellm.llms.openrouter.image_generation.transformation.get_secret_str")
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
            api_key=api_key
        )
        
        assert result["Authorization"] == f"Bearer {api_key}"
        mock_get_secret.assert_not_called()

    @patch("litellm.llms.openrouter.image_generation.transformation.get_secret_str")
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
        mock_get_secret.assert_called_once_with("OPENROUTER_API_KEY")

    def test_transform_image_generation_request_basic(self):
        """Test that transform_image_generation_request creates correct request body."""
        prompt = "A beautiful sunset over mountains"
        optional_params = {}
        
        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        
        assert result["model"] == self.model
        assert result["messages"] == [{"role": "user", "content": prompt}]
        assert "modalities" not in result  # modalities should not be added by default

    def test_transform_image_generation_request_with_image_config(self):
        """Test that transform_image_generation_request includes image_config."""
        prompt = "A beautiful sunset"
        optional_params = {
            "image_config": {
                "aspect_ratio": "16:9",
                "image_size": "4K"
            },
            "n": 2
        }
        
        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        
        assert result["model"] == self.model
        assert result["messages"] == [{"role": "user", "content": prompt}]
        assert result["image_config"]["aspect_ratio"] == "16:9"
        assert result["image_config"]["image_size"] == "4K"
        assert result["n"] == 2

    def test_transform_image_generation_response_with_base64_images(self):
        """Test that transform_image_generation_response correctly extracts base64 images."""
        response_data = {
            "choices": [{
                "message": {
                    "content": "Here is your image!",
                    "role": "assistant",
                    "images": [{
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANS"},
                        "index": 0,
                        "type": "image_url"
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1300,
                "total_tokens": 1310,
                "completion_tokens_details": {"image_tokens": 1290},
                "cost": 0.0387243
            },
            "model": "google/gemini-2.5-flash-image"
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
            encoding=None
        )
        
        assert len(result.data) == 1
        assert result.data[0].b64_json == "iVBORw0KGgoAAAANS"
        assert result.data[0].url is None

    def test_transform_image_generation_response_with_url_images(self):
        """Test that transform_image_generation_response correctly extracts URL images."""
        response_data = {
            "choices": [{
                "message": {
                    "content": "Here is your image!",
                    "role": "assistant",
                    "images": [{
                        "image_url": {"url": "https://example.com/image.png"},
                        "index": 0,
                        "type": "image_url"
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1300,
                "total_tokens": 1310
            },
            "model": "google/gemini-2.5-flash-image"
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
            encoding=None
        )
        
        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/image.png"
        assert result.data[0].b64_json is None

    def test_transform_image_generation_response_with_usage_and_cost(self):
        """Test that transform_image_generation_response correctly extracts usage and cost."""
        response_data = {
            "choices": [{
                "message": {
                    "content": "Here is your image!",
                    "role": "assistant",
                    "images": [{
                        "image_url": {"url": "data:image/png;base64,abc123"},
                        "index": 0,
                        "type": "image_url"
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1300,
                "total_tokens": 1310,
                "completion_tokens_details": {"image_tokens": 1290},
                "cost": 0.0387243,
                "cost_details": {"input_cost": 0.001, "output_cost": 0.037}
            },
            "model": "google/gemini-2.5-flash-image"
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
            encoding=None
        )
        
        # Check usage
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 1290
        assert result.usage.total_tokens == 1310
        assert result.usage.input_tokens_details.text_tokens == 10
        assert result.usage.input_tokens_details.image_tokens == 0
        
        # Check cost
        assert hasattr(result, "_hidden_params")
        assert "additional_headers" in result._hidden_params
        assert result._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == 0.0387243
        
        # Check cost details
        assert "response_cost_details" in result._hidden_params
        assert result._hidden_params["response_cost_details"]["input_cost"] == 0.001
        assert result._hidden_params["response_cost_details"]["output_cost"] == 0.037
        
        # Check model
        assert result._hidden_params["model"] == "google/gemini-2.5-flash-image"

    def test_transform_image_generation_response_multiple_images(self):
        """Test that transform_image_generation_response handles multiple images."""
        response_data = {
            "choices": [{
                "message": {
                    "content": "Here are your images!",
                    "role": "assistant",
                    "images": [
                        {
                            "image_url": {"url": "data:image/png;base64,image1data"},
                            "index": 0,
                            "type": "image_url"
                        },
                        {
                            "image_url": {"url": "data:image/png;base64,image2data"},
                            "index": 1,
                            "type": "image_url"
                        }
                    ]
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2600,
                "total_tokens": 2610
            },
            "model": "google/gemini-2.5-flash-image"
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
            encoding=None
        )
        
        assert len(result.data) == 2
        assert result.data[0].b64_json == "image1data"
        assert result.data[1].b64_json == "image2data"

    def test_transform_image_generation_response_json_error(self):
        """Test that transform_image_generation_response raises error on invalid JSON."""
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.status_code = 500
        mock_response.headers = {}
        
        model_response = ImageResponse(data=[])
        
        with pytest.raises(OpenRouterException) as exc_info:
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
        
        assert "Error parsing OpenRouter response" in str(exc_info.value)
        assert exc_info.value.status_code == 500

    def test_transform_image_generation_response_transformation_error(self):
        """Test that transform_image_generation_response handles transformation errors."""
        response_data = {
            "choices": [{
                "message": {
                    "content": "Here is your image!",
                    "role": "assistant",
                    "images": "invalid_format"  # Invalid format
                }
            }]
        }
        
        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.status_code = 200
        mock_response.headers = {}
        
        model_response = ImageResponse(data=[])
        
        with pytest.raises(OpenRouterException) as exc_info:
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
        
        assert "Error transforming OpenRouter image generation response" in str(exc_info.value)

    def test_get_error_class(self):
        """Test that get_error_class returns OpenRouterException."""
        error = self.config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={"Content-Type": "application/json"}
        )
        
        assert isinstance(error, OpenRouterException)
        assert "Test error" in str(error)
        assert error.status_code == 400
