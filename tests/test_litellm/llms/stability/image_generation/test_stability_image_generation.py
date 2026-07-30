"""
Tests for Stability AI Image Generation transformation

Tests the transformation of OpenAI-compatible requests/responses to Stability AI format.
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.stability.image_generation import (
    StabilityImageGenerationConfig,
    get_stability_image_generation_config,
)
from litellm.types.llms.stability import (
    OPENAI_SIZE_TO_STABILITY_ASPECT_RATIO,
    STABILITY_GENERATION_MODELS,
)
from litellm.types.utils import ImageResponse


class TestStabilityImageGenerationConfig:
    """Test the StabilityImageGenerationConfig class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = StabilityImageGenerationConfig()

    def test_get_supported_openai_params(self):
        """Test that supported OpenAI params are returned"""
        params = self.config.get_supported_openai_params("stability/sd3")
        assert "n" in params
        assert "size" in params
        assert "response_format" in params

    def test_map_openai_params_size_to_aspect_ratio(self):
        """Test that OpenAI size is mapped to Stability aspect_ratio"""
        non_default_params = {"size": "1024x1024"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="stability/sd3",
            drop_params=False,
        )

        assert result.get("aspect_ratio") == "1:1"

    def test_map_openai_params_size_16_9(self):
        """Test that 1792x1024 maps to 16:9 aspect ratio"""
        non_default_params = {"size": "1792x1024"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="stability/sd3",
            drop_params=False,
        )

        assert result.get("aspect_ratio") == "16:9"

    def test_map_openai_params_n_stored_internally(self):
        """Test that n parameter is stored with underscore prefix"""
        non_default_params = {"n": 2}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="stability/sd3",
            drop_params=False,
        )

        assert result.get("_n") == 2
        assert "n" not in result

    def test_map_openai_params_unsupported_raises_error(self):
        """Test that unsupported params raise ValueError when drop_params=False"""
        non_default_params = {"unsupported_param": "value"}
        optional_params = {}

        with pytest.raises(ValueError) as exc_info:
            self.config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model="stability/sd3",
                drop_params=False,
            )

        assert "unsupported_param" in str(exc_info.value)

    def test_map_openai_params_unsupported_dropped(self):
        """Test that unsupported params are dropped when drop_params=True"""
        non_default_params = {"unsupported_param": "value", "size": "1024x1024"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="stability/sd3",
            drop_params=True,
        )

        assert "unsupported_param" not in result
        assert result.get("aspect_ratio") == "1:1"

    def test_get_model_endpoint_sd3(self):
        """Test that SD3 model gets correct endpoint"""
        endpoint = self.config._get_model_endpoint("stability/sd3")
        assert endpoint == "/v2beta/stable-image/generate/sd3"

    def test_get_model_endpoint_sd35_large(self):
        """Test that SD3.5 Large model gets correct endpoint"""
        endpoint = self.config._get_model_endpoint("stability/sd3.5-large")
        assert endpoint == "/v2beta/stable-image/generate/sd3"

    def test_get_model_endpoint_ultra(self):
        """Test that Stable Image Ultra model gets correct endpoint"""
        endpoint = self.config._get_model_endpoint("stability/stable-image-ultra")
        assert endpoint == "/v2beta/stable-image/generate/ultra"

    def test_get_model_endpoint_core(self):
        """Test that Stable Image Core model gets correct endpoint"""
        endpoint = self.config._get_model_endpoint("stability/stable-image-core")
        assert endpoint == "/v2beta/stable-image/generate/core"

    def test_get_complete_url(self):
        """Test that complete URL is constructed correctly"""
        url = self.config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model="stability/sd3",
            optional_params={},
            litellm_params={},
        )

        assert url == "https://api.stability.ai/v2beta/stable-image/generate/sd3"

    def test_get_complete_url_with_custom_base(self):
        """Test that custom api_base is used when provided"""
        url = self.config.get_complete_url(
            api_base="https://custom.stability.ai",
            api_key="test-key",
            model="stability/sd3",
            optional_params={},
            litellm_params={},
        )

        assert url == "https://custom.stability.ai/v2beta/stable-image/generate/sd3"

    def test_validate_environment_sets_headers(self):
        """Test that validate_environment sets correct headers"""
        headers = self.config.validate_environment(
            headers={},
            model="stability/sd3",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-api-key",
        )

        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["Accept"] == "application/json"

    def test_validate_environment_raises_without_api_key(self):
        """Test that validate_environment raises error without API key"""
        with pytest.raises(ValueError) as exc_info:
            self.config.validate_environment(
                headers={},
                model="stability/sd3",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

        assert "STABILITY_API_KEY" in str(exc_info.value)

    def test_transform_image_generation_request(self):
        """Test transformation of request to Stability format"""
        result = self.config.transform_image_generation_request(
            model="stability/sd3",
            prompt="A beautiful sunset",
            optional_params={"aspect_ratio": "16:9", "negative_prompt": "blurry"},
            litellm_params={},
            headers={},
        )

        assert result["prompt"] == "A beautiful sunset"
        assert result["output_format"] == "png"
        assert result["aspect_ratio"] == "16:9"
        assert result["negative_prompt"] == "blurry"

    def test_transform_image_generation_request_ignores_internal_params(self):
        """Test that internal params (prefixed with _) are not included"""
        result = self.config.transform_image_generation_request(
            model="stability/sd3",
            prompt="Test",
            optional_params={"_n": 2, "_response_format": "url", "aspect_ratio": "1:1"},
            litellm_params={},
            headers={},
        )

        assert "_n" not in result
        assert "_response_format" not in result
        assert result["aspect_ratio"] == "1:1"

    def test_transform_image_generation_response(self):
        """Test transformation of Stability response to OpenAI format"""
        # Mock the raw response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "image": "base64encodedimage==",
            "finish_reason": "SUCCESS",
            "seed": 12345,
        }
        mock_response.status_code = 200
        mock_response.headers = {}

        model_response = ImageResponse(data=[])
        mock_logging = MagicMock()

        result = self.config.transform_image_generation_response(
            model="stability/sd3",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 1
        assert result.data[0].b64_json == "base64encodedimage=="
        assert result.data[0].url is None

    def test_transform_image_generation_response_content_filtered(self):
        """Test that content filtered response raises error"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "finish_reason": "CONTENT_FILTERED",
        }
        mock_response.status_code = 200
        mock_response.headers = {}

        model_response = ImageResponse(data=[])
        mock_logging = MagicMock()

        with pytest.raises(Exception) as exc_info:
            self.config.transform_image_generation_response(
                model="stability/sd3",
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=mock_logging,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

        assert "filtered" in str(exc_info.value).lower()


class TestFactoryFunction:
    """Test the factory function"""

    def test_get_stability_image_generation_config(self):
        """Test that factory returns correct config type"""
        config = get_stability_image_generation_config("stability/sd3")
        assert isinstance(config, StabilityImageGenerationConfig)

    def test_factory_returns_config_for_any_model(self):
        """Test that factory works for any model name"""
        config = get_stability_image_generation_config("stability/custom-model")
        assert isinstance(config, StabilityImageGenerationConfig)


class TestOpenAISizeMapping:
    """Test the size to aspect ratio mapping"""

    def test_all_sizes_have_mappings(self):
        """Test that standard OpenAI sizes have mappings"""
        expected_sizes = ["1024x1024", "1792x1024", "1024x1792", "512x512", "256x256"]
        for size in expected_sizes:
            assert size in OPENAI_SIZE_TO_STABILITY_ASPECT_RATIO

    def test_square_sizes_map_to_1_1(self):
        """Test that square sizes map to 1:1"""
        square_sizes = ["1024x1024", "512x512", "256x256"]
        for size in square_sizes:
            assert OPENAI_SIZE_TO_STABILITY_ASPECT_RATIO[size] == "1:1"


class TestStabilityGenerationModels:
    """Test the model endpoint mappings"""

    def test_sd3_models_use_sd3_endpoint(self):
        """Test that SD3 models use the SD3 endpoint"""
        sd3_models = ["sd3", "sd3-large", "sd3-medium", "sd3.5-large"]
        for model in sd3_models:
            assert STABILITY_GENERATION_MODELS[model] == "/v2beta/stable-image/generate/sd3"

    def test_ultra_model_uses_ultra_endpoint(self):
        """Test that Ultra model uses ultra endpoint"""
        assert STABILITY_GENERATION_MODELS["stable-image-ultra"] == "/v2beta/stable-image/generate/ultra"

    def test_core_model_uses_core_endpoint(self):
        """Test that Core model uses core endpoint"""
        assert STABILITY_GENERATION_MODELS["stable-image-core"] == "/v2beta/stable-image/generate/core"
