"""
Unit tests for Black Forest Labs image generation transformation functionality.

Note: Polling tests are now in test_bfl_image_generation_handler.py
since polling logic was moved to the handler.
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.black_forest_labs.image_generation.transformation import (
    BlackForestLabsImageGenerationConfig,
    get_black_forest_labs_image_generation_config,
)
from litellm.llms.black_forest_labs.common_utils import BlackForestLabsError
from litellm.types.utils import ImageObject, ImageResponse


class TestBlackForestLabsImageGenerationTransformation:
    """
    Unit tests for Black Forest Labs image generation transformation functionality.
    """

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = BlackForestLabsImageGenerationConfig()
        self.model = "flux-pro-1.1"
        self.logging_obj = MagicMock()
        self.prompt = "A beautiful sunset over the ocean"

    def test_get_supported_openai_params(self):
        """Test that supported OpenAI params are returned correctly."""
        params = self.config.get_supported_openai_params(self.model)

        assert "n" in params
        assert "size" in params
        assert "response_format" in params
        assert "quality" in params

    def test_map_openai_params_basic(self):
        """Test mapping of OpenAI params to BFL params."""
        non_default_params = {}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params, optional_params, self.model, drop_params=False
        )

        # Empty input should return empty output
        assert result == {}

    def test_map_openai_params_size_mapping(self):
        """Test that OpenAI size is mapped to BFL width/height."""
        non_default_params = {"size": "1024x1024"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params, optional_params, self.model, drop_params=False
        )

        assert result["width"] == 1024
        assert result["height"] == 1024

    def test_map_openai_params_size_custom(self):
        """Test custom size parsing."""
        non_default_params = {"size": "800x600"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params, optional_params, self.model, drop_params=False
        )

        assert result["width"] == 800
        assert result["height"] == 600

    def test_map_openai_params_n_for_ultra(self):
        """Test that n is mapped to num_images for ultra model."""
        non_default_params = {"n": 4}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params, optional_params, "flux-pro-1.1-ultra", drop_params=False
        )

        assert result["num_images"] == 4

    def test_map_openai_params_quality_hd_for_ultra(self):
        """Test that 'hd' quality maps to raw=True for ultra model."""
        non_default_params = {"quality": "hd"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params, optional_params, "flux-pro-1.1-ultra", drop_params=False
        )

        assert result["raw"] is True

    def test_map_openai_params_unsupported_raises(self):
        """Test that unsupported params raise ValueError when drop_params=False."""
        non_default_params = {"unsupported_param": "value"}
        optional_params = {}

        with pytest.raises(ValueError, match="not supported"):
            self.config.map_openai_params(
                non_default_params, optional_params, self.model, drop_params=False
            )

    def test_map_openai_params_unsupported_dropped(self):
        """Test that unsupported params are dropped when drop_params=True."""
        non_default_params = {"unsupported_param": "value"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params, optional_params, self.model, drop_params=True
        )

        assert "unsupported_param" not in result

    def test_validate_environment_with_api_key(self):
        """Test that validate_environment sets headers correctly."""
        headers = {}

        result = self.config.validate_environment(
            headers=headers,
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test_api_key",
        )

        assert result["x-key"] == "test_api_key"
        assert result["Content-Type"] == "application/json"

    def test_validate_environment_missing_api_key(self):
        """Test that validate_environment raises error when API key is missing."""
        headers = {}

        with patch(
            "litellm.llms.black_forest_labs.image_generation.transformation.get_secret_str",
            return_value=None,
        ):
            with pytest.raises(BlackForestLabsError, match="BFL_API_KEY"):
                self.config.validate_environment(
                    headers=headers,
                    model=self.model,
                    messages=[],
                    optional_params={},
                    litellm_params={},
                    api_key=None,
                )

    def test_get_model_endpoint_flux_pro_1_1(self):
        """Test endpoint for flux-pro-1.1 model."""
        endpoint = self.config._get_model_endpoint("flux-pro-1.1")
        assert endpoint == "/v1/flux-pro-1.1"

    def test_get_model_endpoint_flux_pro_1_1_ultra(self):
        """Test endpoint for flux-pro-1.1-ultra model."""
        endpoint = self.config._get_model_endpoint("flux-pro-1.1-ultra")
        assert endpoint == "/v1/flux-pro-1.1-ultra"

    def test_get_model_endpoint_flux_dev(self):
        """Test endpoint for flux-dev model."""
        endpoint = self.config._get_model_endpoint("flux-dev")
        assert endpoint == "/v1/flux-dev"

    def test_get_model_endpoint_flux_pro(self):
        """Test endpoint for flux-pro model."""
        endpoint = self.config._get_model_endpoint("flux-pro")
        assert endpoint == "/v1/flux-pro"

    def test_get_model_endpoint_unknown_defaults(self):
        """Test that unknown models default to flux-pro-1.1."""
        endpoint = self.config._get_model_endpoint("unknown-model")
        assert endpoint == "/v1/flux-pro-1.1"

    def test_get_model_endpoint_with_provider_prefix(self):
        """Test that provider prefix is stripped from model name."""
        endpoint = self.config._get_model_endpoint("black_forest_labs/flux-pro-1.1")
        assert endpoint == "/v1/flux-pro-1.1"

    def test_get_complete_url(self):
        """Test URL construction with default base."""
        url = self.config.get_complete_url(
            api_base=None,
            api_key=None,
            model="flux-pro-1.1",
            optional_params={},
            litellm_params={},
        )

        assert "https://api.bfl.ai/v1/flux-pro-1.1" == url

    def test_get_complete_url_custom_base(self):
        """Test URL construction with custom base."""
        url = self.config.get_complete_url(
            api_base="https://custom.api.com",
            api_key=None,
            model="flux-pro-1.1",
            optional_params={},
            litellm_params={},
        )

        assert "https://custom.api.com/v1/flux-pro-1.1" == url

    def test_transform_image_generation_request(self):
        """Test request body transformation."""
        request = self.config.transform_image_generation_request(
            model=self.model,
            prompt=self.prompt,
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert request["prompt"] == self.prompt
        assert request["output_format"] == "png"

    def test_transform_image_generation_request_custom_format(self):
        """Test request body with custom output format."""
        request = self.config.transform_image_generation_request(
            model=self.model,
            prompt=self.prompt,
            optional_params={"output_format": "jpeg"},
            litellm_params={},
            headers={},
        )

        assert request["output_format"] == "jpeg"

    def test_transform_image_generation_request_ultra_params(self):
        """Test request body with ultra-specific params."""
        request = self.config.transform_image_generation_request(
            model="flux-pro-1.1-ultra",
            prompt=self.prompt,
            optional_params={
                "raw": True,
                "num_images": 2,
                "aspect_ratio": "16:9",
            },
            litellm_params={},
            headers={},
        )

        assert request["raw"] is True
        assert request["num_images"] == 2
        assert request["aspect_ratio"] == "16:9"

    def test_transform_image_generation_response_success(self):
        """Test response transformation with final polled response."""
        # The response is now the FINAL polled response from handler
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "Ready",
            "result": {"sample": "https://example.com/image.png"},
        }
        mock_response.status_code = 200

        model_response = ImageResponse(created=0, data=[])

        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=self.logging_obj,
        )

        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/image.png"

    def test_transform_image_generation_response_multiple_images(self):
        """Test response transformation with multiple images."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "Ready",
            "result": [
                "https://example.com/image1.png",
                "https://example.com/image2.png",
            ],
        }
        mock_response.status_code = 200

        model_response = ImageResponse(created=0, data=[])

        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=self.logging_obj,
        )

        assert len(result.data) == 2
        assert result.data[0].url == "https://example.com/image1.png"
        assert result.data[1].url == "https://example.com/image2.png"

    def test_transform_image_generation_response_no_image(self):
        """Test response transformation when no image URL is present."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "Ready",
            "result": {},
        }
        mock_response.status_code = 200

        model_response = ImageResponse(created=0, data=[])

        with pytest.raises(BlackForestLabsError, match="No image URL"):
            self.config.transform_image_generation_response(
                model=self.model,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=self.logging_obj,
            )

    def test_get_error_class(self):
        """Test that get_error_class returns BlackForestLabsError."""
        error = self.config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={},
        )

        assert isinstance(error, BlackForestLabsError)
        assert error.status_code == 400
        assert "Test error" in str(error.message)

    def test_get_black_forest_labs_image_generation_config(self):
        """Test the factory function."""
        config = get_black_forest_labs_image_generation_config("flux-pro-1.1")

        assert isinstance(config, BlackForestLabsImageGenerationConfig)
