"""
Unit tests for Black Forest Labs image generation transformation functionality.
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
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Should be empty since no params provided
        assert result == {}

    def test_map_openai_params_size_mapping(self):
        """Test that OpenAI size param is mapped to BFL width/height."""
        non_default_params = {"size": "1024x1024"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result.get("width") == 1024
        assert result.get("height") == 1024

    def test_map_openai_params_size_custom(self):
        """Test custom size parsing."""
        non_default_params = {"size": "1920x1080"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result.get("width") == 1920
        assert result.get("height") == 1080

    def test_map_openai_params_n_for_ultra(self):
        """Test that n param is mapped to num_images for ultra model."""
        non_default_params = {"n": 4}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="flux-pro-1.1-ultra",
            drop_params=False,
        )

        assert result.get("num_images") == 4

    def test_map_openai_params_quality_hd_for_ultra(self):
        """Test that quality=hd is mapped to raw=True for ultra model."""
        non_default_params = {"quality": "hd"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="flux-pro-1.1-ultra",
            drop_params=False,
        )

        assert result.get("raw") is True

    def test_map_openai_params_unsupported_raises(self):
        """Test that unsupported param raises error when drop_params=False."""
        non_default_params = {"unsupported_param": "value"}
        optional_params = {}

        with pytest.raises(ValueError) as exc_info:
            self.config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=self.model,
                drop_params=False,
            )

        assert "unsupported_param" in str(exc_info.value)

    def test_map_openai_params_unsupported_dropped(self):
        """Test that unsupported param is dropped when drop_params=True."""
        non_default_params = {"unsupported_param": "value"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=True,
        )

        assert "unsupported_param" not in result

    def test_validate_environment_with_api_key(self):
        """Test environment validation with provided API key."""
        headers = {}

        result = self.config.validate_environment(
            headers=headers,
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-api-key",
        )

        assert result["x-key"] == "test-api-key"
        assert result["Content-Type"] == "application/json"
        assert result["Accept"] == "application/json"

    def test_validate_environment_missing_api_key(self):
        """Test that missing API key raises error."""
        headers = {}

        with patch("litellm.llms.black_forest_labs.image_generation.transformation.get_secret_str") as mock_get_secret:
            mock_get_secret.return_value = None

            with pytest.raises(BlackForestLabsError) as exc_info:
                self.config.validate_environment(
                    headers=headers,
                    model=self.model,
                    messages=[],
                    optional_params={},
                    litellm_params={},
                    api_key=None,
                )

            assert exc_info.value.status_code == 401
            assert "BFL_API_KEY is not set" in exc_info.value.message

    def test_get_model_endpoint_flux_pro_1_1(self):
        """Test endpoint resolution for flux-pro-1.1."""
        endpoint = self.config._get_model_endpoint("flux-pro-1.1")
        assert endpoint == "/v1/flux-pro-1.1"

    def test_get_model_endpoint_flux_pro_1_1_ultra(self):
        """Test endpoint resolution for flux-pro-1.1-ultra."""
        endpoint = self.config._get_model_endpoint("flux-pro-1.1-ultra")
        assert endpoint == "/v1/flux-pro-1.1-ultra"

    def test_get_model_endpoint_flux_dev(self):
        """Test endpoint resolution for flux-dev."""
        endpoint = self.config._get_model_endpoint("flux-dev")
        assert endpoint == "/v1/flux-dev"

    def test_get_model_endpoint_flux_pro(self):
        """Test endpoint resolution for flux-pro."""
        endpoint = self.config._get_model_endpoint("flux-pro")
        assert endpoint == "/v1/flux-pro"

    def test_get_model_endpoint_with_provider_prefix(self):
        """Test endpoint resolution with provider prefix."""
        endpoint = self.config._get_model_endpoint("black_forest_labs/flux-pro-1.1")
        assert endpoint == "/v1/flux-pro-1.1"

    def test_get_model_endpoint_unknown_defaults(self):
        """Test that unknown model defaults to flux-pro-1.1."""
        endpoint = self.config._get_model_endpoint("unknown-model")
        assert endpoint == "/v1/flux-pro-1.1"

    def test_get_complete_url(self):
        """Test complete URL generation."""
        url = self.config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model="flux-pro-1.1",
            optional_params={},
            litellm_params={},
        )

        assert url == "https://api.bfl.ai/v1/flux-pro-1.1"

    def test_get_complete_url_custom_base(self):
        """Test complete URL generation with custom base."""
        url = self.config.get_complete_url(
            api_base="https://custom.api.com/",
            api_key="test-key",
            model="flux-pro-1.1",
            optional_params={},
            litellm_params={},
        )

        assert url == "https://custom.api.com/v1/flux-pro-1.1"

    def test_transform_image_generation_request(self):
        """Test request transformation to BFL format."""
        optional_params = {
            "width": 1024,
            "height": 1024,
            "seed": 42,
        }

        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt=self.prompt,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert result["prompt"] == self.prompt
        assert result["width"] == 1024
        assert result["height"] == 1024
        assert result["seed"] == 42
        assert result["output_format"] == "png"  # Default

    def test_transform_image_generation_request_custom_format(self):
        """Test request transformation with custom output format."""
        optional_params = {
            "output_format": "jpeg",
        }

        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt=self.prompt,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert result["output_format"] == "jpeg"

    def test_transform_image_generation_request_ultra_params(self):
        """Test request transformation with ultra-specific params."""
        optional_params = {
            "raw": True,
            "num_images": 2,
        }

        result = self.config.transform_image_generation_request(
            model="flux-pro-1.1-ultra",
            prompt=self.prompt,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert result["raw"] is True
        assert result["num_images"] == 2

    def test_poll_for_result_success(self):
        """Test successful polling."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "Ready",
            "result": {"sample": "https://example.com/image.png"},
        }

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        with patch("litellm.llms.black_forest_labs.image_generation.transformation._get_httpx_client", return_value=mock_client):
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

        mock_client = MagicMock()
        mock_client.get.side_effect = [pending_response, ready_response]

        with patch("litellm.llms.black_forest_labs.image_generation.transformation._get_httpx_client", return_value=mock_client):
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

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        with patch("litellm.llms.black_forest_labs.image_generation.transformation._get_httpx_client", return_value=mock_client):
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

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        with patch("litellm.llms.black_forest_labs.image_generation.transformation._get_httpx_client", return_value=mock_client):
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

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        with patch("litellm.llms.black_forest_labs.image_generation.transformation._get_httpx_client", return_value=mock_client):
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

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        with patch("litellm.llms.black_forest_labs.image_generation.transformation._get_httpx_client", return_value=mock_client):
            with pytest.raises(BlackForestLabsError) as exc_info:
                self.config._poll_for_result(
                    polling_url="https://api.bfl.ai/v1/get_result?id=123",
                    api_key="test-key",
                    max_wait=10,
                    interval=0.1,
                )

        assert exc_info.value.status_code == 500

    def test_extract_images_from_result_single(self):
        """Test extracting single image from result."""
        result_data = {
            "result": {"sample": "https://example.com/image.png"}
        }
        model_response = ImageResponse(created=0, data=[])

        result = self.config._extract_images_from_result(result_data, model_response)

        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/image.png"

    def test_extract_images_from_result_multiple(self):
        """Test extracting multiple images from result."""
        result_data = {
            "result": [
                "https://example.com/image1.png",
                "https://example.com/image2.png",
            ]
        }
        model_response = ImageResponse(created=0, data=[])

        result = self.config._extract_images_from_result(result_data, model_response)

        assert len(result.data) == 2
        assert result.data[0].url == "https://example.com/image1.png"
        assert result.data[1].url == "https://example.com/image2.png"

    def test_extract_images_from_result_no_image(self):
        """Test error when no image in result."""
        result_data = {"result": {}}
        model_response = ImageResponse(created=0, data=[])

        with pytest.raises(BlackForestLabsError) as exc_info:
            self.config._extract_images_from_result(result_data, model_response)

        assert exc_info.value.status_code == 500
        assert "No image URL" in exc_info.value.message

    def test_parse_initial_response_success(self):
        """Test parsing initial response."""
        mock_request = MagicMock()
        mock_request.headers = {"x-key": "test-key"}

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "task-123",
            "polling_url": "https://api.bfl.ai/v1/get_result?id=task-123",
        }
        mock_response.request = mock_request
        mock_response.status_code = 200

        polling_url, api_key = self.config._parse_initial_response(mock_response)

        assert polling_url == "https://api.bfl.ai/v1/get_result?id=task-123"
        assert api_key == "test-key"

    def test_parse_initial_response_no_polling_url(self):
        """Test error when polling URL is missing."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "task-123"}
        mock_response.status_code = 200

        with pytest.raises(BlackForestLabsError) as exc_info:
            self.config._parse_initial_response(mock_response)

        assert exc_info.value.status_code == 500
        assert "No polling_url" in exc_info.value.message

    def test_parse_initial_response_api_error(self):
        """Test parsing response with API error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "errors": ["Invalid prompt"]
        }
        mock_response.status_code = 400

        with pytest.raises(BlackForestLabsError) as exc_info:
            self.config._parse_initial_response(mock_response)

        assert "Invalid prompt" in exc_info.value.message

    def test_parse_initial_response_json_error(self):
        """Test parsing response with JSON parse error."""
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.status_code = 500

        with pytest.raises(BlackForestLabsError) as exc_info:
            self.config._parse_initial_response(mock_response)

        assert "Error parsing BFL response" in exc_info.value.message

    def test_transform_image_generation_response_success(self):
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
            "result": {"sample": "https://example.com/generated-image.png"},
        }

        mock_client = MagicMock()
        mock_client.get.return_value = poll_response

        model_response = ImageResponse(created=0, data=[])

        with patch("litellm.llms.black_forest_labs.image_generation.transformation._get_httpx_client", return_value=mock_client):
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

        assert isinstance(result, ImageResponse)
        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/generated-image.png"
        assert result.created is not None

    def test_get_error_class(self):
        """Test error class generation."""
        error = self.config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={},
        )

        assert isinstance(error, BlackForestLabsError)
        assert error.status_code == 400
        assert error.message == "Test error"

    def test_get_black_forest_labs_image_generation_config(self):
        """Test factory function returns correct config."""
        config = get_black_forest_labs_image_generation_config("flux-pro-1.1")

        assert isinstance(config, BlackForestLabsImageGenerationConfig)


@pytest.mark.asyncio
class TestBlackForestLabsImageGenerationTransformationAsync:
    """Async tests for Black Forest Labs image generation."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = BlackForestLabsImageGenerationConfig()
        self.model = "flux-pro-1.1"
        self.logging_obj = MagicMock()

    async def test_poll_for_result_async_success(self):
        """Test successful async polling."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "Ready",
            "result": {"sample": "https://example.com/image.png"},
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch("litellm.llms.black_forest_labs.image_generation.transformation.get_async_httpx_client", return_value=mock_client):
            result = await self.config._poll_for_result_async(
                polling_url="https://api.bfl.ai/v1/get_result?id=123",
                api_key="test-key",
                max_wait=10,
                interval=0.1,
            )

        assert result["status"] == "Ready"
        assert result["result"]["sample"] == "https://example.com/image.png"

    async def test_async_transform_image_generation_response_success(self):
        """Test successful async response transformation."""
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
            "result": {"sample": "https://example.com/generated-image.png"},
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = poll_response

        model_response = ImageResponse(created=0, data=[])

        with patch("litellm.llms.black_forest_labs.image_generation.transformation.get_async_httpx_client", return_value=mock_client):
            result = await self.config.async_transform_image_generation_response(
                model=self.model,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

        assert isinstance(result, ImageResponse)
        assert len(result.data) == 1
        assert result.data[0].url == "https://example.com/generated-image.png"
