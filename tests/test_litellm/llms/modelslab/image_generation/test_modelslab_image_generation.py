"""
Tests for ModelsLab Image Generation transformation

Tests the transformation of OpenAI-compatible requests/responses to ModelsLab format.
"""

from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.modelslab.image_generation import (
    ModelsLabImageGenerationConfig,
    get_modelslab_image_generation_config,
)
from litellm.types.utils import ImageResponse


class TestModelsLabImageGenerationConfig:
    """Test the ModelsLabImageGenerationConfig class"""

    def setup_method(self):
        self.config = ModelsLabImageGenerationConfig()

    def test_get_supported_openai_params(self):
        params = self.config.get_supported_openai_params("modelslab/flux")
        assert "n" in params
        assert "size" in params

    def test_validate_environment_raises_without_api_key(self):
        with pytest.raises(ValueError) as exc_info:
            self.config.validate_environment(
                headers={},
                model="modelslab/flux",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )
        assert "MODELSLAB_API_KEY" in str(exc_info.value)

    def test_validate_environment_sets_content_type(self):
        headers = self.config.validate_environment(
            headers={},
            model="modelslab/flux",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-key",
        )
        assert headers["Content-Type"] == "application/json"
        # API key should NOT be in headers (it goes in body)
        assert "Authorization" not in headers

    def test_map_openai_params_n_to_samples(self):
        non_default_params = {"n": 3}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="modelslab/flux",
            drop_params=False,
        )

        assert result["samples"] == 3
        assert "n" not in result

    def test_map_openai_params_size_to_width_height(self):
        non_default_params = {"size": "1024x768"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="modelslab/flux",
            drop_params=False,
        )

        assert result["width"] == 1024
        assert result["height"] == 768
        assert "size" not in result

    def test_map_openai_params_unsupported_raises_error(self):
        non_default_params = {"unsupported_param": "value"}
        optional_params = {}

        with pytest.raises(ValueError) as exc_info:
            self.config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model="modelslab/flux",
                drop_params=False,
            )

        assert "unsupported_param" in str(exc_info.value)

    def test_map_openai_params_unsupported_dropped(self):
        non_default_params = {"unsupported_param": "value", "n": 2}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="modelslab/flux",
            drop_params=True,
        )

        assert "unsupported_param" not in result
        assert result["samples"] == 2

    def test_transform_image_generation_request_includes_key_in_body(self):
        result = self.config.transform_image_generation_request(
            model="modelslab/flux",
            prompt="A beautiful sunset over mountains",
            optional_params={"samples": 2, "width": 1024, "height": 1024},
            litellm_params={"api_key": "test-api-key-123"},
            headers={},
        )

        assert result["key"] == "test-api-key-123"
        assert result["prompt"] == "A beautiful sunset over mountains"
        assert result["model_id"] == "flux"
        assert result["samples"] == 2
        assert result["width"] == 1024
        assert result["height"] == 1024

    def test_transform_image_generation_request_strips_provider_prefix(self):
        result = self.config.transform_image_generation_request(
            model="modelslab/sdxl",
            prompt="test",
            optional_params={},
            litellm_params={"api_key": "key"},
            headers={},
        )

        assert result["model_id"] == "sdxl"

    def test_get_complete_url_default(self):
        url = self.config.get_complete_url(
            api_base=None,
            api_key="test-key",
            model="modelslab/flux",
            optional_params={},
            litellm_params={},
        )

        assert url == "https://modelslab.com/api/v6/images/text2img"

    def test_get_complete_url_custom_base(self):
        url = self.config.get_complete_url(
            api_base="https://custom.modelslab.com/api/v6",
            api_key="test-key",
            model="modelslab/flux",
            optional_params={},
            litellm_params={},
        )

        assert url == "https://custom.modelslab.com/api/v6/images/text2img"

    def test_transform_image_generation_response_success(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "success",
            "output": [
                "https://pub-cdn.modelslab.com/image1.png",
                "https://pub-cdn.modelslab.com/image2.png",
            ],
            "generationTime": 2.5,
            "id": 12345,
            "meta": {},
        }
        mock_response.status_code = 200
        mock_response.headers = {}

        model_response = ImageResponse(data=[])
        mock_logging = MagicMock()

        result = self.config.transform_image_generation_response(
            model="modelslab/flux",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 2
        assert result.data[0].url == "https://pub-cdn.modelslab.com/image1.png"
        assert result.data[1].url == "https://pub-cdn.modelslab.com/image2.png"

    def test_transform_image_generation_response_processing_raises(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "processing",
            "id": 12345,
            "fetch_result": "https://modelslab.com/api/v6/fetch/12345",
            "eta": 10,
            "message": "Image generation in progress",
        }
        mock_response.status_code = 200
        mock_response.headers = {}

        model_response = ImageResponse(data=[])
        mock_logging = MagicMock()

        with pytest.raises(Exception) as exc_info:
            self.config.transform_image_generation_response(
                model="modelslab/flux",
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=mock_logging,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

        error_msg = str(exc_info.value)
        assert "processing" in error_msg.lower()
        assert "https://modelslab.com/api/v6/fetch/12345" in error_msg

    def test_transform_image_generation_response_error_raises(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "error",
            "message": "Invalid API key",
        }
        mock_response.status_code = 401
        mock_response.headers = {}

        model_response = ImageResponse(data=[])
        mock_logging = MagicMock()

        with pytest.raises(Exception) as exc_info:
            self.config.transform_image_generation_response(
                model="modelslab/flux",
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=mock_logging,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

        assert "Invalid API key" in str(exc_info.value)


class TestFactoryFunction:
    def test_get_modelslab_image_generation_config(self):
        config = get_modelslab_image_generation_config("modelslab/flux")
        assert isinstance(config, ModelsLabImageGenerationConfig)

    def test_factory_returns_config_for_any_model(self):
        config = get_modelslab_image_generation_config("modelslab/custom-model")
        assert isinstance(config, ModelsLabImageGenerationConfig)
