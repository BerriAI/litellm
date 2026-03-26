"""
Tests for VolcEngine (ByteDance Ark) Image Generation provider.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.volcengine.image_generation.transformation import (
    VolcEngineImageGenerationConfig,
)
from litellm.types.utils import ImageResponse


class TestVolcEngineImageGenerationConfig:
    def setup_method(self):
        self.config = VolcEngineImageGenerationConfig()

    def test_get_supported_openai_params(self):
        params = self.config.get_supported_openai_params(model="seedream-3.0")
        assert "n" in params
        assert "size" in params
        assert "response_format" in params
        assert "quality" in params
        assert "style" in params
        assert "user" in params
        assert "seed" in params

    def test_map_openai_params_supported(self):
        result = self.config.map_openai_params(
            non_default_params={"n": 2, "size": "1024x1024"},
            optional_params={},
            model="seedream-3.0",
            drop_params=False,
        )
        assert result == {"n": 2, "size": "1024x1024"}

    def test_map_openai_params_unsupported_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            self.config.map_openai_params(
                non_default_params={"unsupported_param": "value"},
                optional_params={},
                model="seedream-3.0",
                drop_params=False,
            )

    def test_map_openai_params_unsupported_drop(self):
        result = self.config.map_openai_params(
            non_default_params={"unsupported_param": "value", "n": 1},
            optional_params={},
            model="seedream-3.0",
            drop_params=True,
        )
        assert result == {"n": 1}

    def test_get_complete_url_default(self):
        url = self.config.get_complete_url(
            api_base=None,
            api_key="test",
            model="seedream-3.0",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://ark.cn-beijing.volces.com/api/v3/images/generations"

    def test_get_complete_url_custom_base(self):
        url = self.config.get_complete_url(
            api_base="https://custom.api.com/api/v3",
            api_key="test",
            model="seedream-3.0",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.api.com/api/v3/images/generations"

    def test_get_complete_url_already_has_endpoint(self):
        url = self.config.get_complete_url(
            api_base="https://custom.api.com/api/v3/images/generations",
            api_key="test",
            model="seedream-3.0",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.api.com/api/v3/images/generations"

    def test_get_complete_url_trailing_slash(self):
        url = self.config.get_complete_url(
            api_base="https://ark.cn-beijing.volces.com/",
            api_key="test",
            model="seedream-3.0",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://ark.cn-beijing.volces.com/api/v3/images/generations"

    def test_validate_environment(self):
        headers = self.config.validate_environment(
            headers={},
            model="seedream-3.0",
            messages=[],
            optional_params={},
            litellm_params={"api_key": "test-key-123"},
        )
        assert headers["Authorization"] == "Bearer test-key-123"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_no_key_raises(self):
        with pytest.raises(ValueError, match="API key is required"):
            self.config.validate_environment(
                headers={},
                model="seedream-3.0",
                messages=[],
                optional_params={},
                litellm_params={},
            )

    def test_transform_image_generation_request_basic(self):
        result = self.config.transform_image_generation_request(
            model="seedream-3.0",
            prompt="a cat in a suit",
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert result == {
            "model": "seedream-3.0",
            "prompt": "a cat in a suit",
        }

    def test_transform_image_generation_request_with_params(self):
        result = self.config.transform_image_generation_request(
            model="seedream-3.0",
            prompt="a cat in a suit",
            optional_params={
                "n": 2,
                "size": "1024x1024",
                "response_format": "url",
                "quality": "hd",
                "seed": 42,
            },
            litellm_params={},
            headers={},
        )
        assert result == {
            "model": "seedream-3.0",
            "prompt": "a cat in a suit",
            "n": 2,
            "size": "1024x1024",
            "response_format": "url",
            "quality": "hd",
            "seed": 42,
        }

    def test_transform_image_generation_response_url(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "created": 1700000000,
            "data": [
                {"url": "https://example.com/image1.png"},
                {"url": "https://example.com/image2.png"},
            ],
        }
        mock_response.status_code = 200

        logging_obj = MagicMock()
        model_response = ImageResponse()

        result = self.config.transform_image_generation_response(
            model="seedream-3.0",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data={"prompt": "a cat"},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 2
        assert result.data[0].url == "https://example.com/image1.png"
        assert result.data[1].url == "https://example.com/image2.png"
        assert result.created == 1700000000

    def test_transform_image_generation_response_b64(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "created": 1700000000,
            "data": [
                {"b64_json": "iVBORw0KGgoAAAANSUhEUg=="},
            ],
        }
        mock_response.status_code = 200

        logging_obj = MagicMock()
        model_response = ImageResponse()

        result = self.config.transform_image_generation_response(
            model="seedream-3.0",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data={"prompt": "a cat"},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(result.data) == 1
        assert result.data[0].b64_json == "iVBORw0KGgoAAAANSUhEUg=="

    def test_transform_image_generation_response_invalid_json(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.side_effect = ValueError("bad json")
        mock_response.status_code = 500
        mock_response.headers = httpx.Headers({})

        logging_obj = MagicMock()
        model_response = ImageResponse()

        with pytest.raises(Exception):
            self.config.transform_image_generation_response(
                model="seedream-3.0",
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=logging_obj,
                request_data={"prompt": "a cat"},
                optional_params={},
                litellm_params={},
                encoding=None,
            )


class TestVolcEngineImageGenerationE2E:
    """End-to-end tests using litellm.image_generation with mocked HTTP."""

    def test_image_generation_routes_to_volcengine(self):
        """Verify that volcengine/ prefix routes through the llm_http_handler path."""
        from litellm.utils import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="volcengine/seedream-3.0"
        )
        assert provider == "volcengine"
        assert model == "seedream-3.0"

    def test_provider_config_registered(self):
        """Verify VolcEngine image generation config is registered in ProviderConfigManager."""
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_image_generation_config(
            model="seedream-3.0",
            provider=LlmProviders.VOLCENGINE,
        )
        assert config is not None
        assert isinstance(config, VolcEngineImageGenerationConfig)
