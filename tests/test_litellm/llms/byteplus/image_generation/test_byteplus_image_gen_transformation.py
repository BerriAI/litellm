import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm import get_llm_provider
from litellm.llms.byteplus.image_generation.transformation import (
    DEFAULT_API_BASE,
    IMAGE_GENERATION_ENDPOINT,
    BytePlusImageGenerationConfig,
)
from litellm.types.utils import ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager

MODULE = "litellm.llms.byteplus.image_generation.transformation.get_secret_str"


class TestBytePlusImageGenerationTransformation:
    def setup_method(self):
        self.config = BytePlusImageGenerationConfig()
        self.model = "seedream-5-0-260128"
        self.logging_obj = MagicMock()

    def test_provider_routing(self):
        model, provider, _, _ = get_llm_provider("byteplus/seedream-5-0-260128")
        assert provider == "byteplus"
        assert model == "seedream-5-0-260128"

    def test_provider_config_registered(self):
        config = ProviderConfigManager.get_provider_image_generation_config(
            model=self.model,
            provider=LlmProviders.BYTEPLUS,
        )
        assert isinstance(config, BytePlusImageGenerationConfig)

    def test_supported_params(self):
        assert self.config.get_supported_openai_params(self.model) == ["n", "size", "response_format"]

    def test_map_openai_params_passthrough(self):
        result = self.config.map_openai_params(
            non_default_params={"size": "2048x2048", "response_format": "url", "watermark": False},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result == {"size": "2048x2048", "response_format": "url", "watermark": False}

    def test_map_openai_params_n_expands_to_sequential(self):
        result = self.config.map_openai_params(
            non_default_params={"n": 4, "size": "1024x1024"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["size"] == "1024x1024"
        assert result["sequential_image_generation"] == "auto"
        assert result["sequential_image_generation_options"] == {"max_images": 4}
        assert "n" not in result

    def test_map_openai_params_n_one_no_sequential(self):
        result = self.config.map_openai_params(
            non_default_params={"n": 1},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert "sequential_image_generation" not in result
        assert "n" not in result

    @patch(MODULE)
    def test_get_complete_url_with_api_base(self, mock_secret):
        result = self.config.get_complete_url(
            api_base="https://custom.ark.example.com/api/v3",
            api_key="k",
            model=self.model,
            optional_params={},
            litellm_params={},
        )
        assert result == f"https://custom.ark.example.com/api/v3/{IMAGE_GENERATION_ENDPOINT}"
        mock_secret.assert_not_called()

    @patch(MODULE)
    def test_get_complete_url_default(self, mock_secret):
        mock_secret.return_value = None
        result = self.config.get_complete_url(
            api_base=None,
            api_key="k",
            model=self.model,
            optional_params={},
            litellm_params={},
        )
        assert result == f"{DEFAULT_API_BASE}/{IMAGE_GENERATION_ENDPOINT}"

    @patch(MODULE)
    def test_get_complete_url_no_double_endpoint(self, mock_secret):
        mock_secret.return_value = None
        base = f"{DEFAULT_API_BASE}/{IMAGE_GENERATION_ENDPOINT}"
        result = self.config.get_complete_url(
            api_base=base,
            api_key="k",
            model=self.model,
            optional_params={},
            litellm_params={},
        )
        assert result == base

    @patch(MODULE)
    def test_validate_environment_with_api_key(self, mock_secret):
        headers = self.config.validate_environment(
            headers={},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="my_key",
        )
        assert headers["Authorization"] == "Bearer my_key"
        assert headers["Content-Type"] == "application/json"
        mock_secret.assert_not_called()

    @patch(MODULE)
    def test_validate_environment_ark_key_fallback(self, mock_secret):
        mock_secret.side_effect = lambda name: "ark_secret" if name == "ARK_API_KEY" else None
        headers = self.config.validate_environment(
            headers={},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )
        assert headers["Authorization"] == "Bearer ark_secret"

    @patch(MODULE)
    def test_validate_environment_no_key_raises(self, mock_secret):
        mock_secret.return_value = None
        with pytest.raises(ValueError, match="BYTEPLUS_API_KEY or ARK_API_KEY is not set"):
            self.config.validate_environment(
                headers={},
                model=self.model,
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

    def test_transform_request(self):
        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt="a cat surfing",
            optional_params={"size": "2048x2048", "watermark": False},
            litellm_params={},
            headers={},
        )
        assert result == {
            "model": self.model,
            "prompt": "a cat surfing",
            "size": "2048x2048",
            "watermark": False,
        }

    def test_transform_response_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"url": "https://img.example.com/1.png"},
                {"b64_json": "abc123"},
            ]
        }
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
        assert result.data[0].url == "https://img.example.com/1.png"
        assert result.data[0].b64_json is None
        assert result.data[1].b64_json == "abc123"
        assert result.data[1].url is None

    def test_transform_response_non_200_raises(self):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "unauthorized"
        mock_response.headers = {}
        with pytest.raises(Exception, match="unauthorized"):
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

    def test_transform_response_api_error_body_raises(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {"error": {"message": "invalid model"}}
        with pytest.raises(Exception, match="invalid model"):
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

    def test_transform_response_json_parse_error_raises(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.side_effect = ValueError("bad json")
        with pytest.raises(Exception, match="Failed to parse BytePlus"):
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
