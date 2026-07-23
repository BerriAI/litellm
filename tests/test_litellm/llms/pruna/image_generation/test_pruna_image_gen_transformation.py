import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm import get_llm_provider
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.pruna.image_generation.transformation import (
    DEFAULT_API_BASE,
    PREDICTIONS_ENDPOINT,
    PrunaImageGenerationConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

MODULE = "litellm.llms.pruna.image_generation.transformation.get_secret_str"


def _response(status_code: int, *, json_body=None, text_body=None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.pruna.ai/v1/predictions")
    if json_body is not None:
        return httpx.Response(status_code, json=json_body, request=request)
    return httpx.Response(status_code, content=text_body, request=request)


class TestPrunaImageGenerationTransformation:
    def setup_method(self):
        self.config = PrunaImageGenerationConfig()
        self.model = "p-image"
        self.logging_obj = MagicMock()

    def test_provider_routing(self):
        model, provider, _, _ = get_llm_provider("pruna/p-image")
        assert provider == "pruna"
        assert model == "p-image"

    def test_provider_config_registered(self):
        config = ProviderConfigManager.get_provider_image_generation_config(
            model=self.model,
            provider=LlmProviders.PRUNA,
        )
        assert isinstance(config, PrunaImageGenerationConfig)

    def test_supported_params(self):
        assert self.config.get_supported_openai_params(self.model) == ["size"]

    def test_map_openai_params_size_to_custom_dimensions(self):
        result = self.config.map_openai_params(
            non_default_params={"size": "1024x768"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result == {"width": 1024, "height": 768, "aspect_ratio": "custom"}

    def test_map_openai_params_passthrough_and_drops_n(self):
        result = self.config.map_openai_params(
            non_default_params={"n": 3, "aspect_ratio": "16:9", "seed": 7},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result == {"aspect_ratio": "16:9", "seed": 7}

    @pytest.mark.parametrize("size", ["auto", "wide", "bigxsmall", "1024x"])
    def test_map_openai_params_ignores_invalid_size(self, size):
        result = self.config.map_openai_params(
            non_default_params={"size": size},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result == {}

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
        assert result == f"{DEFAULT_API_BASE}/{PREDICTIONS_ENDPOINT}"

    @patch(MODULE)
    def test_get_complete_url_no_double_endpoint(self, mock_secret):
        mock_secret.return_value = None
        base = f"{DEFAULT_API_BASE}/{PREDICTIONS_ENDPOINT}"
        result = self.config.get_complete_url(
            api_base=base,
            api_key="k",
            model=self.model,
            optional_params={},
            litellm_params={},
        )
        assert result == base

    @patch(MODULE)
    def test_validate_environment_sets_pruna_headers(self, mock_secret):
        headers = self.config.validate_environment(
            headers={},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="my_key",
        )
        assert headers["apikey"] == "my_key"
        assert headers["Model"] == "p-image"
        assert headers["Try-Sync"] == "true"
        assert headers["Content-Type"] == "application/json"
        mock_secret.assert_not_called()

    @patch(MODULE)
    def test_validate_environment_env_fallback(self, mock_secret):
        mock_secret.return_value = "env_key"
        headers = self.config.validate_environment(
            headers={},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
        )
        assert headers["apikey"] == "env_key"

    @patch(MODULE)
    def test_validate_environment_missing_key_raises(self, mock_secret):
        mock_secret.return_value = None
        with pytest.raises(ValueError, match="PRUNA_API_KEY is not set"):
            self.config.validate_environment(
                headers={},
                model=self.model,
                messages=[],
                optional_params={},
                litellm_params={},
            )

    def test_transform_request_wraps_input(self):
        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt="a lion at sunset",
            optional_params={"aspect_ratio": "16:9", "seed": 7},
            litellm_params={},
            headers={},
        )
        assert result == {"input": {"prompt": "a lion at sunset", "aspect_ratio": "16:9", "seed": 7}}

    def test_transform_response_builds_absolute_url(self):
        raw = _response(
            200,
            json_body={"status": "succeeded", "generation_url": "/v1/predictions/delivery/abc/output.jpg"},
        )
        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=raw,
            model_response=litellm.ImageResponse(),
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert [img.url for img in result.data] == ["https://api.pruna.ai/v1/predictions/delivery/abc/output.jpg"]

    def test_transform_response_keeps_absolute_url(self):
        raw = _response(
            200,
            json_body={"status": "succeeded", "generation_url": "https://cdn.pruna.ai/out.jpg"},
        )
        result = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=raw,
            model_response=litellm.ImageResponse(),
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert [img.url for img in result.data] == ["https://cdn.pruna.ai/out.jpg"]

    def test_transform_response_async_not_completed_raises(self):
        raw = _response(
            200,
            json_body={"id": "abc", "get_url": "https://api.pruna.ai/v1/predictions/status/abc"},
        )
        with pytest.raises(BaseLLMException):
            self.config.transform_image_generation_response(
                model=self.model,
                raw_response=raw,
                model_response=litellm.ImageResponse(),
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

    def test_transform_response_non_200_raises(self):
        raw = _response(401, text_body=b"unauthorized")
        with pytest.raises(BaseLLMException):
            self.config.transform_image_generation_response(
                model=self.model,
                raw_response=raw,
                model_response=litellm.ImageResponse(),
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

    def test_transform_response_json_parse_error_raises(self):
        raw = _response(200, text_body=b"not json")
        with pytest.raises(BaseLLMException):
            self.config.transform_image_generation_response(
                model=self.model,
                raw_response=raw,
                model_response=litellm.ImageResponse(),
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

    def test_image_generation_dispatches_to_pruna_handler(self):
        fake_response = MagicMock()

        with patch.object(
            litellm.images.main.llm_http_handler,
            "image_generation_handler",
            return_value=fake_response,
        ) as mock_handler:
            result = litellm.image_generation(
                model="pruna/p-image",
                prompt="a lion at sunset",
                api_key="sk-test",
            )

        assert result is fake_response
        mock_handler.assert_called_once()
        kwargs = mock_handler.call_args.kwargs
        assert kwargs["custom_llm_provider"] == "pruna"
        assert kwargs["model"] == "p-image"
        assert kwargs["prompt"] == "a lion at sunset"
        assert isinstance(kwargs["image_generation_provider_config"], PrunaImageGenerationConfig)
