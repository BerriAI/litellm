import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm import get_llm_provider
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.novita.image_generation.transformation import (
    DEFAULT_API_BASE,
    NovitaImageGenerationConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

MODULE = "litellm.llms.novita.image_generation.transformation.get_secret_str"


class TestNovitaImageGenerationTransformation:
    def setup_method(self):
        self.config = NovitaImageGenerationConfig()
        self.model = "seedream-4.0"
        self.logging_obj = MagicMock()

    def test_provider_routing(self):
        model, provider, _, _ = get_llm_provider("novita/seedream-4.0")
        assert provider == "novita"
        assert model == "seedream-4.0"

    def test_provider_config_registered(self):
        config = ProviderConfigManager.get_provider_image_generation_config(
            model=self.model,
            provider=LlmProviders.NOVITA,
        )
        assert isinstance(config, NovitaImageGenerationConfig)

    def test_supported_params(self):
        assert self.config.get_supported_openai_params(self.model) == ["n", "size"]

    def test_map_openai_params_passthrough(self):
        result = self.config.map_openai_params(
            non_default_params={"size": "2048x2048", "watermark": False},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result == {"size": "2048x2048", "watermark": False}

    def test_map_openai_params_n_expands_to_sequential(self):
        result = self.config.map_openai_params(
            non_default_params={"n": 4, "size": "1024x1024"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result["size"] == "1024x1024"
        assert result["sequential_image_generation"] == "auto"
        assert result["max_images"] == 4
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
    def test_get_complete_url_default(self, mock_secret):
        mock_secret.return_value = None
        result = self.config.get_complete_url(
            api_base=None,
            api_key="k",
            model=self.model,
            optional_params={},
            litellm_params={},
        )
        assert result == f"{DEFAULT_API_BASE}/v3/seedream-4.0"

    @patch(MODULE)
    def test_get_complete_url_strips_chat_suffix(self, mock_secret):
        mock_secret.return_value = None
        result = self.config.get_complete_url(
            api_base="https://api.novita.ai/v3/openai",
            api_key="k",
            model=self.model,
            optional_params={},
            litellm_params={},
        )
        assert result == "https://api.novita.ai/v3/seedream-4.0"

    @patch(MODULE)
    def test_get_complete_url_custom_base(self, mock_secret):
        mock_secret.return_value = None
        result = self.config.get_complete_url(
            api_base="https://proxy.example.com",
            api_key="k",
            model=self.model,
            optional_params={},
            litellm_params={},
        )
        assert result == "https://proxy.example.com/v3/seedream-4.0"

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
        assert headers["X-Novita-Source"] == "litellm"
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
        assert headers["Authorization"] == "Bearer env_key"

    @patch(MODULE)
    def test_validate_environment_missing_key_raises(self, mock_secret):
        mock_secret.return_value = None
        with pytest.raises(ValueError, match="NOVITA_API_KEY is not set"):
            self.config.validate_environment(
                headers={},
                model=self.model,
                messages=[],
                optional_params={},
                litellm_params={},
            )

    def test_transform_request(self):
        result = self.config.transform_image_generation_request(
            model=self.model,
            prompt="a cat surfing",
            optional_params={"size": "1024x1024", "max_images": 3},
            litellm_params={},
            headers={},
        )
        assert result == {"prompt": "a cat surfing", "size": "1024x1024", "max_images": 3}

    def test_transform_response_url_list(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"images": ["https://cdn/img1.png", "https://cdn/img2.png"]}
        model_response = litellm.ImageResponse()

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
        assert [img.url for img in result.data] == ["https://cdn/img1.png", "https://cdn/img2.png"]

    def test_transform_response_image_url_objects(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"images": [{"image_url": "https://cdn/obj.png", "image_url_ttl": 3600}]}
        model_response = litellm.ImageResponse()

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
        assert [img.url for img in result.data] == ["https://cdn/obj.png"]

    def test_transform_response_non_200_raises(self):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "unauthorized"
        mock_response.headers = {}
        with pytest.raises(BaseLLMException):
            self.config.transform_image_generation_response(
                model=self.model,
                raw_response=mock_response,
                model_response=litellm.ImageResponse(),
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

    def test_transform_response_missing_images_raises(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {"task_id": "abc"}
        with pytest.raises(BaseLLMException):
            self.config.transform_image_generation_response(
                model=self.model,
                raw_response=mock_response,
                model_response=litellm.ImageResponse(),
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
        with pytest.raises(BaseLLMException):
            self.config.transform_image_generation_response(
                model=self.model,
                raw_response=mock_response,
                model_response=litellm.ImageResponse(),
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

    def test_image_generation_dispatches_to_novita_handler(self):
        fake_response = MagicMock()

        with patch.object(
            litellm.images.main.llm_http_handler,
            "image_generation_handler",
            return_value=fake_response,
        ) as mock_handler:
            result = litellm.image_generation(
                model="novita/seedream-4.0",
                prompt="a cat surfing a wave",
                api_key="sk-test",
            )

        assert result is fake_response
        mock_handler.assert_called_once()
        kwargs = mock_handler.call_args.kwargs
        assert kwargs["custom_llm_provider"] == "novita"
        assert kwargs["model"] == "seedream-4.0"
        assert kwargs["prompt"] == "a cat surfing a wave"
        assert isinstance(kwargs["image_generation_provider_config"], NovitaImageGenerationConfig)
