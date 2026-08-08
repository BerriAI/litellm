"""
Unit tests for the MiniMax image generation configuration.

These tests validate the MinimaxImageGenerationConfig class which handles
transformation between OpenAI-compatible image generation params and the
MiniMax image generation API (POST /v1/image_generation).
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.minimax.image_generation.transformation import (
    MinimaxImageGenerationConfig,
)
from litellm.types.utils import ImageResponse


class TestMinimaxImageGenerationTransformation:
    def setup_method(self):
        self.config = MinimaxImageGenerationConfig()
        self.model = "image-01"
        self.logging_obj = MagicMock()

    def test_get_supported_openai_params(self):
        supported_params = self.config.get_supported_openai_params(self.model)

        assert "n" in supported_params
        assert "size" in supported_params
        assert "response_format" in supported_params
        assert "seed" in supported_params
        assert "aspect_ratio" in supported_params

    def test_map_openai_params_passthrough(self):
        non_default_params = {
            "n": 2,
            "seed": 42,
            "aspect_ratio": "16:9",
        }

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result["n"] == 2
        assert result["seed"] == 42
        assert result["aspect_ratio"] == "16:9"

    def test_map_openai_params_size_to_width_height(self):
        result = self.config.map_openai_params(
            non_default_params={"size": "1024x1024"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result["width"] == 1024
        assert result["height"] == 1024
        assert "size" not in result

    def test_map_openai_params_unsupported_size_is_dropped(self):
        result = self.config.map_openai_params(
            non_default_params={"size": "100x50"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert "width" not in result
        assert "height" not in result

    def test_map_openai_params_response_format_b64_json(self):
        result = self.config.map_openai_params(
            non_default_params={"response_format": "b64_json"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result["response_format"] == "base64"

    def test_map_openai_params_response_format_url(self):
        result = self.config.map_openai_params(
            non_default_params={"response_format": "url"},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result["response_format"] == "url"

    def test_get_complete_url_default(self):
        result = self.config.get_complete_url(
            api_base=None,
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert result == "https://api.minimax.io/v1/image_generation"

    def test_get_complete_url_with_custom_base(self):
        result = self.config.get_complete_url(
            api_base="https://api.minimaxi.com",
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert result == "https://api.minimaxi.com/v1/image_generation"

    def test_get_complete_url_with_full_endpoint_base(self):
        result = self.config.get_complete_url(
            api_base="https://api.minimax.io/v1/image_generation",
            api_key="test_key",
            model=self.model,
            optional_params={},
            litellm_params={},
        )

        assert result == "https://api.minimax.io/v1/image_generation"

    @patch("litellm.llms.minimax.image_generation.transformation.get_secret_str")
    def test_validate_environment(self, mock_get_secret):
        mock_get_secret.return_value = "test_api_key"
        headers = {}

        result = self.config.validate_environment(
            headers=headers,
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )

        assert result["Authorization"] == "Bearer test_api_key"
        assert result["Content-Type"] == "application/json"

    @patch("litellm.llms.minimax.image_generation.transformation.get_secret_str")
    def test_validate_environment_missing_api_key(self, mock_get_secret):
        mock_get_secret.return_value = None

        with pytest.raises(ValueError):
            self.config.validate_environment(
                headers={},
                model=self.model,
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

    def test_transform_image_generation_request(self):
        optional_params = {
            "n": 2,
            "response_format": "url",
            "prompt_optimizer": True,
        }

        request_data = self.config.transform_image_generation_request(
            model=self.model,
            prompt="a red apple",
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert request_data["model"] == "image-01"
        assert request_data["prompt"] == "a red apple"
        assert request_data["n"] == 2
        assert request_data["response_format"] == "url"
        assert request_data["prompt_optimizer"] is True

    def test_transform_image_generation_request_merges_extra_body(self):
        optional_params = {
            "extra_body": {"seed": 7, "prompt_optimizer": True},
        }

        request_data = self.config.transform_image_generation_request(
            model=self.model,
            prompt="a red apple",
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert request_data["seed"] == 7
        assert request_data["prompt_optimizer"] is True

    def test_transform_image_generation_response_urls(self):
        raw_response = self._make_response(
            {
                "data": {"image_urls": ["https://example.com/a.png", "https://example.com/b.png"]},
                "metadata": {"success_count": 2, "failed_count": 0},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }
        )

        model_response = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=raw_response,
            model_response=ImageResponse(),
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(model_response.data) == 2
        assert model_response.data[0].url == "https://example.com/a.png"
        assert model_response.data[1].url == "https://example.com/b.png"

    def test_transform_image_generation_response_base64(self):
        raw_response = self._make_response(
            {
                "data": {"image_base64": ["aGVsbG8=", "d29ybGQ="]},
                "metadata": {"success_count": 2, "failed_count": 0},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }
        )

        model_response = self.config.transform_image_generation_response(
            model=self.model,
            raw_response=raw_response,
            model_response=ImageResponse(),
            logging_obj=self.logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert len(model_response.data) == 2
        assert model_response.data[0].b64_json == "aGVsbG8="
        assert model_response.data[1].b64_json == "d29ybGQ="

    def test_transform_image_generation_response_error_status_code(self):
        raw_response = self._make_response(
            {
                "base_resp": {"status_code": 1004, "status_msg": "invalid api key"},
            }
        )

        with pytest.raises(Exception):
            self.config.transform_image_generation_response(
                model=self.model,
                raw_response=raw_response,
                model_response=ImageResponse(),
                logging_obj=self.logging_obj,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

    @staticmethod
    def _make_response(payload: dict) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json=payload,
            request=httpx.Request("POST", "https://api.minimax.io/v1/image_generation"),
        )
