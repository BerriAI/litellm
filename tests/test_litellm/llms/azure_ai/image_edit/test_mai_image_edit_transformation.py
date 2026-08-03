import io
import os
import sys
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.azure_ai.image_edit import (
    AzureFoundryMAIImageEditConfig,
    get_azure_ai_image_edit_config,
)
from litellm.llms.azure_ai.image_generation.mai_transformation import (
    AzureFoundryMAIImageGenerationConfig,
)


class TestAzureMAIImageEdit:
    def test_get_mai_image_edit_url(self):
        url = AzureFoundryMAIImageGenerationConfig.get_mai_image_edit_url(
            api_base="https://my-resource.services.ai.azure.com",
            api_version="preview",
        )
        assert (
            url
            == "https://my-resource.services.ai.azure.com/mai/v1/images/edits?api-version=preview"
        )

    def test_get_mai_image_edit_url_rewrites_generation_url(self):
        url = AzureFoundryMAIImageGenerationConfig.get_mai_image_edit_url(
            api_base=(
                "https://my-resource.services.ai.azure.com/mai/v1/images/generations"
                "?api-version=preview"
            ),
            api_version="preview",
        )
        assert (
            url
            == "https://my-resource.services.ai.azure.com/mai/v1/images/edits?api-version=preview"
        )

    def test_get_mai_image_edit_url_appends_edits_to_mai_root(self):
        url = AzureFoundryMAIImageGenerationConfig.get_mai_image_edit_url(
            api_base="https://my-resource.services.ai.azure.com/mai/v1",
            api_version="preview",
        )
        assert (
            url
            == "https://my-resource.services.ai.azure.com/mai/v1/images/edits?api-version=preview"
        )

    def test_get_azure_ai_image_edit_config_returns_mai(self):
        config = get_azure_ai_image_edit_config("MAI-Image-2.5")
        assert isinstance(config, AzureFoundryMAIImageEditConfig)

    def test_validate_environment_uses_api_key_header(self):
        config = AzureFoundryMAIImageEditConfig()
        headers: dict = {}
        config.validate_environment(headers, "MAI-Image-2.5", api_key="test-key")
        assert headers["api-key"] == "test-key"
        assert "Api-Key" not in headers

    def test_get_complete_url(self):
        config = AzureFoundryMAIImageEditConfig()
        url = config.get_complete_url(
            model="MAI-Image-2.5",
            api_base="https://my-resource.services.ai.azure.com",
            litellm_params={"api_version": "preview"},
        )
        assert "/mai/v1/images/edits" in url
        assert "api-version=preview" in url

    def test_map_openai_params_keeps_size(self):
        config = AzureFoundryMAIImageEditConfig()
        optional_params = config.map_openai_params(
            image_edit_optional_params={"size": "1792x1024", "n": 1},
            model="MAI-Image-2.5",
            drop_params=True,
        )
        assert optional_params["size"] == "1792x1024"
        assert optional_params["n"] == 1
        assert "width" not in optional_params
        assert "height" not in optional_params

    def test_map_openai_params_defaults_size(self):
        config = AzureFoundryMAIImageEditConfig()
        optional_params = config.map_openai_params(
            image_edit_optional_params={},
            model="MAI-Image-2.5",
            drop_params=True,
        )
        assert optional_params["size"] == "1024x1024"

    def test_map_openai_params_unsupported_size_raises(self):
        config = AzureFoundryMAIImageEditConfig()
        with pytest.raises(ValueError, match="Unsupported size value: 'auto'"):
            config.map_openai_params(
                image_edit_optional_params={"size": "auto"},
                model="MAI-Image-2.5",
                drop_params=True,
            )

    def test_map_openai_params_invalid_size_format_raises(self):
        config = AzureFoundryMAIImageEditConfig()
        with pytest.raises(ValueError, match="Invalid size format: '1024xabc'"):
            config.map_openai_params(
                image_edit_optional_params={"size": "1024xabc"},
                model="MAI-Image-2.5",
                drop_params=True,
            )

    def test_transform_image_edit_request_uses_image_field(self):
        config = AzureFoundryMAIImageEditConfig()
        image_bytes = io.BytesIO(b"fake-image-bytes")

        data, files = config.transform_image_edit_request(
            model="MAI-Image-2.5",
            prompt="Turn this into a studio product shot",
            image=image_bytes,
            image_edit_optional_request_params={"size": "1024x1024", "n": 1},
            litellm_params={},
            headers={},
        )

        assert data["model"] == "MAI-Image-2.5"
        assert data["prompt"] == "Turn this into a studio product shot"
        assert data["size"] == "1024x1024"
        assert data["n"] == 1
        assert len(files) == 1
        assert files[0][0] == "image"
        assert files[0][0] != "image[]"

    def test_normalize_mai_image_usage_maps_edit_response_fields(self):
        usage = AzureFoundryMAIImageGenerationConfig.normalize_mai_image_usage(
            {
                "num_output_tokens": 1024,
                "output_image_tokens": 1024,
            }
        )
        assert usage["output_tokens"] == 1024
        assert usage["input_tokens"] == 0
        assert usage["total_tokens"] == 1024
        assert usage["input_tokens_details"]["text_tokens"] == 0
        assert usage["input_tokens_details"]["image_tokens"] == 0

    def test_transform_image_edit_response_parses_mai_usage(self):
        config = AzureFoundryMAIImageEditConfig()
        raw_response = MagicMock(spec=httpx.Response)
        raw_response.status_code = 200
        raw_response.text = ""
        raw_response.json.return_value = {
            "created": 1780897477,
            "data": [{"b64_json": "abc123"}],
            "usage": {
                "num_output_tokens": 1024,
                "output_image_tokens": 1024,
            },
        }

        logging_obj = MagicMock()
        image_response = config.transform_image_edit_response(
            model="MAI-Image-2.5",
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert image_response.data[0].b64_json == "abc123"
        assert image_response.usage.output_tokens == 1024
        assert image_response.usage.total_tokens == 1024
