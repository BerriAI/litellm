import io
from unittest.mock import MagicMock

import httpx
import pytest


import litellm
from litellm.images.utils import ImageEditRequestUtils
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

    def test_get_optional_params_image_edit_size_raises_400(self, monkeypatch):
        monkeypatch.setattr(litellm, "drop_params", False)
        with pytest.raises(litellm.UnsupportedParamsError, match="size") as exc_info:
            ImageEditRequestUtils.get_optional_params_image_edit(
                model="MAI-Image-2.5",
                image_edit_provider_config=AzureFoundryMAIImageEditConfig(),
                image_edit_optional_params={"size": "1024x1024", "n": 1},
            )
        assert exc_info.value.status_code == 400

    def test_get_optional_params_image_edit_size_dropped_with_drop_params(self, monkeypatch):
        monkeypatch.setattr(litellm, "drop_params", False)
        optional_params = ImageEditRequestUtils.get_optional_params_image_edit(
            model="MAI-Image-2.5",
            image_edit_provider_config=AzureFoundryMAIImageEditConfig(),
            image_edit_optional_params={"size": "1024x1024", "n": 1},
            drop_params=True,
        )
        assert "size" not in optional_params
        assert optional_params["n"] == 1

    def test_get_optional_params_image_edit_without_size_forwards_nothing_extra(self, monkeypatch):
        monkeypatch.setattr(litellm, "drop_params", False)
        optional_params = ImageEditRequestUtils.get_optional_params_image_edit(
            model="MAI-Image-2.5",
            image_edit_provider_config=AzureFoundryMAIImageEditConfig(),
            image_edit_optional_params={},
        )
        assert optional_params == {}

    def test_image_edit_size_surfaces_as_400(self, monkeypatch):
        monkeypatch.setattr(litellm, "drop_params", False)
        with pytest.raises(litellm.BadRequestError) as exc_info:
            litellm.image_edit(
                model="azure_ai/MAI-Image-2.5",
                image=io.BytesIO(b"fake-image-bytes"),
                prompt="Turn this into a studio product shot",
                size="1024x1024",
                api_key="test-key",
                api_base="https://my-resource.services.ai.azure.com",
            )
        assert exc_info.value.status_code == 400

    def test_transform_image_edit_request_uses_image_field(self):
        config = AzureFoundryMAIImageEditConfig()
        image_bytes = io.BytesIO(b"fake-image-bytes")

        data, files = config.transform_image_edit_request(
            model="MAI-Image-2.5",
            prompt="Turn this into a studio product shot",
            image=image_bytes,
            image_edit_optional_request_params={"n": 1},
            litellm_params={},
            headers={},
        )

        assert data["model"] == "MAI-Image-2.5"
        assert data["prompt"] == "Turn this into a studio product shot"
        assert "size" not in data
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


def test_mai_validate_environment_with_entra_token(monkeypatch):
    monkeypatch.delenv("AZURE_AI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)

    headers = AzureFoundryMAIImageEditConfig().validate_environment(
        headers={},
        model="MAI-Image-2.5",
        litellm_params={"azure_ad_token": "entra-token"},
    )

    assert headers == {"Authorization": "Bearer entra-token"}
