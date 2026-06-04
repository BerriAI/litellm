import os
import sys

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.azure.azure import AzureChatCompletion
from litellm.llms.azure.image_generation.http_utils import (
    azure_deployment_image_generation_json_body,
)
from litellm.llms.azure_ai.image_generation import (
    AzureFoundryMAIImageGenerationConfig,
    get_azure_ai_image_generation_config,
)
from litellm.utils import get_optional_params_image_gen


class TestAzureMAIImageGeneration:
    def test_is_mai_model(self):
        assert AzureFoundryMAIImageGenerationConfig.is_mai_model("MAI-Image-2.5")
        assert AzureFoundryMAIImageGenerationConfig.is_mai_model(
            "azure_ai/MAI-Image-2.5"
        )
        assert not AzureFoundryMAIImageGenerationConfig.is_mai_model("flux.2-pro")
        assert not AzureFoundryMAIImageGenerationConfig.is_mai_model("MAI-DS-R1")

    def test_get_mai_image_generation_url(self):
        url = AzureFoundryMAIImageGenerationConfig.get_mai_image_generation_url(
            api_base="https://my-resource.services.ai.azure.com",
            api_version="preview",
        )
        assert (
            url
            == "https://my-resource.services.ai.azure.com/mai/v1/images/generations?api-version=preview"
        )

    def test_get_mai_image_generation_url_preserves_full_path(self):
        api = (
            "https://my-resource.services.ai.azure.com/mai/v1/images/generations"
            "?api-version=preview"
        )
        url = AzureFoundryMAIImageGenerationConfig.get_mai_image_generation_url(
            api_base=api,
            api_version="preview",
        )
        assert url == api

    def test_get_azure_ai_image_generation_config_returns_mai(self):
        config = get_azure_ai_image_generation_config("MAI-Image-2.5")
        assert isinstance(config, AzureFoundryMAIImageGenerationConfig)

    def test_map_openai_params_size_to_width_height(self):
        config = AzureFoundryMAIImageGenerationConfig()
        optional_params = config.map_openai_params(
            non_default_params={"size": "1024x1024", "n": 1},
            optional_params={},
            model="MAI-Image-2.5",
            drop_params=True,
        )
        assert optional_params["width"] == 1024
        assert optional_params["height"] == 1024
        assert optional_params["n"] == 1
        assert "size" not in optional_params

    def test_map_openai_params_defaults(self):
        config = AzureFoundryMAIImageGenerationConfig()
        optional_params = config.map_openai_params(
            non_default_params={},
            optional_params={},
            model="MAI-Image-2.5",
            drop_params=True,
        )
        assert optional_params["width"] == 1024
        assert optional_params["height"] == 1024

    def test_get_optional_params_image_gen_mai(self):
        config = AzureFoundryMAIImageGenerationConfig()
        optional_params = get_optional_params_image_gen(
            model="MAI-Image-2.5",
            size="1792x1024",
            n=1,
            custom_llm_provider="azure_ai",
            provider_config=config,
            drop_params=True,
        )
        assert optional_params["width"] == 1792
        assert optional_params["height"] == 1024
        assert "size" not in optional_params

    def test_azure_create_azure_base_url_mai(self):
        azure_chat = AzureChatCompletion()
        url = azure_chat.create_azure_base_url(
            azure_client_params={
                "azure_endpoint": "https://my-resource.services.ai.azure.com",
                "api_version": "preview",
            },
            model="MAI-Image-2.5",
        )
        assert "/mai/v1/images/generations" in url
        assert "api-version=preview" in url

    def test_mai_json_body_keeps_model(self):
        api = (
            "https://my-resource.services.ai.azure.com/mai/v1/images/generations"
            "?api-version=preview"
        )
        data = {
            "model": "MAI-Image-2.5",
            "prompt": "A photograph of a red fox",
            "width": 1024,
            "height": 1024,
            "n": 1,
        }
        out = azure_deployment_image_generation_json_body(api, data)
        assert out == data
