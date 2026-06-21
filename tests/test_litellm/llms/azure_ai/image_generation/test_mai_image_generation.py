import os
import sys
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

import litellm
from litellm.llms.azure.azure import AzureChatCompletion
from litellm.llms.azure.image_generation import get_azure_image_generation_config
from litellm.llms.azure.image_generation.http_utils import (
    azure_deployment_image_generation_json_body,
)
from litellm.llms.azure_ai.image_generation import (
    AzureFoundryMAIImageGenerationConfig,
    get_azure_ai_image_generation_config,
)
from litellm.llms.azure_ai.image_generation.cost_calculator import (
    cost_calculator as azure_ai_image_cost_calculator,
)
from litellm.types.utils import (
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
)
from litellm.utils import get_optional_params_image_gen


class TestAzureMAIImageGeneration:
    def test_is_mai_model(self):
        assert AzureFoundryMAIImageGenerationConfig.is_mai_model("MAI-Image-2.5")
        assert AzureFoundryMAIImageGenerationConfig.is_mai_model(
            "azure_ai/MAI-Image-2.5"
        )
        assert AzureFoundryMAIImageGenerationConfig.is_mai_model("MAI-Image-2.5-Flash")
        assert AzureFoundryMAIImageGenerationConfig.is_mai_model("MAI-Image-2e")
        assert not AzureFoundryMAIImageGenerationConfig.is_mai_model("flux.2-pro")
        assert not AzureFoundryMAIImageGenerationConfig.is_mai_model("MAI-DS-R1")

    def test_mai_flash_and_2e_model_pricing_in_cost_map(self):
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        flash_info = litellm.get_model_info(
            model="azure_ai/MAI-Image-2.5-Flash",
            custom_llm_provider="azure_ai",
        )
        assert flash_info["input_cost_per_token"] == 1.75e-06
        assert flash_info["input_cost_per_image_token"] == 1.75e-06
        assert flash_info["output_cost_per_image_token"] == 3.3e-05

        image_2e_info = litellm.get_model_info(
            model="azure_ai/MAI-Image-2e",
            custom_llm_provider="azure_ai",
        )
        assert image_2e_info["input_cost_per_token"] == 5e-06
        assert image_2e_info["output_cost_per_image_token"] == 1.95e-05

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

    def test_get_mai_image_generation_url_appends_generations_to_mai_root(self):
        url = AzureFoundryMAIImageGenerationConfig.get_mai_image_generation_url(
            api_base="https://my-resource.services.ai.azure.com/mai/v1",
            api_version="preview",
        )
        assert (
            url
            == "https://my-resource.services.ai.azure.com/mai/v1/images/generations?api-version=preview"
        )

    def test_get_azure_ai_image_generation_config_returns_mai(self):
        config = get_azure_ai_image_generation_config("MAI-Image-2.5")
        assert isinstance(config, AzureFoundryMAIImageGenerationConfig)

    def test_azure_image_generation_config_returns_mai(self):
        config = get_azure_image_generation_config("MAI-Image-2.5")
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

    def test_map_openai_params_custom_size(self):
        config = AzureFoundryMAIImageGenerationConfig()
        optional_params = config.map_openai_params(
            non_default_params={"size": "768x768"},
            optional_params={},
            model="MAI-Image-2.5",
            drop_params=True,
        )
        assert optional_params["width"] == 768
        assert optional_params["height"] == 768

    def test_map_openai_params_width_only_gets_height_default(self):
        config = AzureFoundryMAIImageGenerationConfig()
        optional_params = config.map_openai_params(
            non_default_params={"width": 1792},
            optional_params={},
            model="MAI-Image-2.5",
            drop_params=True,
        )
        assert optional_params["width"] == 1792
        assert optional_params["height"] == config.DEFAULT_HEIGHT

    def test_map_openai_params_height_only_gets_width_default(self):
        config = AzureFoundryMAIImageGenerationConfig()
        optional_params = config.map_openai_params(
            non_default_params={"height": 1792},
            optional_params={},
            model="MAI-Image-2.5",
            drop_params=True,
        )
        assert optional_params["width"] == config.DEFAULT_WIDTH
        assert optional_params["height"] == 1792

    def test_map_openai_params_unsupported_size_raises(self):
        config = AzureFoundryMAIImageGenerationConfig()
        with pytest.raises(ValueError, match="Unsupported size value: 'auto'"):
            config.map_openai_params(
                non_default_params={"size": "auto"},
                optional_params={},
                model="MAI-Image-2.5",
                drop_params=True,
            )

    def test_map_openai_params_invalid_custom_size_raises(self):
        config = AzureFoundryMAIImageGenerationConfig()
        with pytest.raises(ValueError, match="Invalid size format: '1024xabc'"):
            config.map_openai_params(
                non_default_params={"size": "1024xabc"},
                optional_params={},
                model="MAI-Image-2.5",
                drop_params=True,
            )

    def test_map_openai_params_unsupported_param_raises(self):
        config = AzureFoundryMAIImageGenerationConfig()
        with pytest.raises(ValueError, match="Parameter quality is not supported"):
            config.map_openai_params(
                non_default_params={"quality": "hd"},
                optional_params={},
                model="MAI-Image-2.5",
                drop_params=False,
            )

    def test_transform_image_generation_response_normalizes_mai_usage(self):
        config = AzureFoundryMAIImageGenerationConfig()
        raw_response = MagicMock(spec=httpx.Response)
        raw_response.json.return_value = {
            "created": 1780897477,
            "data": [{"b64_json": "abc123"}],
            "usage": {
                "num_output_tokens": 1024,
                "num_input_text_tokens": 22,
                "output_image_tokens": 1024,
            },
        }

        logging_obj = MagicMock()
        image_response = config.transform_image_generation_response(
            model="MAI-Image-2.5",
            raw_response=raw_response,
            model_response=ImageResponse(),
            logging_obj=logging_obj,
            request_data={"prompt": "A red fox"},
            optional_params={"width": 1024, "height": 1024},
            litellm_params={},
            encoding=None,
        )

        assert image_response.data[0].b64_json == "abc123"
        assert image_response.usage.output_tokens == 1024
        assert image_response.usage.input_tokens == 22
        assert image_response.usage.total_tokens == 1046

    def test_transform_image_generation_response_non_json_raises_openai_error(self):
        from litellm.llms.openai.common_utils import OpenAIError

        config = AzureFoundryMAIImageGenerationConfig()
        raw_response = MagicMock(spec=httpx.Response)
        raw_response.json.side_effect = ValueError("not json")
        raw_response.text = "upstream gateway error"
        raw_response.status_code = 502

        with pytest.raises(OpenAIError) as exc_info:
            config.transform_image_generation_response(
                model="MAI-Image-2.5",
                raw_response=raw_response,
                model_response=ImageResponse(),
                logging_obj=MagicMock(),
                request_data={"prompt": "A red fox"},
                optional_params={"width": 1024, "height": 1024},
                litellm_params={},
                encoding=None,
            )

        assert exc_info.value.status_code == 502
        assert exc_info.value.message == "upstream gateway error"

    def test_normalize_mai_usage_preserves_zero_output_tokens(self):
        config = AzureFoundryMAIImageGenerationConfig()
        normalized = config.normalize_mai_image_usage(
            {
                "num_output_tokens": 0,
                "output_image_tokens": 1024,
                "num_input_text_tokens": 22,
            }
        )
        assert normalized["output_tokens"] == 0
        assert normalized["input_tokens"] == 22
        assert normalized["total_tokens"] == 22

    def test_azure_sync_image_generation_uses_mai_response_transform(self):
        raw_response = MagicMock(spec=httpx.Response)
        raw_response.json.return_value = {
            "created": 1780897477,
            "data": [{"b64_json": "abc123"}],
            "usage": {
                "num_output_tokens": 1024,
                "num_input_text_tokens": 22,
            },
        }

        class MAIImageGenerationAzureChatCompletion(AzureChatCompletion):
            def make_sync_azure_httpx_request(self, **kwargs):
                return raw_response

        logging_obj = MagicMock()
        image_response = MAIImageGenerationAzureChatCompletion().image_generation(
            prompt="A red fox",
            timeout=60.0,
            optional_params={"width": 1792, "height": 1024},
            logging_obj=logging_obj,
            headers={},
            model="MAI-Image-2.5",
            api_key="test-key",
            api_base="https://my-resource.services.ai.azure.com",
            api_version="preview",
            litellm_params={},
        )

        assert image_response.data[0].b64_json == "abc123"
        assert image_response.usage.output_tokens == 1024
        assert image_response.usage.input_tokens == 22
        assert image_response.usage.total_tokens == 1046
        assert image_response.size == "1792x1024"

    def test_mai_image_cost_calculator_token_based(self):
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        model = "azure_ai/MAI-Image-2.5"
        model_info = litellm.get_model_info(model=model, custom_llm_provider="azure_ai")
        input_text_tokens = 100
        output_image_tokens = 1024

        image_response = ImageResponse(
            data=[ImageObject(b64_json="img1")],
            usage=ImageUsage(
                input_tokens=input_text_tokens,
                input_tokens_details=ImageUsageInputTokensDetails(
                    text_tokens=input_text_tokens,
                    image_tokens=0,
                ),
                output_tokens=output_image_tokens,
                total_tokens=input_text_tokens + output_image_tokens,
            ),
        )

        cost = azure_ai_image_cost_calculator(
            model=model,
            image_response=image_response,
        )

        expected_cost = (
            input_text_tokens * model_info["input_cost_per_token"]
            + output_image_tokens * model_info["output_cost_per_image_token"]
        )
        assert round(cost, 10) == round(expected_cost, 10)

    def test_mai_image_cost_calculator_falls_back_to_flat_image_pricing(self):
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        model = "azure_ai/MAI-Image-2.5"
        model_info = litellm.get_model_info(model=model, custom_llm_provider="azure_ai")
        image_response = ImageResponse(
            data=[ImageObject(b64_json="img1"), ImageObject(b64_json="img2")]
        )

        cost = azure_ai_image_cost_calculator(
            model=model,
            image_response=image_response,
        )

        assert (
            cost == len(image_response.data or []) * model_info["output_cost_per_image"]
        )
        assert cost > 0
