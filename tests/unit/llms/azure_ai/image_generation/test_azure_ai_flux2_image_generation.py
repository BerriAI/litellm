from collections.abc import Mapping
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils
from litellm.llms.azure.azure import AzureChatCompletion
from litellm.llms.azure.image_generation import get_azure_image_generation_config
from litellm.llms.azure.image_generation.http_utils import azure_deployment_image_generation_json_body
from litellm.llms.azure_ai.image_generation.flux_transformation import (
    AzureFoundryFluxImageGenerationConfig,
)
from litellm.types.utils import ImageObject, ImageResponse
from litellm.utils import _invalidate_model_cost_lowercase_map, get_optional_params_image_gen

_PRICE: Final = TypeAdapter(float)


@pytest.fixture(autouse=True)
def use_local_model_cost_map(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", GetModelCostMap.load_local_model_cost_map())
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()
    yield
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()


@pytest.mark.parametrize(
    ("model", "provider_path"),
    [
        ("FLUX.2-flex", "flux-2-flex"),
        ("FLUX.2-pro", "flux-2-pro"),
    ],
)
def test_flux2_uses_model_specific_provider_url(model: str, provider_path: str):
    url = AzureChatCompletion().create_azure_base_url(
        azure_client_params={
            "azure_endpoint": "https://example.services.ai.azure.com/",
            "api_version": "preview",
        },
        model=model,
    )

    assert (
        url == f"https://example.services.ai.azure.com/providers/blackforestlabs/v1/{provider_path}?api-version=preview"
    )


def test_flux2_flex_maps_openai_and_provider_parameters():
    config = AzureFoundryFluxImageGenerationConfig()
    mapped_params = config.map_openai_params(
        non_default_params={
            "n": 2,
            "size": "1536x1024",
            "guidance": 4.5,
            "steps": 32,
            "output_format": "jpeg",
        },
        optional_params={},
        model="FLUX.2-flex",
        drop_params=False,
    )
    url = config.get_flux2_image_generation_url(
        api_base="https://example.services.ai.azure.com",
        model="FLUX.2-flex",
        api_version="preview",
    )
    request = azure_deployment_image_generation_json_body(
        api_base=url,
        data={"model": "FLUX.2-flex", "prompt": "A red fox", **mapped_params},
        deployment_name="FLUX.2-flex",
    )

    assert request == {
        "model": "FLUX.2-flex",
        "prompt": "A red fox",
        "num_images": 2,
        "width": 1536,
        "height": 1024,
        "guidance": 4.5,
        "steps": 32,
        "output_format": "jpeg",
    }


def test_flux2_flex_rejects_invalid_size_as_bad_request():
    with pytest.raises(litellm.BadRequestError, match="Expected 'WxH'") as raised:
        get_optional_params_image_gen(
            model="FLUX.2-flex",
            custom_llm_provider="azure_ai",
            provider_config=AzureFoundryFluxImageGenerationConfig(),
            size="large",
        )

    assert raised.value.status_code == 400


@pytest.mark.parametrize("model", ("FLUX.2-pro", "FLUX.2-flex"))
def test_flux2_accepts_and_drops_openai_only_image_parameters(model: str):
    optional_params: Final = get_optional_params_image_gen(
        model=model,
        custom_llm_provider="azure_ai",
        provider_config=AzureFoundryFluxImageGenerationConfig(),
        n=1,
        size="auto",
        quality="high",
        user="end-user-1",
        background="transparent",
        moderation="low",
        output_compression=50,
    )

    assert optional_params == {"num_images": 1}


def test_flux2_flex_model_info():
    model_info = litellm.get_model_info(
        model="FLUX.2-flex",
        custom_llm_provider="azure_ai",
    )
    catalog_info = litellm.model_cost["azure_ai/FLUX.2-flex"]

    assert model_info["mode"] == "image_generation"
    assert model_info["max_input_tokens"] == 32000
    assert model_info["max_tokens"] == 32000
    assert model_info["supported_endpoints"] == ["/v1/images/generations", "/v1/images/edits"]
    assert catalog_info["input_cost_per_pixel"] == 5e-08
    assert catalog_info["supported_modalities"] == ["text", "image"]
    assert catalog_info["supported_output_modalities"] == ["image"]


def test_flux2_flex_cost_uses_generated_megapixels():
    response = ImageResponse(
        data=[
            ImageObject(url="https://example.com/one.png"),
            ImageObject(url="https://example.com/two.png"),
        ]
    )

    cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="FLUX.2-flex",
        completion_response=response,
        custom_llm_provider="azure_ai",
        size="2048x1024",
        call_type="image_generation",
    )

    assert cost == pytest.approx(5e-08 * 2048 * 1024 * 2)


@pytest.mark.parametrize("model", ("FLUX-1.1-pro", "FLUX.1-Kontext-pro"))
def test_flux1_preserves_existing_openai_parameters(model: str):
    params: Final = {"n": 2, "size": "1536x1024", "quality": "high", "user": "test-user"}

    mapped: Final = AzureFoundryFluxImageGenerationConfig().map_openai_params(
        non_default_params=params,
        optional_params={},
        model=model,
        drop_params=False,
    )

    assert mapped == params


@pytest.mark.parametrize("dimensions", ({"size": "2048x1024"}, {"width": 2048, "height": 1024}))
def test_flux2_cost_uses_mapped_dimensions_after_response_transformation(dimensions: Mapping[str, int | str]):
    params: Final = AzureFoundryFluxImageGenerationConfig().map_openai_params(
        non_default_params={"n": 2, **dimensions},
        optional_params={},
        model="FLUX.2-flex",
        drop_params=False,
    )
    response: Final = get_azure_image_generation_config("FLUX.2-flex").transform_image_generation_response(
        model="FLUX.2-flex",
        raw_response=httpx.Response(200, json={"data": [{"b64_json": "aW1n"}, {"b64_json": "aW1n"}]}),
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={"prompt": "A red fox", **params},
        optional_params=params,
        litellm_params={},
        encoding=None,
    )

    assert litellm.completion_cost(
        model="azure_ai/FLUX.2-flex",
        completion_response=response,
        optional_params=params,
        call_type="image_generation",
    ) == pytest.approx(5e-08 * 2048 * 1024 * 2)


def test_flux2_flex_cost_accepts_lowercase_model_spelling():
    response: Final = ImageResponse(data=[ImageObject(b64_json="aW1n"), ImageObject(b64_json="aW1n")])

    cost: Final = litellm.completion_cost(
        model="azure_ai/flux.2-flex",
        completion_response=response,
        optional_params={"width": 1536, "height": 1024, "num_images": 2},
        call_type="image_generation",
    )

    assert cost == pytest.approx(5e-08 * 1536 * 1024 * 2)


def test_flux2_flex_cost_prefers_deployment_input_cost_per_pixel() -> None:
    response: Final = ImageResponse(data=[ImageObject(b64_json="aW1n"), ImageObject(b64_json="aW1n")])

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="FLUX.2-flex",
        completion_response=response,
        custom_llm_provider="azure_ai",
        size="2048x1024",
        call_type="image_generation",
        model_info={"input_cost_per_pixel": 2e-07},
    )

    assert cost == pytest.approx(2e-07 * 2048 * 1024 * 2)


@pytest.mark.parametrize(
    ("size", "optional_params", "response_size", "per_image"),
    [
        ("512x512", None, None, 0.05 * 0.25),
        ("1024-x-1024", None, None, 0.05),
        ("1920x1080", None, None, 0.05 + 0.02 * (1920 * 1080 / 1_048_576 - 1)),
        ("1024x1024", {"width": 2048, "height": 2048}, None, 0.05 + 0.02 * 3),
        (None, None, "1536x1024", 0.05 + 0.02 * 0.5),
    ],
)
def test_flux2_pro_cost_bills_first_then_additional_fractional_megapixels_per_image(
    size: str | None, optional_params: dict[str, object] | None, response_size: str | None, per_image: float
) -> None:
    response: Final = ImageResponse(
        data=[ImageObject(b64_json="aW1n"), ImageObject(b64_json="aW1n")], size=response_size
    )

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="FLUX.2-pro",
        completion_response=response,
        custom_llm_provider="azure_ai",
        size=size,
        call_type="image_generation",
        optional_params=optional_params,
        model_info={"output_cost_per_first_megapixel": 0.05, "output_cost_per_additional_megapixel": 0.02},
    )

    assert cost == pytest.approx(2 * per_image)


def test_flux2_pro_catalog_row_bills_its_megapixel_tiers() -> None:
    info: Final = litellm.get_model_info("azure_ai/FLUX.2-pro")
    response: Final = ImageResponse(data=[ImageObject(b64_json="aW1n")])

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="FLUX.2-pro",
        completion_response=response,
        custom_llm_provider="azure_ai",
        size="1920x1080",
        call_type="image_generation",
    )

    first: Final = _PRICE.validate_python(info.get("output_cost_per_first_megapixel"))
    additional: Final = _PRICE.validate_python(info.get("output_cost_per_additional_megapixel"))
    assert cost == pytest.approx(first + additional * (1920 * 1080 / 1_048_576 - 1))
    assert cost != pytest.approx(info.get("output_cost_per_image"))


@pytest.mark.parametrize(
    ("deployment_prices", "deployment_first_price", "deployment_additional_price"),
    [
        ({"output_cost_per_first_megapixel": 0.05, "output_cost_per_additional_megapixel": None}, 0.05, None),
        ({"output_cost_per_first_megapixel": None, "output_cost_per_additional_megapixel": 0.02}, None, 0.02),
    ],
)
def test_flux2_pro_partial_deployment_megapixel_price_keeps_the_other_catalog_tier(
    deployment_prices: dict[str, float | None],
    deployment_first_price: float | None,
    deployment_additional_price: float | None,
) -> None:
    info: Final = litellm.get_model_info("azure_ai/FLUX.2-pro")
    first: Final = deployment_first_price or _PRICE.validate_python(info.get("output_cost_per_first_megapixel"))
    additional: Final = deployment_additional_price or _PRICE.validate_python(
        info.get("output_cost_per_additional_megapixel")
    )
    response: Final = ImageResponse(data=[ImageObject(b64_json="aW1n")])

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="FLUX.2-pro",
        completion_response=response,
        custom_llm_provider="azure_ai",
        size="2048x2048",
        call_type="image_generation",
        model_info=deployment_prices,
    )

    assert cost == pytest.approx(first + 3 * additional)


def test_flux2_pro_deployment_flat_image_price_overrides_catalog_megapixel_tiers() -> None:
    response: Final = ImageResponse(data=[ImageObject(b64_json="aW1n"), ImageObject(b64_json="aW1n")])

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="FLUX.2-pro",
        completion_response=response,
        custom_llm_provider="azure_ai",
        size="2048x2048",
        call_type="image_generation",
        model_info={"output_cost_per_image": 0.07},
    )

    assert cost == pytest.approx(0.07 * 2)


def test_flux2_pro_unparseable_size_falls_back_to_flat_image_price() -> None:
    per_image: Final = _PRICE.validate_python(
        litellm.get_model_info("azure_ai/FLUX.2-pro").get("output_cost_per_image")
    )
    response: Final = ImageResponse(data=[ImageObject(b64_json="aW1n"), ImageObject(b64_json="aW1n")])

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="FLUX.2-pro",
        completion_response=response,
        custom_llm_provider="azure_ai",
        size="auto",
        call_type="image_generation",
    )

    assert cost == pytest.approx(per_image * 2)


def test_unlisted_azure_ai_model_bills_deployment_input_cost_per_pixel() -> None:
    response: Final = ImageResponse(data=[ImageObject(b64_json="aW1n"), ImageObject(b64_json="aW1n")])

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="unlisted-flux-deployment",
        completion_response=response,
        custom_llm_provider="azure_ai",
        size="1024x1024",
        call_type="image_generation",
        model_info={"input_cost_per_pixel": 1e-07},
    )

    assert cost == pytest.approx(1e-07 * 1024 * 1024 * 2)


def test_flux2_response_preserves_mapped_dimensions():
    config = AzureFoundryFluxImageGenerationConfig()
    params = config.map_openai_params(
        non_default_params={"size": "2048x1024"}, optional_params={}, model="FLUX.2-flex", drop_params=False
    )
    response = config.transform_image_generation_response(
        model="FLUX.2-flex",
        raw_response=httpx.Response(200, json={"data": [{"b64_json": "aW1n"}]}),
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={"prompt": "A landscape"},
        optional_params=params,
        litellm_params={},
        encoding=None,
    )
    assert response.size == "2048x1024"
