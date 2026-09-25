from collections.abc import Mapping
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils
from litellm.llms.azure.azure import AzureChatCompletion
from litellm.llms.azure.image_generation import get_azure_image_generation_config
from litellm.llms.azure.image_generation.http_utils import azure_deployment_image_generation_json_body
from litellm.llms.azure_ai.image_generation.cost_calculator import cost_calculator
from litellm.llms.azure_ai.image_generation.flux_transformation import (
    AzureFoundryFluxImageGenerationConfig,
)
from litellm.types.utils import (
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
)
from litellm.utils import _invalidate_model_cost_lowercase_map, get_optional_params_image_gen


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
    assert catalog_info["input_cost_per_pixel"] * 1024 * 1024 == pytest.approx(0.05), (
        "Azure Retail Prices API, product 'Azure BFL Flux Models', meters 'Flex Megapixel' and "
        "'Flex Ref Megapixel' are $0.05 per MP where 1 MP = 1024x1024 pixels; confirmed against "
        "Azure Cost Management usage on 2026-09-22 (1024x1024 image metered as 1.0 MP)"
    )
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

    assert cost == pytest.approx(litellm.model_cost["azure_ai/FLUX.2-flex"]["input_cost_per_pixel"] * 2048 * 1024 * 2)


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
    ) == pytest.approx(litellm.model_cost["azure_ai/FLUX.2-flex"]["input_cost_per_pixel"] * 2048 * 1024 * 2)


def test_flux2_flex_cost_accepts_lowercase_model_spelling():
    response: Final = ImageResponse(data=[ImageObject(b64_json="aW1n"), ImageObject(b64_json="aW1n")])

    cost: Final = litellm.completion_cost(
        model="azure_ai/flux.2-flex",
        completion_response=response,
        optional_params={"width": 1536, "height": 1024, "num_images": 2},
        call_type="image_generation",
    )

    assert cost == pytest.approx(litellm.model_cost["azure_ai/FLUX.2-flex"]["input_cost_per_pixel"] * 1536 * 1024 * 2)


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


def _catalog_pixel_rate() -> float:
    return litellm.model_cost["azure_ai/FLUX.2-flex"]["input_cost_per_pixel"]


_ONE_MEGAPIXEL: Final = 1024 * 1024


def test_flux2_cost_bills_references_once_for_multi_image_edits() -> None:
    response: Final = ImageResponse(data=[ImageObject(b64_json="aW1n"), ImageObject(b64_json="aW1n")])
    response.set_reference_pixels(_ONE_MEGAPIXEL)

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="FLUX.2-flex",
        completion_response=response,
        custom_llm_provider="azure_ai",
        size="1024x1024",
        call_type="image_edit",
        n=2,
    )

    assert cost == pytest.approx(_catalog_pixel_rate() * (_ONE_MEGAPIXEL * 2 + _ONE_MEGAPIXEL))


@pytest.mark.parametrize(
    "dimensions",
    (
        {"width": True, "height": 1024},
        {"width": -2048, "height": 1024},
        {"width": 2048, "height": 0},
        {"width": 2048.0, "height": 1024},
    ),
)
def test_flux2_cost_rejects_bool_and_non_positive_dimensions_for_the_size_string(
    dimensions: Mapping[str, int | float | bool],
) -> None:
    response: Final = ImageResponse(data=[ImageObject(b64_json="aW1n")])

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="FLUX.2-flex",
        completion_response=response,
        custom_llm_provider="azure_ai",
        optional_params=dimensions,
        size="2048x1024",
        call_type="image_edit",
        n=1,
    )

    assert cost == pytest.approx(_catalog_pixel_rate() * 2048 * 1024)


@pytest.mark.parametrize("size", ("auto", "big", "1024", "1024x"))
def test_flux2_cost_skips_unparseable_size_strings(size: str) -> None:
    response: Final = ImageResponse(data=[ImageObject(b64_json="aW1n")])
    response.set_reference_pixels(_ONE_MEGAPIXEL)

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="FLUX.2-flex",
        completion_response=response,
        custom_llm_provider="azure_ai",
        size=size,
        call_type="image_edit",
    )

    assert cost == pytest.approx(_catalog_pixel_rate() * _ONE_MEGAPIXEL)


def test_flux2_cost_honors_an_explicit_zero_output_cost_per_image() -> None:
    response: Final = ImageResponse(data=[ImageObject(b64_json="aW1n")])
    response.set_reference_pixels(_ONE_MEGAPIXEL)

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="FLUX.2-flex",
        completion_response=response,
        custom_llm_provider="azure_ai",
        size="1024x1024",
        call_type="image_edit",
        model_info={"output_cost_per_image": 0.0, "input_cost_per_pixel": _catalog_pixel_rate()},
    )

    assert cost == pytest.approx(_catalog_pixel_rate() * _ONE_MEGAPIXEL)


def test_flux2_cost_ignores_negative_reference_pixels() -> None:
    response: Final = ImageResponse(data=[ImageObject(b64_json="aW1n")])
    response.set_reference_pixels(-2_097_152)

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="FLUX.2-flex",
        completion_response=response,
        custom_llm_provider="azure_ai",
        size="1024x1024",
        call_type="image_edit",
    )

    assert cost == pytest.approx(_catalog_pixel_rate() * _ONE_MEGAPIXEL)


def test_flux2_cost_bills_pixels_when_usage_carries_no_token_rates() -> None:
    response: Final = ImageResponse(
        data=[ImageObject(b64_json="aW1n")],
        usage=ImageUsage(
            input_tokens=100,
            input_tokens_details={"image_tokens": 50, "text_tokens": 50},
            output_tokens=50,
            total_tokens=150,
        ),
    )
    response.set_reference_pixels(_ONE_MEGAPIXEL)

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="FLUX.2-flex",
        completion_response=response,
        custom_llm_provider="azure_ai",
        size="1024x1024",
        call_type="image_edit",
    )

    assert cost == pytest.approx(_catalog_pixel_rate() * _ONE_MEGAPIXEL * 2)


def test_flux2_cost_reads_pricing_declared_in_litellm_params_kwargs() -> None:
    deployment_rate: Final = 1e-06
    response: Final = ImageResponse(data=[ImageObject(b64_json="aW1n")])
    response.set_reference_pixels(_ONE_MEGAPIXEL)
    logging_obj: Final = MagicMock()
    logging_obj.litellm_params = {"input_cost_per_pixel": deployment_rate}

    cost: Final = litellm.completion_cost(
        completion_response=response,
        model="azure_ai/FLUX.2-flex",
        custom_llm_provider="azure_ai",
        custom_pricing=True,
        litellm_logging_obj=logging_obj,
        size="1024x1024",
        call_type="image_edit",
    )

    assert cost == pytest.approx(deployment_rate * _ONE_MEGAPIXEL * 2)


def test_flux2_cost_overlays_litellm_params_kwargs_on_nested_model_info() -> None:
    nested_rate: Final = 1e-06
    kwargs_rate: Final = 2e-06
    response: Final = ImageResponse(data=[ImageObject(b64_json="aW1n")])
    response.set_reference_pixels(_ONE_MEGAPIXEL)
    logging_obj: Final = MagicMock()
    logging_obj.litellm_params = {
        "metadata": {"model_info": {"input_cost_per_pixel": nested_rate}},
        "input_cost_per_pixel": kwargs_rate,
    }

    cost: Final = litellm.completion_cost(
        completion_response=response,
        model="azure_ai/FLUX.2-flex",
        custom_llm_provider="azure_ai",
        custom_pricing=True,
        litellm_logging_obj=logging_obj,
        size="1024x1024",
        call_type="image_edit",
    )

    assert cost == pytest.approx(kwargs_rate * _ONE_MEGAPIXEL * 2)


def _edit_response(reference_pixels: int | None = _ONE_MEGAPIXEL * 2, **kwargs: object) -> ImageResponse:
    response: Final = ImageResponse(data=[ImageObject(b64_json="aW1n")], size="1024x1024", **kwargs)
    if reference_pixels is not None:
        response.set_reference_pixels(reference_pixels)
    return response


def _edit_cost(response: ImageResponse, size: str = "1024x1024", **kwargs: object) -> float:
    return CostCalculatorUtils.route_image_generation_cost_calculator(
        model="FLUX.2-flex",
        completion_response=response,
        custom_llm_provider="azure_ai",
        size=size,
        call_type="image_edit",
        **kwargs,
    )


def test_get_model_info_surfaces_flux2_flex_pixel_rate() -> None:
    model_info = litellm.get_model_info(model="FLUX.2-flex", custom_llm_provider="azure_ai")

    assert model_info["input_cost_per_pixel"] == _catalog_pixel_rate()


def test_flux2_cost_adds_reference_pixels_to_generated_pixels() -> None:
    cost: Final = _edit_cost(_edit_response())

    assert cost == pytest.approx(_catalog_pixel_rate() * _ONE_MEGAPIXEL * 3)


def test_flux2_cost_prefers_deployment_rates_over_catalog_for_edits() -> None:
    cost: Final = _edit_cost(_edit_response(), model_info={"input_cost_per_pixel": 2e-07})

    assert cost == pytest.approx(2e-07 * _ONE_MEGAPIXEL * 3)


def test_flux2_cost_honors_explicit_zero_pixel_rate() -> None:
    cost: Final = _edit_cost(_edit_response(), model_info={"input_cost_per_pixel": 0.0})

    assert cost == pytest.approx(0.0)


def test_flux2_cost_per_image_rate_beats_per_pixel_rate() -> None:
    cost: Final = _edit_cost(
        _edit_response(),
        model_info={"output_cost_per_image": 0.04, "input_cost_per_pixel": 1e-07},
    )

    assert cost == pytest.approx(0.04 + 1e-07 * _ONE_MEGAPIXEL * 2)


def test_flux2_cost_bills_token_rates_when_usage_carries_them() -> None:
    response: Final = _edit_response(
        usage=ImageUsage(
            input_tokens=150,
            input_tokens_details=ImageUsageInputTokensDetails(image_tokens=100, text_tokens=50),
            output_tokens=1000,
            total_tokens=1150,
        )
    )

    cost: Final = _edit_cost(
        response,
        model_info={
            "input_cost_per_pixel": 5e-08,
            "input_cost_per_token": 1e-05,
            "input_cost_per_image_token": 2e-05,
            "output_cost_per_image_token": 4e-05,
            "output_cost_per_token": 4e-05,
        },
    )

    assert cost == pytest.approx(50 * 1e-05 + 100 * 2e-05 + 1000 * 4e-05)


def test_flux2_cost_parses_dashed_size_string() -> None:
    cost: Final = _edit_cost(_edit_response(), size="1024-x-1024")

    assert cost == pytest.approx(_catalog_pixel_rate() * _ONE_MEGAPIXEL * 3)


def test_flux2_cost_optional_params_dimensions_beat_response_size() -> None:
    cost: Final = _edit_cost(
        _edit_response(reference_pixels=None),
        optional_params={"width": 2048, "height": 1024},
    )

    assert cost == pytest.approx(_catalog_pixel_rate() * 2048 * 1024)


def test_flux2_cost_rejects_non_image_response() -> None:
    with pytest.raises(TypeError, match="must be of type ImageResponse"):
        cost_calculator(model="FLUX.2-flex", image_response=object())


def test_flux2_pro_edit_bills_references_on_the_pro_ref_meter() -> None:
    pro_pixel_rate: Final = litellm.model_cost["azure_ai/flux.2-pro"]["input_cost_per_pixel"]
    assert pro_pixel_rate * _ONE_MEGAPIXEL == pytest.approx(0.015), (
        "Azure Retail Prices API, product 'Azure BFL Flux Models', meter 'Flux 2 Ref MP Megapixel' is $0.015 "
        "per MP, checked 2026-09-24"
    )

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="flux.2-pro",
        completion_response=_edit_response(),
        custom_llm_provider="azure_ai",
        size="1024x1024",
        call_type="image_edit",
    )

    assert cost == pytest.approx(litellm.model_cost["azure_ai/flux.2-pro"]["output_cost_per_image"] + 0.015 * 2)
