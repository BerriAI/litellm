import json
from collections.abc import Callable, Mapping
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
import respx

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


FOUNDRY_BASE: Final = "https://flux-billing.services.ai.azure.com"


def _flux2_route(provider_path: str) -> str:
    return rf"https://flux-billing\.services\.ai\.azure\.com/providers/blackforestlabs/v1/{provider_path}\?.*"


def _metered_body(output_mp: float, input_mp: float) -> Mapping[str, object]:
    return {
        "created": 1,
        "data": [{"b64_json": "aW1n"}],
        "request_meta": {"cost": 999.0, "input_mp": input_mp, "output_mp": output_mp, "total_pixels": 1},
    }


def _price(model: str, key: str) -> float:
    price: Final = litellm.model_cost[model][key]
    assert isinstance(price, float) and price > 0
    return price


# request_meta pairs Azure returned for FLUX.2 [pro] on 2026-10-27, whose credit charges matched whole megapixels
# rounded up: one 512x512 reference reported input_mp 0.25 and two of them reported 2.0
@pytest.mark.parametrize(
    ("output_mp", "input_mp", "additional_output_megapixels", "input_megapixels"),
    [
        (0.25, 0.0, 0, 0),
        (1.5, 0.0, 1, 0),
        (2.07, 0.0, 2, 0),
        (1.0, 0.25, 0, 1),
        (1.0, 1.5, 0, 2),
        (1.0, 2.0, 0, 2),
    ],
)
@respx.mock
def test_flux2_pro_generation_bills_the_reported_megapixels_rounded_up(
    output_mp: float, input_mp: float, additional_output_megapixels: int, input_megapixels: int
) -> None:
    respx.post(url__regex=_flux2_route("flux-2-pro")).mock(
        return_value=httpx.Response(200, json=_metered_body(output_mp, input_mp))
    )

    response: Final = litellm.image_generation(
        model="azure_ai/flux.2-pro", prompt="A red fox", api_base=FOUNDRY_BASE, api_key="test-key"
    )

    assert response._hidden_params["response_cost"] == pytest.approx(
        _price("azure_ai/flux.2-pro", "output_cost_per_first_megapixel")
        + additional_output_megapixels * _price("azure_ai/flux.2-pro", "output_cost_per_megapixel")
        + input_megapixels * _price("azure_ai/flux.2-pro", "input_cost_per_megapixel")
    )


@pytest.mark.asyncio
async def test_async_flux2_pro_generation_bills_the_reported_megapixels(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post(url__regex=_flux2_route("flux-2-pro")).mock(
        return_value=httpx.Response(200, json=_metered_body(1.5, 0.25))
    )

    response: Final = await litellm.aimage_generation(
        model="azure_ai/flux.2-pro", prompt="A red fox", api_base=FOUNDRY_BASE, api_key="test-key"
    )

    assert response._hidden_params["response_cost"] == pytest.approx(
        _price("azure_ai/flux.2-pro", "output_cost_per_first_megapixel")
        + _price("azure_ai/flux.2-pro", "output_cost_per_megapixel")
        + _price("azure_ai/flux.2-pro", "input_cost_per_megapixel")
    )


@respx.mock
def test_flux2_flex_generation_bills_every_reported_megapixel_at_one_rate() -> None:
    respx.post(url__regex=_flux2_route("flux-2-flex")).mock(
        return_value=httpx.Response(200, json=_metered_body(1.5, 0.25))
    )

    response: Final = litellm.image_generation(
        model="azure_ai/FLUX.2-flex", prompt="A red fox", api_base=FOUNDRY_BASE, api_key="test-key"
    )

    assert response._hidden_params["response_cost"] == pytest.approx(
        2 * _price("azure_ai/FLUX.2-flex", "output_cost_per_megapixel")
        + _price("azure_ai/FLUX.2-flex", "input_cost_per_megapixel")
    )


@pytest.mark.parametrize(
    ("deployment_prices", "expected_cost"),
    [
        (
            {},
            lambda price: price("output_cost_per_first_megapixel")
            + price("output_cost_per_megapixel")
            + price("input_cost_per_megapixel"),
        ),
        ({"output_cost_per_image": 0.05}, lambda _: 0.05),
        (
            {"output_cost_per_megapixel": 0.02},
            lambda price: price("output_cost_per_first_megapixel") + 0.02 + price("input_cost_per_megapixel"),
        ),
        (
            {"output_cost_per_first_megapixel": 0.04, "output_cost_per_megapixel": 0.02, "input_cost_per_megapixel": 0.01},
            lambda _: 0.04 + 0.02 + 0.01,
        ),
    ],
)
@respx.mock
def test_flux2_router_deployment_prices_layer_over_the_catalog_megapixel_prices(
    deployment_prices: Mapping[str, float], expected_cost: Callable[[Callable[[str], float]], float]
) -> None:
    respx.post(url__regex=_flux2_route("flux-2-pro")).mock(
        return_value=httpx.Response(200, json=_metered_body(1.5, 0.25))
    )
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "flux-2-pro",
                "litellm_params": {
                    "model": "azure_ai/flux.2-pro",
                    "api_base": FOUNDRY_BASE,
                    "api_key": "test-key",
                    **deployment_prices,
                },
            }
        ]
    )

    response: Final = router.image_generation(model="flux-2-pro", prompt="A red fox")

    assert response._hidden_params["response_cost"] == pytest.approx(
        expected_cost(lambda key: _price("azure_ai/flux.2-pro", key))
    )


@pytest.mark.parametrize(
    "request_meta",
    [
        None,
        {"input_mp": 0.0, "output_mp": 0.0},
        {"input_mp": -1.0, "output_mp": 1.0},
        {"input_mp": 0.0, "output_mp": float("inf")},
        {"input_mp": 0.0},
    ],
)
@respx.mock
def test_flux2_pro_generation_without_usable_megapixels_keeps_the_per_image_price(
    request_meta: Mapping[str, float] | None,
) -> None:
    body: Final = {"created": 1, "data": [{"b64_json": "aW1n"}], "request_meta": request_meta}
    respx.post(url__regex=_flux2_route("flux-2-pro")).mock(
        return_value=httpx.Response(200, content=json.dumps(body).encode())
    )

    response: Final = litellm.image_generation(
        model="azure_ai/flux.2-pro", prompt="A red fox", api_base=FOUNDRY_BASE, api_key="test-key"
    )

    assert response._hidden_params["response_cost"] == pytest.approx(
        _price("azure_ai/flux.2-pro", "output_cost_per_image")
    )


@respx.mock
def test_flux2_pro_generation_bills_no_references_when_input_megapixels_are_null() -> None:
    body: Final = {"created": 1, "data": [{"b64_json": "aW1n"}], "request_meta": {"input_mp": None, "output_mp": 1.5}}
    respx.post(url__regex=_flux2_route("flux-2-pro")).mock(return_value=httpx.Response(200, json=body))

    response: Final = litellm.image_generation(
        model="azure_ai/flux.2-pro", prompt="A red fox", api_base=FOUNDRY_BASE, api_key="test-key"
    )

    assert response._hidden_params["response_cost"] == pytest.approx(
        _price("azure_ai/flux.2-pro", "output_cost_per_first_megapixel")
        + _price("azure_ai/flux.2-pro", "output_cost_per_megapixel")
    )
