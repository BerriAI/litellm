import base64
import struct
from collections.abc import Callable, Mapping
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
from litellm.llms.azure_ai.image_generation.cost_calculator import cost_calculator as azure_ai_image_cost_calculator
from litellm.llms.azure_ai.image_generation.flux_transformation import (
    AzureFoundryFluxImageGenerationConfig,
)
from litellm.types.utils import ImageObject, ImageResponse, ImageUsage, ImageUsageInputTokensDetails
from litellm.utils import _invalidate_model_cost_lowercase_map, get_optional_params_image_gen


@pytest.fixture(autouse=True)
def use_local_model_cost_map(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", GetModelCostMap.load_local_model_cost_map())
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()
    yield
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()


def _flex_pixel_rate() -> float:
    return litellm.model_cost["azure_ai/FLUX.2-flex"]["input_cost_per_pixel"]


def _flex_edit_cost(response: ImageResponse, model_info: dict | None = None) -> float:
    return CostCalculatorUtils.route_image_generation_cost_calculator(
        model="FLUX.2-flex",
        completion_response=response,
        custom_llm_provider="azure_ai",
        size="1024x1024",
        call_type="image_edit",
        model_info=model_info,
    )


def _edit_response(reference_image_pixels: object) -> ImageResponse:
    return ImageResponse(
        data=[ImageObject(b64_json="aW1n")],
        size="1024x1024",
        hidden_params={"reference_image_pixels": reference_image_pixels},
    )


def _pro_megapixel_rates() -> tuple[float, float]:
    row: Final = litellm.model_cost["azure_ai/flux.2-pro"]
    return row["output_cost_per_image"], row["input_cost_per_pixel"] * 1024 * 1024


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
    assert catalog_info["supported_modalities"] == ["text", "image"]
    assert catalog_info["supported_output_modalities"] == ["image"]


@pytest.mark.parametrize("model", ("flux.2-pro", "FLUX.2-flex"))
def test_flux2_model_info_lists_the_edit_endpoint(model: str):
    model_info: Final = litellm.get_model_info(model=model, custom_llm_provider="azure_ai")

    assert "/v1/images/edits" in model_info["supported_endpoints"]


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

    assert cost == pytest.approx(_flex_pixel_rate() * 2048 * 1024 * 2)


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
    ) == pytest.approx(_flex_pixel_rate() * 2048 * 1024 * 2)


def test_flux2_flex_cost_accepts_lowercase_model_spelling():
    response: Final = ImageResponse(data=[ImageObject(b64_json="aW1n"), ImageObject(b64_json="aW1n")])

    cost: Final = litellm.completion_cost(
        model="azure_ai/flux.2-flex",
        completion_response=response,
        optional_params={"width": 2048, "height": 1024, "num_images": 2},
        call_type="image_generation",
    )

    assert cost == pytest.approx(_flex_pixel_rate() * 2048 * 1024 * 2)


def test_flux2_flex_generation_rounds_each_image_up_to_whole_megapixels():
    response: Final = ImageResponse(data=[ImageObject(b64_json="aW1n"), ImageObject(b64_json="aW1n")])

    cost: Final = litellm.completion_cost(
        model="azure_ai/FLUX.2-flex",
        completion_response=response,
        optional_params={"width": 1536, "height": 1024, "num_images": 2},
        call_type="image_generation",
    )

    assert cost == pytest.approx(_flex_pixel_rate() * 2 * 1024 * 1024 * 2)


@pytest.mark.parametrize(("size", "megapixels"), (("256x256", 1), ("1024x1280", 2), ("2048x2048", 4)))
def test_flux2_pro_generation_bills_the_first_megapixel_then_each_additional_one(size: str, megapixels: int):
    first, additional = _pro_megapixel_rates()

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="flux.2-pro",
        completion_response=ImageResponse(data=[ImageObject(b64_json="aW1n")]),
        custom_llm_provider="azure_ai",
        size=size,
        call_type="image_generation",
    )

    assert cost == pytest.approx(first + additional * (megapixels - 1))


def _catalog_image_cost(model: str, megapixels: int) -> float:
    row: Final = litellm.model_cost[f"azure_ai/{model}"]
    megapixel_rate: Final = row["input_cost_per_pixel"] * 1024 * 1024
    return row.get("output_cost_per_image", megapixel_rate) + megapixel_rate * (megapixels - 1)


@pytest.mark.parametrize("model", ("flux.2-pro", "FLUX.2-flex"))
@pytest.mark.parametrize(("size", "megapixels"), (("1024x1024", 1), ("2048x2048", 4)))
@pytest.mark.parametrize(
    ("deployment_prices", "expected"),
    (
        pytest.param(None, None, id="catalog-prices"),
        pytest.param({"output_cost_per_image": 0.07}, lambda megapixels: 0.07, id="deployment-flat-price"),
        pytest.param(
            {"output_cost_per_image": 0.07, "input_cost_per_pixel": 1e-07},
            lambda megapixels: 0.07,
            id="deployment-flat-price-wins-over-its-pixel-rate",
        ),
        pytest.param(
            {"input_cost_per_pixel": 1e-07},
            lambda megapixels: 1e-07 * 1024 * 1024 * megapixels,
            id="deployment-pixel-rate-prices-every-megapixel",
        ),
    ),
)
def test_flux2_generation_bills_each_price_source_as_its_owner_set_it(
    model: str,
    size: str,
    megapixels: int,
    deployment_prices: Mapping[str, float] | None,
    expected: Callable[[int], float] | None,
):
    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model=model,
        completion_response=ImageResponse(data=[ImageObject(b64_json="aW1n")]),
        custom_llm_provider="azure_ai",
        size=size,
        call_type="image_generation",
        model_info=None if deployment_prices is None else dict(deployment_prices),
    )

    assert cost == pytest.approx(_catalog_image_cost(model, megapixels) if expected is None else expected(megapixels))


@pytest.mark.parametrize("model", ("flux.2-pro", "FLUX.2-flex"))
@pytest.mark.parametrize(
    ("deployment_prices", "reference_megapixel_price"),
    (
        pytest.param(None, None, id="catalog-megapixel-rate"),
        pytest.param({"output_cost_per_image": 0.07}, 0.0, id="deployment-flat-price-covers-references"),
        pytest.param(
            {"output_cost_per_image": 0.07, "input_cost_per_pixel": 1e-07},
            0.0,
            id="deployment-flat-price-covers-references-despite-its-pixel-rate",
        ),
        pytest.param(
            {"input_cost_per_pixel": 1e-07}, 1e-07 * 1024 * 1024, id="deployment-pixel-rate-prices-references"
        ),
    ),
)
def test_flux2_edit_prices_references_from_the_source_that_priced_the_image(
    model: str, deployment_prices: Mapping[str, float] | None, reference_megapixel_price: float | None
):
    def edit_cost(reference_pixels: tuple[int, ...]) -> float:
        return CostCalculatorUtils.route_image_generation_cost_calculator(
            model=model,
            completion_response=_edit_response(reference_pixels),
            custom_llm_provider="azure_ai",
            size="1024x1024",
            call_type="image_edit",
            model_info=None if deployment_prices is None else dict(deployment_prices),
        )

    expected_per_megapixel: Final = (
        litellm.model_cost[f"azure_ai/{model}"]["input_cost_per_pixel"] * 1024 * 1024
        if reference_megapixel_price is None
        else reference_megapixel_price
    )

    assert edit_cost((1024 * 1280,)) - edit_cost(()) == pytest.approx(expected_per_megapixel * 2)


def _png_b64(width: int, height: int) -> str:
    png: Final = (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x02\x00\x00\x00"
    )
    return base64.b64encode(png).decode()


@pytest.mark.parametrize(
    ("returned_images", "requested_size", "expected_megapixels"),
    (
        pytest.param(((1024, 1280),), "1024x1024", (2,), id="returned-size-wins-over-requested"),
        pytest.param(((512, 512), (2048, 1024)), "1024x1024", (1, 2), id="each-returned-image-measured"),
        pytest.param((None,), "1024x1280", (2,), id="unmeasurable-image-falls-back-to-requested"),
    ),
)
def test_flux2_pro_bills_each_generated_image_by_its_returned_size(
    returned_images: tuple[tuple[int, int] | None, ...], requested_size: str, expected_megapixels: tuple[int, ...]
):
    first, additional = _pro_megapixel_rates()
    response: Final = ImageResponse(
        data=[ImageObject(b64_json="aW1n" if image is None else _png_b64(*image)) for image in returned_images]
    )

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="flux.2-pro",
        completion_response=response,
        custom_llm_provider="azure_ai",
        size=requested_size,
        call_type="image_generation",
    )

    assert cost == pytest.approx(sum(first + additional * (megapixels - 1) for megapixels in expected_megapixels))


@pytest.mark.parametrize("size", ("auto", "large", "0x1024", "1024x"))
def test_flux2_pro_bills_one_megapixel_for_a_size_it_cannot_measure(size: str):
    first, _additional = _pro_megapixel_rates()

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="flux.2-pro",
        completion_response=ImageResponse(data=[ImageObject(b64_json="aW1n")]),
        custom_llm_provider="azure_ai",
        size=size,
        call_type="image_generation",
    )

    assert cost == pytest.approx(first)


def test_flux2_pro_bills_the_requested_size_when_the_returned_image_is_not_valid_base64():
    first, additional = _pro_megapixel_rates()

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="flux.2-pro",
        completion_response=ImageResponse(data=[ImageObject(b64_json="abc")]),
        custom_llm_provider="azure_ai",
        size="2048x1024",
        call_type="image_generation",
    )

    assert cost == pytest.approx(first + additional)


@pytest.mark.parametrize(("n", "billed_images"), ((2, 2), (None, 0)))
def test_flux2_pro_bills_the_requested_image_count_when_the_response_lists_none(n: int | None, billed_images: int):
    first, _additional = _pro_megapixel_rates()

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="flux.2-pro",
        completion_response=ImageResponse(data=[]),
        custom_llm_provider="azure_ai",
        size="1024x1024",
        n=n,
        call_type="image_generation",
    )

    assert cost == pytest.approx(first * billed_images)


def test_flux2_pro_reads_an_uppercase_size():
    first, additional = _pro_megapixel_rates()

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="flux.2-pro",
        completion_response=ImageResponse(data=[ImageObject(b64_json="aW1n")]),
        custom_llm_provider="azure_ai",
        size="2048X1024",
        call_type="image_generation",
    )

    assert cost == pytest.approx(first + additional)


def test_flux2_edit_parses_string_prices_registered_in_the_cost_map(monkeypatch: pytest.MonkeyPatch):
    row: Final = litellm.model_cost["azure_ai/FLUX.2-flex"]
    monkeypatch.setitem(row, "input_cost_per_pixel", "2e-07")
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()

    assert _flex_edit_cost(_edit_response((1024 * 1024,))) == pytest.approx(2e-07 * 1024 * 1024 * 2)


@pytest.fixture
def distinct_pro_megapixel_prices(monkeypatch: pytest.MonkeyPatch) -> tuple[float, float]:
    row: Final = litellm.model_cost["azure_ai/flux.2-pro"]
    monkeypatch.setitem(row, "output_cost_per_image", 0.05)
    monkeypatch.setitem(row, "input_cost_per_pixel", 2e-08)
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()
    return 0.05, 2e-08 * 1024 * 1024


# Billable megapixels as Azure's request_meta reported them for FLUX.2-pro edits on 2026-09-25
@pytest.mark.parametrize(
    ("size", "reference_pixels", "output_megapixels", "reference_megapixels"),
    (
        pytest.param("1024x1024", (1024 * 1024,), 1, 1, id="one-whole-megapixel-reference"),
        pytest.param("1024x1024", (1024 * 1280,), 1, 2, id="fractional-reference-rounds-up"),
        pytest.param("1024x1280", (1024 * 1280,), 2, 2, id="fractional-output-rounds-up"),
        pytest.param("1024x1024", (4032 * 3024,), 1, 4, id="lone-reference-caps-at-four-megapixels"),
        pytest.param("1024x1024", (1024 * 1280,) * 2, 1, 2, id="each-of-several-references-is-one-megapixel"),
        pytest.param("1024x1024", (1024 * 1024, 4032 * 3024), 1, 2, id="large-reference-among-several"),
        pytest.param("1024x1024", (1024 * 1280,) * 3, 1, 3, id="three-references"),
    ),
)
def test_flux2_pro_edit_bills_the_megapixels_azure_meters(
    size: str,
    reference_pixels: tuple[int, ...],
    output_megapixels: int,
    reference_megapixels: int,
    distinct_pro_megapixel_prices: tuple[float, float],
):
    first, megapixel = distinct_pro_megapixel_prices

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="flux.2-pro",
        completion_response=_edit_response(reference_pixels),
        custom_llm_provider="azure_ai",
        size=size,
        call_type="image_edit",
    )

    assert cost == pytest.approx(first + megapixel * (output_megapixels - 1 + reference_megapixels))


def test_flux2_flex_edit_bills_reference_pixels_on_top_of_generated_pixels():
    generated_only: Final = _flex_edit_cost(ImageResponse(data=[ImageObject(b64_json="aW1n")], size="1024x1024"))

    assert generated_only == pytest.approx(_flex_pixel_rate() * 1024 * 1024)
    assert _flex_edit_cost(_edit_response((1024 * 1024,))) - generated_only == pytest.approx(
        _flex_pixel_rate() * 1024 * 1024
    )
    assert _flex_edit_cost(_edit_response((3 * 1024 * 1024,))) - generated_only == pytest.approx(
        _flex_pixel_rate() * 3 * 1024 * 1024
    )


def test_flux2_flex_edit_reference_cost_does_not_scale_with_image_count():
    two_outputs: Final = ImageResponse(
        data=[ImageObject(b64_json="aW1n"), ImageObject(b64_json="aW1n")],
        size="1024x1024",
        hidden_params={"reference_image_pixels": (2048 * 1024,)},
    )

    assert _flex_edit_cost(two_outputs) == pytest.approx(
        _flex_pixel_rate() * 1024 * 1024 * 2 + _flex_pixel_rate() * 2048 * 1024
    )


@pytest.mark.parametrize(
    "reference_image_pixels",
    ((), (0,), (-1,), (True,), None, (1048576.0,), ("1048576",), 1048576, (1048576, "1048576"), (1048576, 0)),
)
def test_flux2_flex_edit_bills_only_positive_integer_reference_counts(reference_image_pixels: object):
    generated_only: Final = _flex_edit_cost(ImageResponse(data=[ImageObject(b64_json="aW1n")], size="1024x1024"))

    assert _flex_edit_cost(_edit_response(reference_image_pixels)) == generated_only


def test_flux2_flex_edit_ignores_reference_pixels_when_provider_reports_token_usage():
    response: Final = ImageResponse(
        data=[ImageObject(b64_json="aW1n")],
        size="1024x1024",
        hidden_params={"reference_image_pixels": (1024 * 1024,)},
        usage=ImageUsage(
            input_tokens=150,
            input_tokens_details=ImageUsageInputTokensDetails(image_tokens=100, text_tokens=50),
            output_tokens=1000,
            total_tokens=1150,
        ),
    )
    token_rates: Final = {
        "input_cost_per_token": 1e-05,
        "input_cost_per_image_token": 2e-05,
        "output_cost_per_image_token": 4e-05,
        "output_cost_per_token": 4e-05,
    }

    assert _flex_edit_cost(response, model_info=token_rates) == pytest.approx(50 * 1e-05 + 100 * 2e-05 + 1000 * 4e-05)


def test_flux2_flex_edit_prices_output_from_the_request_size_not_the_reference():
    large_reference_small_output: Final = ImageResponse(
        data=[ImageObject(b64_json="aW1n")],
        size="1024x1024",
        hidden_params={"reference_image_pixels": (2048 * 2048,)},
    )

    assert _flex_edit_cost(large_reference_small_output) == pytest.approx(
        _flex_pixel_rate() * 1024 * 1024 + _flex_pixel_rate() * 2048 * 2048
    )


def test_flux2_flex_edit_with_a_free_deployment_pixel_rate_bills_nothing():
    cost: Final = _flex_edit_cost(_edit_response((1024 * 1024,)), model_info={"input_cost_per_pixel": 0.0})

    assert cost == 0.0


def test_custom_named_flux2_deployment_bills_its_own_megapixel_rate_for_output_and_references() -> None:
    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="my-flux2-prod",
        completion_response=_edit_response((1024 * 1280,)),
        custom_llm_provider="azure_ai",
        size="1024x1280",
        call_type="image_edit",
        model_info={"input_cost_per_pixel": 1e-07},
    )

    assert cost == pytest.approx(1e-07 * 1024 * 1024 * 2 + 1e-07 * 1024 * 1024 * 2)


def test_custom_named_flux2_deployment_without_an_image_price_bills_nothing() -> None:
    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="my-flux2-prod",
        completion_response=_edit_response((1024 * 1280,)),
        custom_llm_provider="azure_ai",
        size="1024x1280",
        call_type="image_edit",
        model_info={"input_cost_per_second": 1.0},
    )

    assert cost == 0.0


def test_azure_ai_image_cost_calculator_rejects_a_response_that_is_not_an_image_response() -> None:
    with pytest.raises(ValueError, match="must be of type ImageResponse"):
        azure_ai_image_cost_calculator(model="flux.2-pro", image_response={"data": []})


@pytest.mark.parametrize("model", ("FLUX-1.1-pro", "FLUX.1-Kontext-pro"))
def test_flat_priced_flux_edit_ignores_reference_pixels(model: str):
    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model=model,
        completion_response=_edit_response((4 * 1024 * 1024,)),
        custom_llm_provider="azure_ai",
        size="1024x1024",
        call_type="image_edit",
    )

    assert cost == pytest.approx(litellm.model_cost[f"azure_ai/{model}"]["output_cost_per_image"])


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


def test_unlisted_azure_ai_deployment_without_a_generated_image_price_bills_nothing() -> None:
    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="unlisted-flux-deployment",
        completion_response=ImageResponse(data=[ImageObject(b64_json="aW1n")]),
        custom_llm_provider="azure_ai",
        size="1024x1024",
        call_type="image_generation",
        model_info={"input_cost_per_second": 1.0},
    )

    assert cost == 0.0


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
