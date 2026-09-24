from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils
from litellm.llms.azure_ai.image_generation.cost_calculator import cost_calculator
from litellm.types.utils import ImageObject, ImageResponse, ImageUsage, ImageUsageInputTokensDetails
from litellm.utils import _invalidate_model_cost_lowercase_map

REFERENCE_PIXELS: Final = 2 * 1024 * 1024


@pytest.fixture(autouse=True)
def use_local_model_cost_map(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "model_cost", GetModelCostMap.load_local_model_cost_map())
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()
    yield
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()


def _edit_response(reference_pixels: int | None = REFERENCE_PIXELS, **kwargs: object) -> ImageResponse:
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


def test_get_model_info_surfaces_flux2_flex_pixel_rates() -> None:
    model_info = litellm.get_model_info(model="FLUX.2-flex", custom_llm_provider="azure_ai")
    catalog_info = litellm.model_cost["azure_ai/FLUX.2-flex"]

    assert model_info["input_cost_per_pixel"] == catalog_info["input_cost_per_pixel"]
    assert model_info["input_cost_per_reference_pixel"] == catalog_info["input_cost_per_reference_pixel"]


def test_flux2_flex_catalog_pixel_rates_match_reference_rates() -> None:
    catalog_info = litellm.model_cost["azure_ai/FLUX.2-flex"]

    assert catalog_info["input_cost_per_reference_pixel"] == catalog_info["input_cost_per_pixel"]
    assert catalog_info["input_cost_per_pixel"] * 1024 * 1024 == pytest.approx(0.05), (
        "Azure Retail Prices API, product 'Azure BFL Flux Models', meters 'Flex Megapixel' and "
        "'Flex Ref Megapixel' are $0.05 per MP where 1 MP = 1024x1024 pixels; confirmed against "
        "Azure Cost Management usage on 2026-09-22 (1024x1024 image metered as 1.0 MP)"
    )


def test_edit_cost_adds_reference_pixels_to_generated_pixels() -> None:
    catalog_rate: Final = litellm.model_cost["azure_ai/FLUX.2-flex"]["input_cost_per_pixel"]

    cost: Final = _edit_cost(_edit_response())

    assert cost == pytest.approx(catalog_rate * 1024 * 1024 + catalog_rate * REFERENCE_PIXELS)


def test_edit_cost_prefers_deployment_rates_over_catalog() -> None:
    cost: Final = _edit_cost(
        _edit_response(), model_info={"input_cost_per_pixel": 2e-07, "input_cost_per_reference_pixel": 3e-07}
    )

    assert cost == pytest.approx(2e-07 * 1024 * 1024 + 3e-07 * REFERENCE_PIXELS)


def test_edit_cost_honors_explicit_zero_reference_rate() -> None:
    cost: Final = _edit_cost(
        _edit_response(), model_info={"input_cost_per_pixel": 2e-07, "input_cost_per_reference_pixel": 0.0}
    )

    assert cost == pytest.approx(2e-07 * 1024 * 1024)


def test_edit_cost_honors_explicit_zero_generated_rate() -> None:
    cost: Final = _edit_cost(
        _edit_response(), model_info={"input_cost_per_pixel": 0.0, "input_cost_per_reference_pixel": 1e-07}
    )

    assert cost == pytest.approx(1e-07 * REFERENCE_PIXELS)


def test_per_image_rate_beats_per_pixel_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    cost: Final = _edit_cost(
        _edit_response(),
        model_info={
            "output_cost_per_image": 0.04,
            "input_cost_per_pixel": 1e-07,
            "input_cost_per_reference_pixel": 1.5e-08,
        },
    )

    assert cost == pytest.approx(0.04 + 1.5e-08 * REFERENCE_PIXELS)


def test_usage_short_circuits_pixel_billing() -> None:
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
            "input_cost_per_reference_pixel": 5e-08,
            "input_cost_per_token": 1e-05,
            "input_cost_per_image_token": 2e-05,
            "output_cost_per_image_token": 4e-05,
            "output_cost_per_token": 4e-05,
        },
    )

    assert cost == pytest.approx(50 * 1e-05 + 100 * 2e-05 + 1000 * 4e-05)


def test_edit_without_measurement_bills_generated_pixels_only() -> None:
    catalog_rate: Final = litellm.model_cost["azure_ai/FLUX.2-flex"]["input_cost_per_pixel"]

    cost: Final = _edit_cost(_edit_response(reference_pixels=None))

    assert cost == pytest.approx(catalog_rate * 1024 * 1024)


@pytest.mark.parametrize(
    ("size", "expected_generated"),
    [
        ("1024-x-1024", "computed"),
        ("auto", "reference-only"),
        ("garbage", "reference-only"),
    ],
    ids=["dashed-size", "auto-size-unparsed", "garbage-size-unparsed"],
)
def test_size_string_parsing(size: str, expected_generated: str) -> None:
    catalog_rate: Final = litellm.model_cost["azure_ai/FLUX.2-flex"]["input_cost_per_pixel"]
    expected: Final = (
        catalog_rate * 1024 * 1024 if expected_generated == "computed" else 0.0
    ) + catalog_rate * REFERENCE_PIXELS

    cost: Final = _edit_cost(_edit_response(), size=size)

    assert cost == pytest.approx(expected)


def test_optional_params_dimensions_beat_response_size() -> None:
    cost: Final = _edit_cost(
        _edit_response(reference_pixels=None),
        optional_params={"width": 2048, "height": 1024},
    )

    assert cost == pytest.approx(litellm.model_cost["azure_ai/FLUX.2-flex"]["input_cost_per_pixel"] * 2048 * 1024)


def test_unlisted_model_bills_deployment_rates() -> None:
    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="unlisted-flux-deployment",
        completion_response=_edit_response(),
        custom_llm_provider="azure_ai",
        size="1024x1024",
        call_type="image_edit",
        model_info={"input_cost_per_pixel": 1e-07, "input_cost_per_reference_pixel": 1e-07},
    )

    assert cost == pytest.approx(1e-07 * 1024 * 1024 + 1e-07 * REFERENCE_PIXELS)


def test_non_image_response_raises() -> None:
    with pytest.raises(ValueError, match="must be of type ImageResponse"):
        cost_calculator(model="FLUX.2-flex", image_response=object())
