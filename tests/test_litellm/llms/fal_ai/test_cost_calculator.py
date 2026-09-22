from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils
from litellm.llms.fal_ai.cost_calculator import cost_calculator
from litellm.types.utils import ImageObject, ImageResponse

@pytest.fixture(autouse=True)
def _use_local_model_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


def _image_response(num_images: int = 1) -> ImageResponse:
    return ImageResponse(data=[ImageObject(url="https://example.com/img.png") for _ in range(num_images)])


def _image_response_with_dimensions(dimensions: tuple[tuple[int, int], ...]) -> ImageResponse:
    return ImageResponse(
        data=[
            ImageObject(
                url=f"https://example.com/img-{index}.png",
                provider_specific_fields={"width": width, "height": height},
            )
            for index, (width, height) in enumerate(dimensions)
        ]
    )


GPT_IMAGE_25_MODELS = (
    "openai/gpt-image-2.5/flare/text-to-image",
    "openai/gpt-image-2.5/flare/edit",
    "openai/gpt-image-2.5/sunburst/text-to-image",
    "openai/gpt-image-2.5/sunburst/edit",
)


@pytest.mark.parametrize("model", GPT_IMAGE_25_MODELS)
def test_gpt_image_25_default_request_matches_high_1024x768_keyed_row(model):
    default_cost = cost_calculator(model=f"fal_ai/{model}", image_response=_image_response(), optional_params={})
    keyed_cost = litellm.model_cost[f"fal_ai/high/1024-x-768/{model}"]["output_cost_per_image"]
    assert default_cost == keyed_cost > 0


@pytest.mark.parametrize("model", GPT_IMAGE_25_MODELS)
def test_gpt_image_25_quality_and_size_pick_keyed_row(model):
    cost = cost_calculator(
        model=f"fal_ai/{model}",
        image_response=_image_response(num_images=2),
        optional_params={"quality": "max", "image_size": {"width": 3840, "height": 2160}},
    )
    assert cost == 2 * litellm.model_cost[f"fal_ai/max/3840-x-2160/{model}"]["output_cost_per_image"] > 0


def test_gpt_image_25_edit_auto_size_still_honors_quality():
    model = "fal_ai/openai/gpt-image-2.5/flare/edit"
    low = cost_calculator(
        model=model, image_response=_image_response(), optional_params={"quality": "low", "image_size": "auto"}
    )
    high = cost_calculator(
        model=model, image_response=_image_response(), optional_params={"quality": "high", "image_size": "auto"}
    )
    assert 0 < low < high


def test_gpt_image_response_dimensions_override_request_size():
    model = "fal_ai/openai/gpt-image-2.5/flare/text-to-image"
    cost = cost_calculator(
        model=model,
        image_response=_image_response_with_dimensions(((1024, 1536),)),
        optional_params={"quality": "low", "image_size": {"width": 1024, "height": 768}},
    )
    expected = litellm.model_cost[f"fal_ai/low/1024-x-1536/{model.removeprefix('fal_ai/')}"]["output_cost_per_image"]
    assert cost == expected


def test_gpt_image_response_dimensions_use_nearest_keyed_row_when_unpriced():
    model = "fal_ai/openai/gpt-image-2.5/flare/text-to-image"
    cost = cost_calculator(
        model=model,
        image_response=_image_response_with_dimensions(((777, 888),)),
        optional_params={"quality": "low", "image_size": {"width": 1024, "height": 1536}},
    )
    expected = litellm.model_cost[f"fal_ai/low/1024-x-768/{model.removeprefix('fal_ai/')}"]["output_cost_per_image"]
    assert cost == expected


def test_gpt_image_25_noncanonical_response_uses_nearest_keyed_row():
    model: Final = "fal_ai/openai/gpt-image-2.5/flare/text-to-image"
    cost: Final = cost_calculator(
        model=model,
        image_response=_image_response_with_dimensions(((1536, 1024),)),
        optional_params={"quality": "low", "image_size": {"width": 1536, "height": 1024}},
    )
    expected: Final = litellm.model_cost[
        "fal_ai/low/1024-x-1536/openai/gpt-image-2.5/flare/text-to-image"
    ]["output_cost_per_image"]
    assert cost == expected


def test_gpt_image_25_quality_tiers_are_monotonic():
    costs = tuple(
        cost_calculator(
            model="fal_ai/openai/gpt-image-2.5/sunburst/text-to-image",
            image_response=_image_response(),
            optional_params={"quality": quality, "image_size": "square_hd"},
        )
        for quality in ("low", "medium", "high", "xhigh", "max")
    )
    assert costs == tuple(sorted(costs)) and len(set(costs)) == len(costs)


def test_flux_dev_cost_is_nonzero_and_distinct_from_schnell():
    dev = cost_calculator(
        model="fal_ai/fal-ai/flux/dev", image_response=_image_response(num_images=3), optional_params={}
    )
    schnell = cost_calculator(
        model="fal_ai/fal-ai/flux/schnell", image_response=_image_response(num_images=3), optional_params={}
    )
    assert dev > schnell > 0
    assert dev == 3 * litellm.model_cost["fal_ai/fal-ai/flux/dev"]["output_cost_per_image"]


def test_flux_dev_cost_uses_response_megapixels_per_image():
    model = "fal_ai/fal-ai/flux/dev"
    cost = cost_calculator(
        model=model,
        image_response=_image_response_with_dimensions(((1024, 1024), (1920, 1080), (512, 512))),
        optional_params={},
    )
    output_cost_per_pixel = litellm.model_cost[model]["output_cost_per_pixel"]
    assert cost == pytest.approx(output_cost_per_pixel * 1_048_576 * (1 + 2 + 1))


@pytest.mark.parametrize(
    "dimensions",
    (
        ((True, 1024),),
        ((1024, 0),),
        ((-1, 1024),),
    ),
)
def test_flux_dev_invalid_response_dimensions_use_flat_price(dimensions):
    model = "fal_ai/fal-ai/flux/dev"
    cost = cost_calculator(
        model=model,
        image_response=_image_response_with_dimensions(dimensions),
        optional_params={},
    )
    assert cost == litellm.model_cost[model]["output_cost_per_image"] * len(dimensions)


def test_unknown_fal_model_raises_when_flat_pricing_is_needed():
    with pytest.raises(Exception, match="isn't mapped yet"):
        cost_calculator(
            model="fal_ai/fal-ai/unknown-model",
            image_response=_image_response(),
            optional_params={},
        )


def test_image_edit_call_type_routes_to_fal_keyed_pricing():
    model = "openai/gpt-image-2.5/flare/edit"
    cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model=model,
        completion_response=_image_response(),
        custom_llm_provider="fal_ai",
        optional_params={"quality": "medium", "image_size": {"width": 1024, "height": 1024}},
        call_type="aimage_edit",
    )
    assert cost == litellm.model_cost[f"fal_ai/medium/1024-x-1024/{model}"]["output_cost_per_image"] > 0
