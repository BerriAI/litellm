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


def _price(key: str) -> float:
    return float(litellm.model_cost[key]["output_cost_per_image"])


def test_high_quality_1024x1024_uses_keyed_price():
    cost = cost_calculator(
        model="openai/gpt-image-2",
        image_response=_image_response(),
        optional_params={"quality": "high", "image_size": {"width": 1024, "height": 1024}},
    )
    assert cost == pytest.approx(_price("fal_ai/high/1024-x-1024/openai/gpt-image-2"))


def test_alias_model_uses_keyed_price():
    cost = cost_calculator(
        model="gpt-image-2",
        image_response=_image_response(),
        optional_params={"quality": "high", "image_size": {"width": 1024, "height": 1024}},
    )
    assert cost == pytest.approx(_price("fal_ai/high/1024-x-1024/openai/gpt-image-2"))


def test_provider_prefixed_model_uses_keyed_price():
    cost = cost_calculator(
        model="fal_ai/openai/gpt-image-2",
        image_response=_image_response(),
        optional_params={"quality": "high", "image_size": {"width": 1024, "height": 1024}},
    )
    assert cost == pytest.approx(_price("fal_ai/high/1024-x-1024/openai/gpt-image-2"))


def test_provider_prefixed_edit_model_uses_keyed_edit_price():
    cost = cost_calculator(
        model="fal_ai/openai/gpt-image-2/edit",
        image_response=_image_response(),
        optional_params={"quality": "high", "image_size": {"width": 1024, "height": 1024}},
    )
    assert cost == pytest.approx(_price("fal_ai/high/1024-x-1024/openai/gpt-image-2/edit"))


def test_default_request_priced_at_default_size_and_quality():
    cost: Final = cost_calculator(
        model="openai/gpt-image-2",
        image_response=_image_response(),
        optional_params={},
    )
    no_params_cost: Final = cost_calculator(
        model="openai/gpt-image-2",
        image_response=_image_response(),
        optional_params=None,
    )
    keyed_cost: Final = cost_calculator(
        model="openai/gpt-image-2",
        image_response=_image_response(),
        optional_params={"quality": "high", "image_size": {"width": 1024, "height": 1024}},
    )
    assert cost == pytest.approx(no_params_cost)
    assert cost != pytest.approx(keyed_cost)


def test_auto_quality_priced_as_high():
    cost = cost_calculator(
        model="openai/gpt-image-2",
        image_response=_image_response(),
        optional_params={"quality": "auto", "image_size": {"width": 1024, "height": 1024}},
    )
    assert cost == pytest.approx(_price("fal_ai/high/1024-x-1024/openai/gpt-image-2"))


def test_low_quality_4k_uses_keyed_price():
    cost = cost_calculator(
        model="openai/gpt-image-2",
        image_response=_image_response(),
        optional_params={"quality": "low", "image_size": {"width": 3840, "height": 2160}},
    )
    assert cost == pytest.approx(_price("fal_ai/low/3840-x-2160/openai/gpt-image-2"))


def test_named_fal_size_uses_keyed_price():
    cost = cost_calculator(
        model="openai/gpt-image-2",
        image_response=_image_response(),
        optional_params={"quality": "high", "image_size": "square_hd"},
    )
    assert cost == pytest.approx(_price("fal_ai/high/1024-x-1024/openai/gpt-image-2"))


def test_edit_model_uses_keyed_edit_price():
    cost = cost_calculator(
        model="openai/gpt-image-2/edit",
        image_response=_image_response(),
        optional_params={"quality": "high", "image_size": {"width": 1024, "height": 1024}},
    )
    assert cost == pytest.approx(_price("fal_ai/high/1024-x-1024/openai/gpt-image-2/edit"))


def test_edit_model_without_size_falls_back_to_flat_price():
    cost: Final = cost_calculator(
        model="openai/gpt-image-2/edit",
        image_response=_image_response(),
        optional_params={"quality": "high"},
    )
    no_params_cost: Final = cost_calculator(
        model="openai/gpt-image-2/edit",
        image_response=_image_response(),
        optional_params=None,
    )
    keyed_cost: Final = cost_calculator(
        model="openai/gpt-image-2/edit",
        image_response=_image_response(),
        optional_params={"quality": "high", "image_size": {"width": 1024, "height": 1024}},
    )
    assert cost == pytest.approx(no_params_cost)
    assert cost != pytest.approx(keyed_cost)


def test_missing_optional_params_falls_back_to_flat_price():
    cost: Final = cost_calculator(
        model="openai/gpt-image-2",
        image_response=_image_response(),
        optional_params=None,
    )
    default_cost: Final = cost_calculator(
        model="openai/gpt-image-2",
        image_response=_image_response(),
        optional_params={},
    )
    keyed_cost: Final = cost_calculator(
        model="openai/gpt-image-2",
        image_response=_image_response(),
        optional_params={"quality": "high", "image_size": {"width": 1024, "height": 1024}},
    )
    assert cost == pytest.approx(default_cost)
    assert cost != pytest.approx(keyed_cost)


def test_unlisted_size_falls_back_to_flat_price():
    cost: Final = cost_calculator(
        model="openai/gpt-image-2",
        image_response=_image_response(),
        optional_params={"quality": "high", "image_size": {"width": 999, "height": 999}},
    )
    no_params_cost: Final = cost_calculator(
        model="openai/gpt-image-2",
        image_response=_image_response(),
        optional_params=None,
    )
    keyed_cost: Final = cost_calculator(
        model="openai/gpt-image-2",
        image_response=_image_response(),
        optional_params={"quality": "high", "image_size": {"width": 1024, "height": 1024}},
    )
    assert cost == pytest.approx(no_params_cost)
    assert cost != pytest.approx(keyed_cost)


def test_keyed_price_multiplies_per_image():
    cost = cost_calculator(
        model="openai/gpt-image-2",
        image_response=_image_response(num_images=2),
        optional_params={"quality": "high", "image_size": {"width": 1024, "height": 1024}},
    )
    assert cost == pytest.approx(2 * _price("fal_ai/high/1024-x-1024/openai/gpt-image-2"))


def test_route_image_generation_passes_optional_params_to_fal():
    cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="openai/gpt-image-2",
        completion_response=_image_response(),
        custom_llm_provider="fal_ai",
        optional_params={"quality": "high", "image_size": {"width": 1024, "height": 1024}},
    )
    assert cost == pytest.approx(_price("fal_ai/high/1024-x-1024/openai/gpt-image-2"))


def test_route_image_generation_with_provider_prefixed_model_uses_keyed_price():
    cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="fal_ai/openai/gpt-image-2",
        completion_response=_image_response(),
        custom_llm_provider="fal_ai",
        optional_params={"quality": "high", "image_size": {"width": 1024, "height": 1024}},
    )
    assert cost == pytest.approx(_price("fal_ai/high/1024-x-1024/openai/gpt-image-2"))
