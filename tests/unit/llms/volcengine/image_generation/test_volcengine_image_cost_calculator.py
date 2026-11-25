from typing import Final

import httpx
import pytest

from litellm import get_model_info
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils
from litellm.llms.volcengine.image_generation.cost_calculator import cost_calculator
from litellm.llms.volcengine.image_generation.transformation import ark_image_response
from litellm.types.utils import ImageObject, ImageResponse, ImageUsage, ModelInfo

ARK_SEEDREAM_MODELS: Final = (
    "volcengine/doubao-seedream-4-0-260415",
    "volcengine/doubao-seedream-4-0-20260415",
    "volcengine/doubao-seedream-5-0-flash-260915",
    "volcengine/doubao-seedream-5-0-pro-260628",
    "volcengine/doubao-seedream-5-5-261028",
)
BYTEPLUS_SEEDREAM_MODELS: Final = (
    "dola-seedream-5-5-261028",
    "dola-seedream-5-0-pro-260628",
    "dola-seedream-5-0-flash-260915",
    "seedream-5-0-lite-260128",
    "seedream-4-5-251128",
    "seedream-4-0-250828",
)


def _response(sizes: tuple[str, ...], *, input_images: int = 0, layers: tuple[int, ...] = ()) -> ImageResponse:
    return ImageResponse(
        created=1,
        data=[
            ImageObject(
                url=f"https://ark.example/{index}.png",
                provider_specific_fields={"size": size, "z_index": layers[index] if index < len(layers) else 0},
            )
            for index, size in enumerate(sizes)
        ],
        usage=ImageUsage.model_validate(
            {
                "input_tokens": 0,
                "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                "output_tokens": 0,
                "total_tokens": 0,
                "input_images": input_images,
            }
        ),
    )


def test_output_images_bill_at_the_pixel_tier_ark_reports() -> None:
    model_info: Final[ModelInfo] = {
        "output_cost_per_image": 2.0,
        "output_cost_per_image_1K": 1.0,
        "output_cost_per_image_2K": 2.0,
    }

    cost: Final = cost_calculator(
        model="doubao-seedream-5-5-261028",
        image_response=_response(("1024x1024", "2048x2048")),
        model_info=model_info,
    )

    assert cost == 3.0


# Tier cutoff is 1536^2 x 1.05^2 pixels per https://docs.byteplus.com/en/docs/ModelArk/1544106 (updated 2026-11-24)
@pytest.mark.parametrize(("size", "expected_cost"), [("1612x1613", 1.0), ("1612x1614", 2.0)])
def test_an_output_just_past_the_ark_tier_cutoff_bills_the_higher_tier(size: str, expected_cost: float) -> None:
    model_info: Final[ModelInfo] = {"output_cost_per_image_1K": 1.0, "output_cost_per_image_2K": 2.0}

    cost: Final = cost_calculator(
        model="doubao-seedream-5-5-261028",
        image_response=_response((size,)),
        model_info=model_info,
    )

    assert cost == expected_cost


def test_layer_decomposition_outputs_bill_at_the_layer_rate() -> None:
    model_info: Final[ModelInfo] = {
        "output_cost_per_image": 2.0,
        "output_cost_per_image_1K": 2.0,
        "provider_specific_entry": {
            "volcengine_image": {
                "layer_output_cost_per_image_1K": 0.5,
                "layer_output_cost_per_image_2K": 1.5,
            }
        },
    }

    cost: Final = cost_calculator(
        model="doubao-seedream-5-0-pro-260628",
        image_response=_response(("1024x1024",), layers=(1,)),
        model_info=model_info,
    )

    assert cost == 0.5


def test_every_output_of_a_layer_decomposition_bills_at_the_layer_rate() -> None:
    model_info: Final[ModelInfo] = {
        "output_cost_per_image": 2.0,
        "output_cost_per_image_1K": 2.0,
        "provider_specific_entry": {"volcengine_image": {"layer_output_cost_per_image_1K": 0.5}},
    }

    cost: Final = cost_calculator(
        model="doubao-seedream-5-0-pro-260628",
        image_response=_response(("1024x1024", "1024x1024", "1024x1024"), layers=(0, 1, 2)),
        model_info=model_info,
    )

    assert cost == 1.5


def test_images_ark_failed_to_output_are_not_billed() -> None:
    ark_body: Final = {
        "created": 1,
        "data": [
            {"url": "https://ark.example/0.png", "size": "1024x1024"},
            {"error": {"code": "OutputImageSensitiveContentDetected", "message": "blocked"}},
        ],
        "usage": {"generated_images": 1, "output_tokens": 4096, "total_tokens": 4096},
    }

    cost: Final = cost_calculator(
        model="doubao-seedream-5-5-261028",
        image_response=ark_image_response(httpx.Response(200, json=ark_body)),
        model_info={"output_cost_per_image": 2.0, "output_cost_per_image_1K": 1.0},
    )

    assert cost == 1.0


def test_input_images_bill_after_the_free_allowance() -> None:
    model_info: Final[ModelInfo] = {
        "output_cost_per_image": 1.0,
        "input_cost_per_image": 0.25,
        "provider_specific_entry": {"volcengine_image": {"free_input_images": 1}},
    }

    cost: Final = cost_calculator(
        model="doubao-seedream-5-5-261028",
        image_response=_response(("1024x1024",), input_images=3),
        model_info=model_info,
    )

    assert cost == 1.5


def test_a_deployment_per_image_price_replaces_the_shared_tier_prices() -> None:
    cost: Final = cost_calculator(
        model="doubao-seedream-5-5-261028",
        image_response=_response(("1024x1024",)),
        model_info={"output_cost_per_image": 3.0},
    )

    assert cost == 3.0


def test_a_deployment_per_image_price_also_replaces_the_shared_layer_prices() -> None:
    cost: Final = cost_calculator(
        model="doubao-seedream-5-0-pro-260628",
        image_response=_response(("1024x1024", "1024x1024", "1024x1024"), layers=(0, 1, 2)),
        model_info={"output_cost_per_image": 3.0},
    )

    assert cost == 9.0


def test_a_deployment_flat_price_covers_the_tier_the_deployment_does_not_price() -> None:
    model_info: Final[ModelInfo] = {"output_cost_per_image": 3.0, "output_cost_per_image_1K": 2.0}

    cost: Final = cost_calculator(
        model="doubao-seedream-5-5-261028",
        image_response=_response(("1024x1024", "2048x2048")),
        model_info=model_info,
    )

    assert cost == 5.0


@pytest.mark.parametrize("model", ARK_SEEDREAM_MODELS)
def test_every_ark_seedream_model_is_priced_as_an_image_model(model: str) -> None:
    catalog: Final = GetModelCostMap.load_local_model_cost_map()
    info: Final = catalog[model]

    assert info["mode"] == "image_generation"
    assert info["litellm_provider"] == "volcengine"
    assert info["output_cost_per_image"] > 0
    assert info["source"].startswith("https://")


@pytest.mark.parametrize("model", BYTEPLUS_SEEDREAM_MODELS)
def test_a_byteplus_model_name_is_priced_without_a_base_model(model: str) -> None:
    info: Final = GetModelCostMap.load_local_model_cost_map()[f"volcengine/{model}"]

    cost: Final = cost_calculator(model=model, image_response=_response(("1024x1024",)))

    assert cost == info.get("output_cost_per_image_1K", info["output_cost_per_image"])
    assert cost > 0


def test_the_volcengine_image_route_uses_the_provider_calculator() -> None:
    catalog: Final = GetModelCostMap.load_local_model_cost_map()
    response: Final = _response(("1248x832",), input_images=1)

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="doubao-seedream-5-5-261028",
        completion_response=response,
        custom_llm_provider="volcengine",
        model_info=catalog["volcengine/doubao-seedream-5-5-261028"],
    )

    assert cost == cost_calculator(model="doubao-seedream-5-5-261028", image_response=response)


@pytest.mark.parametrize(
    ("size", "tier_key"), (("1024x1024", "output_cost_per_image_1K"), ("2048x2048", "output_cost_per_image_2K"))
)
def test_a_cataloged_seedream_image_bills_at_the_catalog_price_of_its_pixel_tier(size: str, tier_key: str) -> None:
    catalog: Final = GetModelCostMap.load_local_model_cost_map()

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="doubao-seedream-5-5-261028",
        completion_response=_response((size,)),
        custom_llm_provider="volcengine",
    )

    assert cost == pytest.approx(catalog["volcengine/doubao-seedream-5-5-261028"][tier_key])


def test_a_deployment_tier_price_replaces_the_catalog_tier_price() -> None:
    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="doubao-seedream-5-5-261028",
        completion_response=_response(("1024x1024", "2048x2048")),
        custom_llm_provider="volcengine",
        model_info={"output_cost_per_image_1K": 0.5, "output_cost_per_image_2K": 0.75},
    )

    assert cost == pytest.approx(1.25)


def test_get_model_info_reports_the_catalog_tier_prices() -> None:
    catalog_row: Final = GetModelCostMap.load_local_model_cost_map()["volcengine/doubao-seedream-5-5-261028"]

    info: Final = get_model_info(model="doubao-seedream-5-5-261028", custom_llm_provider="volcengine")

    assert (info.get("output_cost_per_image_1K"), info.get("output_cost_per_image_2K")) == (
        catalog_row["output_cost_per_image_1K"],
        catalog_row["output_cost_per_image_2K"],
    )


@pytest.mark.parametrize("model", ["ep-20261101-abc", "doubao-seedream-5-5-261028"])
@pytest.mark.parametrize(
    ("deployment_price", "expected_cost"),
    [({"input_cost_per_pixel": 1e-8}, 1024 * 1024 * 1e-8), ({"input_cost_per_image": 0.03}, 0.03)],
    ids=["per-pixel", "per-input-image"],
)
def test_a_deployment_priced_without_an_output_image_price_bills_its_own_price(
    model: str, deployment_price: ModelInfo, expected_cost: float
) -> None:
    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model=model,
        completion_response=_response(("1024x1024",)),
        custom_llm_provider="volcengine",
        model_info=deployment_price,
    )

    assert cost == pytest.approx(expected_cost)


@pytest.mark.parametrize(("input_images", "expected_cost"), [(0, 0.03), (1, 0.04)], ids=["generation", "edit"])
def test_a_deployment_with_both_image_prices_bills_outputs_and_inputs_like_the_cost_map(
    input_images: int, expected_cost: float
) -> None:
    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="ep-20261101-abc",
        completion_response=_response(("1024x1024",), input_images=input_images),
        custom_llm_provider="volcengine",
        model_info={"input_cost_per_image": 0.01, "output_cost_per_image": 0.03},
    )

    assert cost == pytest.approx(expected_cost)


@pytest.mark.parametrize(
    ("size", "input_images", "expected_cost"),
    [("1024x1024", 0, 0.02), ("2048x2048", 0, 0.05), ("1024x1024", 1, 0.03)],
    ids=["1K", "2K", "1K-edit"],
)
def test_a_deployment_that_prices_tiers_and_input_images_bills_each_tier(
    size: str, input_images: int, expected_cost: float
) -> None:
    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="ep-20261101-abc",
        completion_response=_response((size,), input_images=input_images),
        custom_llm_provider="volcengine",
        model_info={"input_cost_per_image": 0.01, "output_cost_per_image_1K": 0.02, "output_cost_per_image_2K": 0.05},
    )

    assert cost == pytest.approx(expected_cost)
