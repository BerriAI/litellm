from typing import Final

import pytest

from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils
from litellm.llms.volcengine.image_generation.cost_calculator import cost_calculator
from litellm.types.utils import ImageObject, ImageResponse, ImageUsage, ModelInfo

ARK_SEEDREAM_MODELS: Final = (
    "volcengine/doubao-seedream-4-0-250828",
    "volcengine/doubao-seedream-4-5-251128",
    "volcengine/doubao-seedream-5-0-260128",
    "volcengine/doubao-seedream-5-0-flash-260915",
    "volcengine/doubao-seedream-5-0-lite-260128",
    "volcengine/doubao-seedream-5-0-pro-260628",
    "volcengine/doubao-seedream-5-5-261028",
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


def test_an_unpriced_model_costs_nothing_instead_of_raising() -> None:
    cost: Final = cost_calculator(
        model="doubao-seedream-5-5-261028",
        image_response=_response(("1024x1024",)),
        model_info={},
    )

    assert cost == 0.0


@pytest.mark.parametrize("model", ARK_SEEDREAM_MODELS)
def test_every_ark_seedream_model_is_priced_as_an_image_model(model: str) -> None:
    catalog: Final = GetModelCostMap.load_local_model_cost_map()
    info: Final = catalog[model]

    assert info["mode"] == "image_generation"
    assert info["litellm_provider"] == "volcengine"
    assert info["output_cost_per_image"] > 0
    assert info["source"].startswith("https://")


def test_the_volcengine_image_route_uses_the_provider_calculator() -> None:
    catalog: Final = GetModelCostMap.load_local_model_cost_map()
    response: Final = _response(("1248x832",), input_images=1)

    cost: Final = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="doubao-seedream-5-5-261028",
        completion_response=response,
        custom_llm_provider="volcengine",
        model_info=catalog["volcengine/doubao-seedream-5-5-261028"],
    )

    assert cost > 0
