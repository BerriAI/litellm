import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm

litellm.model_cost = litellm.get_model_cost_map(url="")
from litellm.llms.together_ai.image_generation.cost_calculator import cost_calculator
from litellm.types.utils import ImageObject, ImageResponse


def test_seedream_pricing_registered():
    info = litellm.get_model_info(
        model="ByteDance-Seed/Seedream-4.0",
        custom_llm_provider=litellm.LlmProviders.TOGETHER_AI.value,
    )
    assert info["output_cost_per_image"] == 0.03
    assert info["mode"] == "image_generation"


def test_cost_calculator_scales_with_image_count():
    image_response = ImageResponse(
        data=[ImageObject(url="https://x/1.png"), ImageObject(url="https://x/2.png")]
    )
    cost = cost_calculator(model="ByteDance-Seed/Seedream-4.0", image_response=image_response)
    assert cost == pytest.approx(0.06)


@pytest.mark.parametrize("call_type", ["image_generation", "image_edit"])
def test_completion_cost_routes_together_ai_image_calls(call_type):
    image_response = ImageResponse(data=[ImageObject(url="https://x/1.png")])
    cost = litellm.completion_cost(
        completion_response=image_response,
        model="together_ai/ByteDance-Seed/Seedream-4.0",
        call_type=call_type,
    )
    assert cost == pytest.approx(0.03)
