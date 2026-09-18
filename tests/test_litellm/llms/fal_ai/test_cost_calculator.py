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
