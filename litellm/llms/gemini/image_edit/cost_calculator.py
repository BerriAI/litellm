"""
Gemini Image Edit Cost Calculator
"""

from litellm.litellm_core_utils.llm_cost_calc.utils import ImageTokenCostReporter
from litellm.llms.gemini.image_generation.cost_calculator import (
    cost_calculator as image_generation_cost_calculator,
)
from litellm.types.utils import ModelInfo


def cost_calculator(
    model: str,
    image_response: object,
    model_info: ModelInfo | None = None,
    *,
    report_token_cost: ImageTokenCostReporter | None = None,
) -> float:
    """
    Gemini image edit cost calculator.

    Gemini image edits and generations share image response billing behavior:
    use provider token usage when present, otherwise fall back to per-image pricing.
    """
    return image_generation_cost_calculator(
        model=model,
        image_response=image_response,
        model_info=model_info,
        report_token_cost=report_token_cost,
    )
