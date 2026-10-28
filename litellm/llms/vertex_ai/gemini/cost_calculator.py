"""
Cost calculator for Vertex AI Gemini.

Delegates to the shared Gemini cost calculator which reads pricing and
billing unit from model_info.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo, Usage


def cost_per_web_search_request(usage: "Usage", model_info: "ModelInfo") -> float:
    """
    Calculate the cost of a web search request for Vertex AI Gemini.

    Billing differs by ``web_search_billing_unit`` in ``model_info``:
    - ``"per_query"``: charged per individual search query (Gemini 3.x).
    - ``"per_prompt"`` (default): charged per grounded prompt (Gemini 2.x).

    Delegates to the shared Gemini cost calculator.
    """
    from litellm.llms.gemini.cost_calculator import (
        cost_per_web_search_request as _gemini_cost,
    )

    return _gemini_cost(usage=usage, model_info=model_info)
