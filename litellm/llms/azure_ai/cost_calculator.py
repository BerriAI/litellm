"""
Handles custom cost calculation for Azure AI models.

Custom cost calculation for Azure AI models only requied for rerank.
"""

from typing import Tuple

from litellm.utils import get_model_info


def cost_per_query(model: str, num_queries: int = 1) -> Tuple[float, float]:
    """
    Calculates the cost per query for a given rerank model.

    Input:
        - model: str, the model name without provider prefix

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    model_info = get_model_info(model=model, custom_llm_provider="azure_ai")

    if (
        "input_cost_per_query" not in model_info
        or model_info["input_cost_per_query"] is None
    ):
        return 0.0, 0.0

    prompt_cost = model_info["input_cost_per_query"] * num_queries

    return prompt_cost, 0.0
