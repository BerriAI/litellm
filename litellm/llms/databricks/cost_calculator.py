"""
Helper util for handling databricks-specific cost calculation
- e.g.: handling 'dbrx-instruct-*'
"""

from typing import Tuple

from litellm.types.utils import Usage
from litellm.utils import get_model_info


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    base_model = model
    if model.startswith("databricks/dbrx-instruct") or model.startswith(
        "dbrx-instruct"
    ):
        base_model = "databricks-dbrx-instruct"

    ## GET MODEL INFO
    model_info = get_model_info(model=base_model, custom_llm_provider="databricks")

    ## CALCULATE INPUT COST

    prompt_cost: float = usage["prompt_tokens"] * model_info["input_cost_per_token"]

    ## CALCULATE OUTPUT COST
    completion_cost = usage["completion_tokens"] * model_info["output_cost_per_token"]

    return prompt_cost, completion_cost
