"""
For calculating cost of fireworks ai serverless inference models.
"""

from typing import Tuple

from litellm.types.utils import Usage
from litellm.utils import get_model_info


# Extract the number of billion parameters from the model name
# only used for together_computer LLMs
def get_model_params_and_category(model_name: str) -> str:
    """
    Helper function for calculating together ai pricing.

    Returns:
    - str: model pricing category if mapped else received model name
    """
    import re

    model_name = model_name.lower()

    # Check for MoE models in the form <number>x<number>b
    moe_match = re.search(r"(\d+)x(\d+)b", model_name)
    if moe_match:
        total_billion = int(moe_match.group(1)) * int(moe_match.group(2))
        if total_billion <= 56:
            return "fireworks-ai-moe-up-to-56b"
        elif total_billion <= 176:
            return "fireworks-ai-56b-to-176b"

    # Check for standard models in the form <number>b
    re_params_match = re.search(r"(\d+)b", model_name)
    if re_params_match is not None:
        params_match = str(re_params_match.group(1))
        params_billion = float(params_match)

        # Determine the category based on the number of parameters
        if params_billion <= 16.0:
            return "fireworks-ai-up-to-16b"
        elif params_billion <= 80.0:
            return "fireworks-ai-16b-80b"

    # If no matches, return the original model_name
    return model_name


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    base_model = get_model_params_and_category(model_name=model)

    ## GET MODEL INFO
    model_info = get_model_info(model=base_model, custom_llm_provider="fireworks_ai")

    ## CALCULATE INPUT COST

    prompt_cost: float = usage["prompt_tokens"] * model_info["input_cost_per_token"]

    ## CALCULATE OUTPUT COST
    completion_cost = usage["completion_tokens"] * model_info["output_cost_per_token"]

    return prompt_cost, completion_cost
