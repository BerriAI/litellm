"""
For calculating cost of fireworks ai serverless inference models.
"""

from typing import Tuple

from litellm.constants import (
    FIREWORKS_AI_4_B,
    FIREWORKS_AI_16_B,
    FIREWORKS_AI_56_B_MOE,
    FIREWORKS_AI_176_B_MOE,
)
from litellm.types.utils import Usage
from litellm.utils import get_model_info


# Extract the number of billion parameters from the model name
# only used for together_computer LLMs
def get_base_model_for_pricing(model_name: str) -> str:
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
        if total_billion <= FIREWORKS_AI_56_B_MOE:
            return "fireworks-ai-moe-up-to-56b"
        elif total_billion <= FIREWORKS_AI_176_B_MOE:
            return "fireworks-ai-56b-to-176b"

    # Check for standard models in the form <number>b
    re_params_match = re.search(r"(\d+)b", model_name)
    if re_params_match is not None:
        params_match = str(re_params_match.group(1))
        params_billion = float(params_match)

        # Determine the category based on the number of parameters
        if params_billion <= FIREWORKS_AI_4_B:
            return "fireworks-ai-up-to-4b"
        elif params_billion <= FIREWORKS_AI_16_B:
            return "fireworks-ai-4.1b-to-16b"
        elif params_billion > FIREWORKS_AI_16_B:
            return "fireworks-ai-above-16b"

    # If no matches, return the original model_name
    return "fireworks-ai-default"


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Delegates to generic_cost_per_token which handles prompt caching
    (cache_read_input_tokens, cache_creation_input_tokens) correctly.

    For unmapped models, falls back to parameter-based pricing categories.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing caching information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    from litellm.litellm_core_utils.llm_cost_calc.utils import (
        generic_cost_per_token,
    )

    ## check if model is mapped in model_prices_and_context_window.json
    try:
        get_model_info(model=model, custom_llm_provider="fireworks_ai")
    except Exception:
        ## model not mapped — fall back to parameter-based pricing category
        model = get_base_model_for_pricing(model_name=model)

    return generic_cost_per_token(
        model=model, usage=usage, custom_llm_provider="fireworks_ai"
    )
