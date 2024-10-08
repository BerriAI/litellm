"""
Helper util for handling azure openai-specific cost calculation
- e.g.: prompt caching
"""

from typing import Optional, Tuple

from litellm._logging import verbose_logger
from litellm.types.utils import Usage
from litellm.utils import get_model_info


def cost_per_token(
    model: str, usage: Usage, response_time_ms: Optional[float] = 0.0
) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    ## GET MODEL INFO
    model_info = get_model_info(model=model, custom_llm_provider="azure")

    ## CALCULATE INPUT COST
    prompt_cost: float = usage["prompt_tokens"] * model_info["input_cost_per_token"]

    ## CALCULATE OUTPUT COST
    completion_cost: float = (
        usage["completion_tokens"] * model_info["output_cost_per_token"]
    )

    ## Prompt Caching cost calculation
    if model_info.get("cache_read_input_token_cost") is not None:
        # Note: We read ._cache_read_input_tokens from the Usage - since cost_calculator.py standardizes the cache read tokens on usage._cache_read_input_tokens
        prompt_cost += usage._cache_read_input_tokens * (
            model_info.get("cache_read_input_token_cost", 0) or 0
        )

    ## Speech / Audio cost calculation
    if (
        "output_cost_per_second" in model_info
        and model_info["output_cost_per_second"] is not None
        and response_time_ms is not None
    ):
        verbose_logger.debug(
            f"For model={model} - output_cost_per_second: {model_info.get('output_cost_per_second')}; response time: {response_time_ms}"
        )
        ## COST PER SECOND ##
        prompt_cost = 0
        completion_cost = model_info["output_cost_per_second"] * response_time_ms / 1000

    return prompt_cost, completion_cost
