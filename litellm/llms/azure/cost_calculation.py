"""
Helper util for handling azure openai-specific cost calculation
- e.g.: prompt caching, audio tokens
"""

from typing import Optional, Tuple

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import Usage
from litellm.utils import get_model_info


def cost_per_token(
    model: str, usage: Usage, response_time_ms: Optional[float] = 0.0
) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing caching and audio token information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    ## GET MODEL INFO
    model_info = get_model_info(model=model, custom_llm_provider="azure")

    ## Speech / Audio cost calculation (cost per second for TTS models)
    if (
        "output_cost_per_second" in model_info
        and model_info["output_cost_per_second"] is not None
        and response_time_ms is not None
    ):
        verbose_logger.debug(
            f"For model={model} - output_cost_per_second: {model_info.get('output_cost_per_second')}; response time: {response_time_ms}"
        )
        ## COST PER SECOND ##
        prompt_cost = 0.0
        completion_cost = model_info["output_cost_per_second"] * response_time_ms / 1000
        return prompt_cost, completion_cost

    ## Use generic cost calculator for all other cases
    ## This properly handles: text tokens, audio tokens, cached tokens, reasoning tokens, etc.
    return generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="azure",
    )
