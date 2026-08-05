"""
Helper util for handling azure openai-specific cost calculation
- e.g.: prompt caching, audio tokens
"""

from typing import Final

from litellm._logging import verbose_logger
from litellm.constants import AZURE_WEB_SEARCH_COST_PER_CALL
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    cost_for_web_search_requests,
    generic_cost_per_token,
)
from litellm.types.utils import ModelInfo, Usage
from litellm.utils import get_model_info


def cost_per_web_search_request(usage: Usage, model_info: ModelInfo) -> float | None:
    """
    Cost of the hosted web search tool on Azure OpenAI, charged per billable search.

    Azure serves the tool through Grounding with Bing Search at $14 / 1k transactions,
    so it is priced apart from OpenAI's own $10 / 1k calls.

    https://www.microsoft.com/en-us/bing/apis/grounding-pricing
    """
    return cost_for_web_search_requests(
        usage=usage,
        model_info=model_info,
        default_cost_per_request=AZURE_WEB_SEARCH_COST_PER_CALL,
    )


def cost_per_token(
    model: str,
    usage: Usage,
    response_time_ms: float | None = 0.0,
    service_tier: str | None = None,
) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing caching and audio token information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    ## GET MODEL INFO
    model_info: Final = get_model_info(model=model, custom_llm_provider="azure")

    ## Speech / Audio cost calculation (cost per second for TTS models)
    if (
        "output_cost_per_second" in model_info
        and model_info["output_cost_per_second"] is not None
        and response_time_ms is not None
    ):
        verbose_logger.debug(
            "For model=%s - output_cost_per_second: %s; response time: %s",
            model,
            model_info.get("output_cost_per_second"),
            response_time_ms,
        )
        ## COST PER SECOND ##
        prompt_cost: Final = 0.0
        completion_cost: Final = model_info["output_cost_per_second"] * response_time_ms / 1000
        return prompt_cost, completion_cost

    ## Use generic cost calculator for all other cases
    ## This properly handles: text tokens, audio tokens, cached tokens, reasoning tokens, etc.
    return generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="azure",
        service_tier=service_tier,
    )
