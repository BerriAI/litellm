"""
Helper util for handling azure openai-specific cost calculation
- e.g.: prompt caching, audio tokens, Model Router fee
"""

from typing import Final

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.llms.azure_ai.cost_calculator import (
    calculate_azure_model_router_flat_cost,
    is_azure_model_router,
    is_router_fee_entry,
)
from litellm.types.utils import Usage


def _router_name(model: str, request_model: str | None) -> str | None:
    if is_azure_model_router(model):
        return model
    if request_model is not None and is_azure_model_router(request_model):
        return request_model
    return None


def cost_per_token(
    model: str,
    usage: Usage,
    response_time_ms: float | None = 0.0,
    service_tier: str | None = None,
    request_model: str | None = None,
) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    When the priced model or request_model is an Azure Model Router name, the router fee is added on top of the
    routed model's own tokens exactly once. A response priced as the router entry itself already carries the fee,
    and a router call whose priced model is missing from the cost map costs the fee alone.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing caching and audio token information
        - request_model: the original request model; a Model Router name adds the routing fee

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    router_name: Final = _router_name(model=model, request_model=request_model)
    if router_name is None:
        return generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider="azure",
            service_tier=service_tier,
        )
    router_fee: Final = calculate_azure_model_router_flat_cost(
        router_name, usage.prompt_tokens, custom_llm_provider="azure"
    )
    try:
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider="azure",
            service_tier=service_tier,
        )
    except Exception as e:
        verbose_logger.debug(
            "Azure Model Router: model '%s' not in cost map, only the routing fee applies. Error: %s", model, e
        )
        return router_fee, 0.0
    if is_router_fee_entry(model, custom_llm_provider="azure"):
        return prompt_cost, completion_cost
    return prompt_cost + router_fee, completion_cost
