"""
Azure AI cost calculation helper.
Handles Azure AI Foundry Model Router flat cost and other Azure AI specific pricing.
"""

from typing import Final

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import Usage
from litellm.utils import get_model_info


def _is_azure_model_router(model: str) -> bool:
    """
    Check if the model is Azure AI Foundry Model Router.

    Detects patterns like:
    - "azure-model-router"
    - "model-router"
    - "model_router/<actual-model>"
    - "model-router/<actual-model>"

    Args:
        model: The model name

    Returns:
        bool: True if this is a model router model
    """
    model_lower: Final = model.lower()
    return "model-router" in model_lower or "model_router" in model_lower or model_lower == "azure-model-router"


ROUTER_FEE_ENTRY_NAMES: Final = frozenset({"model-router", "model_router"})


def is_router_fee_entry(model: str) -> bool:
    return model.lower().removeprefix("azure_ai/") in ROUTER_FEE_ENTRY_NAMES


def _router_fee_entry_name(model: str) -> str:
    entry_name: Final = model.lower().removeprefix("azure_ai/")
    return entry_name if entry_name in ROUTER_FEE_ENTRY_NAMES else "model_router"


def calculate_azure_model_router_flat_cost(model: str, prompt_tokens: int) -> float:
    """
    Calculate the flat cost for Azure AI Foundry Model Router.

    Args:
        model: The model name (should be a model router model)
        prompt_tokens: Number of prompt tokens

    Returns:
        float: The flat cost in USD, or 0.0 if not applicable
    """
    if not _is_azure_model_router(model):
        return 0.0
    model_info: Final = get_model_info(model=_router_fee_entry_name(model), custom_llm_provider="azure_ai")
    router_flat_cost_per_token: Final = model_info.get("input_cost_per_token", 0)
    if router_flat_cost_per_token and router_flat_cost_per_token > 0:
        return prompt_tokens * router_flat_cost_per_token
    return 0.0


def _response_model_cost(model: str, usage: Usage, service_tier: str | None) -> tuple[float, float]:
    try:
        return generic_cost_per_token(
            model=model, usage=usage, custom_llm_provider="azure_ai", service_tier=service_tier
        )
    except Exception as e:
        if not _is_azure_model_router(model):
            raise
        verbose_logger.debug(
            "Azure AI Model Router: model '%s' not in cost map, only the routing fee applies. Error: %s", model, e
        )
        return 0.0, 0.0


def cost_per_token(
    model: str,
    usage: Usage,
    response_time_ms: float | None = 0.0,
    request_model: str | None = None,
    service_tier: str | None = None,
) -> tuple[float, float]:
    """
    Price the response model's own tokens for Azure AI, plus the Model Router fee when the caller names the
    router as the request model.

    completion_cost never passes request_model: it charges the fee once through
    AzureModelRouterConfig.calculate_additional_costs as the "Azure Model Router Flat Cost" line of the cost
    breakdown. A response priced as the router entry itself already carries the fee, so request_model adds
    nothing on top of it, and a router deployment name that is missing from the cost map prices at zero here.

    Args:
        model: str, the model name without provider prefix (from response)
        usage: LiteLLM Usage block
        response_time_ms: Optional response time in milliseconds
        request_model: Optional[str], the original request model name; a Model Router name adds the routing fee
        service_tier: Optional service tier the request was priced on

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd

    Raises:
        ValueError: If a model that is not a Model Router name is missing from the cost map
    """
    prompt_cost, completion_cost = _response_model_cost(model=model, usage=usage, service_tier=service_tier)
    if request_model is None or not _is_azure_model_router(request_model) or is_router_fee_entry(model):
        return prompt_cost, completion_cost
    return prompt_cost + calculate_azure_model_router_flat_cost(request_model, usage.prompt_tokens), completion_cost
