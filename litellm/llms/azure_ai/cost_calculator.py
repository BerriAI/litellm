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

    # Get the model router pricing from model_prices_and_context_window.json
    # Use "model_router" as the key (without actual model name suffix)
    model_info: Final = get_model_info(model="model_router", custom_llm_provider="azure_ai")
    router_flat_cost_per_token: Final = model_info.get("input_cost_per_token", 0)

    if router_flat_cost_per_token and router_flat_cost_per_token > 0:
        return prompt_tokens * router_flat_cost_per_token

    return 0.0


ROUTER_FEE_ENTRY_NAMES: Final = frozenset({"model-router", "model_router"})


def _prices_router_fee_itself(model: str) -> bool:
    return model.lower().rsplit("/", 1)[-1] in ROUTER_FEE_ENTRY_NAMES


def _base_cost_per_token(model: str, usage: Usage, service_tier: str | None) -> tuple[float, float] | None:
    try:
        return generic_cost_per_token(
            model=model, usage=usage, custom_llm_provider="azure_ai", service_tier=service_tier
        )
    except Exception as e:
        if not _is_azure_model_router(model):
            raise
        verbose_logger.debug(
            "Azure AI Model Router: model '%s' not in cost map, calculating routing flat cost only. Error: %s", model, e
        )
        return None


def cost_per_token(
    model: str,
    usage: Usage,
    response_time_ms: float | None = 0.0,
    request_model: str | None = None,
    service_tier: str | None = None,
) -> tuple[float, float]:
    """
    Calculate the cost per token for Azure AI models.

    For Azure AI Foundry Model Router the routing fee (the azure_ai/model_router entry, $0.14 per
    million input tokens) is added on top of the routed model's cost. When the response model is
    the router entry itself, generic_cost_per_token has already charged that fee.

    Args:
        model: str, the model name without provider prefix (from response)
        usage: LiteLLM Usage block
        response_time_ms: Optional response time in milliseconds
        request_model: Optional[str], the original request model name (to detect router usage)

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd

    Raises:
        ValueError: If the model is not found in the cost map and cost cannot be calculated
            (except for Model Router models where we return just the routing flat cost)
    """
    is_router_request: Final = _is_azure_model_router(model) or (
        request_model is not None and _is_azure_model_router(request_model)
    )
    base_cost: Final = _base_cost_per_token(model=model, usage=usage, service_tier=service_tier)
    prompt_cost, completion_cost = base_cost if base_cost is not None else (0.0, 0.0)
    if not is_router_request or (base_cost is not None and _prices_router_fee_itself(model)):
        return prompt_cost, completion_cost
    router_flat_cost: Final = calculate_azure_model_router_flat_cost(request_model or model, usage.prompt_tokens)
    return prompt_cost + router_flat_cost, completion_cost
