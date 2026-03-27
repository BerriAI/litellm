"""
Azure AI cost calculation helper.
Handles Azure AI Foundry Model Router flat cost and other Azure AI specific pricing.
"""

from typing import Optional, Tuple

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    InputCostBreakdown,
    OutputCostBreakdown,
    generic_cost_per_token,
)
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
    model_lower = model.lower()
    return (
        "model-router" in model_lower
        or "model_router" in model_lower
        or model_lower == "azure-model-router"
    )


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
    model_info = get_model_info(model="model_router", custom_llm_provider="azure_ai")
    router_flat_cost_per_token = model_info.get("input_cost_per_token", 0)

    if router_flat_cost_per_token > 0:
        return prompt_tokens * router_flat_cost_per_token

    return 0.0


def cost_per_token(
    model: str,
    usage: Usage,
    response_time_ms: Optional[float] = 0.0,
    request_model: Optional[str] = None,
) -> Tuple[InputCostBreakdown, OutputCostBreakdown]:
    """
    Calculate the cost per token for Azure AI models.

    For Azure AI Foundry Model Router:
    - Adds a flat cost of $0.14 per million input tokens (from model_prices_and_context_window.json)
    - Plus the cost of the actual model used (handled by generic_cost_per_token)

    Args:
        model: str, the model name without provider prefix (from response)
        usage: LiteLLM Usage block
        response_time_ms: Optional response time in milliseconds
        request_model: Optional[str], the original request model name (to detect router usage)

    Returns:
        Tuple[InputCostBreakdown, OutputCostBreakdown] - granular input and output cost breakdowns

    Raises:
        ValueError: If the model is not found in the cost map and cost cannot be calculated
            (except for Model Router models where we return just the routing flat cost)
    """
    input_bd = InputCostBreakdown(total=0.0)
    output_bd = OutputCostBreakdown(total=0.0)

    is_router_request = _is_azure_model_router(model) or (
        request_model is not None and _is_azure_model_router(request_model)
    )

    try:
        input_bd, output_bd = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider="azure_ai",
        )
    except Exception as e:
        if not _is_azure_model_router(model):
            raise
        verbose_logger.debug(
            f"Azure AI Model Router: model '{model}' not in cost map, calculating routing flat cost only. Error: {e}"
        )

    if is_router_request:
        router_model_for_calc = request_model if request_model else model
        router_flat_cost = calculate_azure_model_router_flat_cost(
            router_model_for_calc, usage.prompt_tokens
        )

        if router_flat_cost > 0:
            verbose_logger.debug(
                f"Azure AI Model Router flat cost: ${router_flat_cost:.6f} "
                f"({usage.prompt_tokens} tokens × ${router_flat_cost / usage.prompt_tokens:.9f}/token)"
            )

            input_bd["text_cost"] = input_bd.get("text_cost", 0.0) + router_flat_cost
            input_bd["total"] = input_bd.get("total", 0.0) + router_flat_cost

    return input_bd, output_bd
