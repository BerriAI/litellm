"""
Azure AI cost calculation helper.
Handles Azure AI Foundry Model Router flat cost and other Azure AI specific pricing.
"""

from typing import Optional, Tuple

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import Usage


# Azure AI Foundry Model Router pricing
# Source: https://azure.microsoft.com/en-us/pricing/details/ai-services/
AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS = 0.14  # $0.14 per M input tokens


def _is_azure_model_router(model: str) -> bool:
    """
    Check if the model is Azure AI Foundry Model Router.
    
    Args:
        model: The model name (e.g., "azure-model-router", "model-router")
    
    Returns:
        bool: True if this is a model router model
    """
    model_lower = model.lower()
    return "model-router" in model_lower or model_lower == "azure-model-router"


def cost_per_token(
    model: str, usage: Usage, response_time_ms: Optional[float] = 0.0
) -> Tuple[float, float]:
    """
    Calculate the cost per token for Azure AI models.
    
    For Azure AI Foundry Model Router:
    - Adds a flat cost of $0.14 per million input tokens
    - Plus the cost of the actual model used (handled by generic_cost_per_token)
    
    Args:
        model: str, the model name without provider prefix
        usage: LiteLLM Usage block
        response_time_ms: Optional response time in milliseconds
    
    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    # Calculate base cost using generic cost calculator
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="azure_ai",
    )
    
    # Add flat cost for Azure Model Router
    if _is_azure_model_router(model):
        # Flat cost per million input tokens
        flat_cost_per_token = AZURE_MODEL_ROUTER_FLAT_COST_PER_M_INPUT_TOKENS / 1_000_000
        router_flat_cost = usage.prompt_tokens * flat_cost_per_token
        
        verbose_logger.debug(
            f"Azure AI Model Router flat cost: ${router_flat_cost:.6f} "
            f"({usage.prompt_tokens} tokens × ${flat_cost_per_token:.9f}/token)"
        )
        
        # Add flat cost to prompt cost
        prompt_cost += router_flat_cost
    
    return prompt_cost, completion_cost
