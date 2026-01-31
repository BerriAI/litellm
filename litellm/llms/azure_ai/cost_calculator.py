"""
Azure AI cost calculation helper.
Handles Azure AI Foundry Model Router flat cost and other Azure AI specific pricing.
"""

from typing import Optional, Tuple

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
    model: str, usage: Usage, response_time_ms: Optional[float] = 0.0
) -> Tuple[float, float]:
    """
    Calculate the cost per token for Azure AI models.
    
    For Azure AI Foundry Model Router:
    - Adds a flat cost of $0.14 per million input tokens (from model_prices_and_context_window.json)
    - The Model Router itself is just a routing layer, so no additional model cost is added
      unless the actual model is also specified
    
    Args:
        model: str, the model name without provider prefix
        usage: LiteLLM Usage block
        response_time_ms: Optional response time in milliseconds
    
    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    prompt_cost = 0.0
    completion_cost = 0.0
    
    # Check if this is a Model Router deployment
    is_model_router = _is_azure_model_router(model)
    
    # Try to calculate base cost using generic cost calculator
    # For Model Router deployments, this may fail if the deployment name isn't mapped
    try:
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider="azure_ai",
        )
    except Exception as e:
        # If model isn't mapped and this is a Model Router deployment, that's expected
        # Model Router is just a routing layer - the actual model cost would be separate
        if is_model_router:
            verbose_logger.debug(
                f"Model Router deployment '{model}' not found in model cost map. "
                f"Only flat routing cost will be calculated. Error: {str(e)}"
            )
        else:
            # Re-raise if it's not a model router - the model should be mapped
            raise
    
    # Add flat cost for Azure Model Router
    # The flat cost is defined in model_prices_and_context_window.json for azure_ai/model_router
    if is_model_router:
        router_flat_cost = calculate_azure_model_router_flat_cost(model, usage.prompt_tokens)
        
        if router_flat_cost > 0:
            verbose_logger.debug(
                f"Azure AI Model Router flat cost: ${router_flat_cost:.6f} "
                f"({usage.prompt_tokens} tokens × ${router_flat_cost / usage.prompt_tokens:.9f}/token)"
            )
            
            # Add flat cost to prompt cost
            prompt_cost += router_flat_cost
    
    return prompt_cost, completion_cost
