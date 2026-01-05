"""
COST TRACKING SETTINGS MANAGEMENT

Endpoints for managing cost discount and margin configuration

GET /config/cost_discount_config - Get current cost discount configuration
PATCH /config/cost_discount_config - Update cost discount configuration
GET /config/cost_margin_config - Get current cost margin configuration
PATCH /config/cost_margin_config - Update cost margin configuration
POST /cost/estimate - Estimate cost for a given model and token counts
"""

from typing import Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.cost_calculator import completion_cost
from litellm.proxy._types import (
    CommonProxyErrors,
    CostEstimateRequest,
    CostEstimateResponse,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.utils import LlmProvidersSet

router = APIRouter()


@router.get(
    "/config/cost_discount_config",
    tags=["Cost Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_cost_discount_config(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get current cost discount configuration.
    
    Returns the cost_discount_config from litellm_settings.
    """
    from litellm.proxy.proxy_server import prisma_client, proxy_config
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    try:
        # Load config from DB
        config = await proxy_config.get_config()
        
        # Get cost_discount_config from litellm_settings
        litellm_settings = config.get("litellm_settings", {})
        cost_discount_config = litellm_settings.get("cost_discount_config", {})
        
        return {"values": cost_discount_config}
    except Exception as e:
        verbose_proxy_logger.error(
            f"Error fetching cost discount config: {str(e)}"
        )
        return {"values": {}}


@router.patch(
    "/config/cost_discount_config",
    tags=["Cost Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_cost_discount_config(
    cost_discount_config: Dict[str, float],
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update cost discount configuration.
    
    Updates the cost_discount_config in litellm_settings.
    Discounts should be between 0 and 1 (e.g., 0.05 = 5% discount).
    
    Example:
    ```json
    {
        "vertex_ai": 0.05,
        "gemini": 0.05,
        "openai": 0.01
    }
    ```
    """
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_config,
        store_model_in_db,
    )
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    if store_model_in_db is not True:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."
            },
        )
    
    # Validate that all providers are valid LiteLLM providers
    invalid_providers = []
    for provider in cost_discount_config.keys():
        if provider not in LlmProvidersSet:
            invalid_providers.append(provider)
    
    if invalid_providers:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Invalid provider(s): {', '.join(invalid_providers)}. Must be valid LiteLLM providers. See https://docs.litellm.ai/docs/providers for the full list."
            },
        )
    
    # Validate discount values are between 0 and 1
    for provider, discount in cost_discount_config.items():
        if not isinstance(discount, (int, float)):
            raise HTTPException(
                status_code=400,
                detail=f"Discount for {provider} must be a number"
            )
        if not (0 <= discount <= 1):
            raise HTTPException(
                status_code=400,
                detail=f"Discount for {provider} must be between 0 and 1 (0% to 100%)"
            )
    
    try:
        # Load existing config
        config = await proxy_config.get_config()
        
        # Ensure litellm_settings exists
        if "litellm_settings" not in config:
            config["litellm_settings"] = {}
        
        # Update cost_discount_config
        config["litellm_settings"]["cost_discount_config"] = cost_discount_config
        
        # Save the updated config to DB
        await proxy_config.save_config(new_config=config)
        
        # Update in-memory litellm.cost_discount_config
        litellm.cost_discount_config = cost_discount_config
        
        verbose_proxy_logger.info(
            f"Updated cost_discount_config: {cost_discount_config}"
        )
        
        return {
            "message": "Cost discount configuration updated successfully",
            "status": "success",
            "values": cost_discount_config
        }
    except Exception as e:
        verbose_proxy_logger.error(
            f"Error updating cost discount config: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to update cost discount config: {str(e)}"}
        )


@router.get(
    "/config/cost_margin_config",
    tags=["Cost Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_cost_margin_config(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get current cost margin configuration.
    
    Returns the cost_margin_config from litellm_settings.
    """
    from litellm.proxy.proxy_server import prisma_client, proxy_config
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    try:
        # Load config from DB
        config = await proxy_config.get_config()
        
        # Get cost_margin_config from litellm_settings
        litellm_settings = config.get("litellm_settings", {})
        cost_margin_config = litellm_settings.get("cost_margin_config", {})
        
        return {"values": cost_margin_config}
    except Exception as e:
        verbose_proxy_logger.error(
            f"Error fetching cost margin config: {str(e)}"
        )
        return {"values": {}}


@router.patch(
    "/config/cost_margin_config",
    tags=["Cost Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_cost_margin_config(
    cost_margin_config: Dict[str, Union[float, Dict[str, float]]],
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update cost margin configuration.
    
    Updates the cost_margin_config in litellm_settings.
    Margins can be:
    - Percentage: {"openai": 0.10} = 10% margin
    - Fixed amount: {"openai": {"fixed_amount": 0.001}} = $0.001 per request
    - Combined: {"vertex_ai": {"percentage": 0.08, "fixed_amount": 0.0005}}
    - Global: {"global": 0.05} = 5% global margin on all providers
    
    Example:
    ```json
    {
        "global": 0.05,
        "openai": 0.10,
        "anthropic": {"fixed_amount": 0.001},
        "vertex_ai": {"percentage": 0.08, "fixed_amount": 0.0005}
    }
    ```
    """
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_config,
        store_model_in_db,
    )
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    if store_model_in_db is not True:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."
            },
        )
    
    # Validate that all providers are valid LiteLLM providers (except "global")
    invalid_providers = []
    for provider in cost_margin_config.keys():
        if provider != "global" and provider not in LlmProvidersSet:
            invalid_providers.append(provider)
    
    if invalid_providers:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Invalid provider(s): {', '.join(invalid_providers)}. Must be valid LiteLLM providers or 'global'. See https://docs.litellm.ai/docs/providers for the full list."
            },
        )
    
    # Validate margin values
    for provider, margin_value in cost_margin_config.items():
        if isinstance(margin_value, (int, float)):
            # Simple percentage format: {"openai": 0.10}
            if not (0 <= margin_value <= 10):  # Allow up to 1000% margin
                raise HTTPException(
                    status_code=400,
                    detail=f"Margin percentage for {provider} must be between 0 and 10 (0% to 1000%)"
                )
        elif isinstance(margin_value, dict):
            # Complex format: {"percentage": 0.08, "fixed_amount": 0.0005}
            if "percentage" in margin_value:
                percentage = margin_value["percentage"]
                if not isinstance(percentage, (int, float)):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Margin percentage for {provider} must be a number"
                    )
                if not (0 <= percentage <= 10):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Margin percentage for {provider} must be between 0 and 10 (0% to 1000%)"
                    )
            if "fixed_amount" in margin_value:
                fixed_amount = margin_value["fixed_amount"]
                if not isinstance(fixed_amount, (int, float)):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Fixed margin amount for {provider} must be a number"
                    )
                if fixed_amount < 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Fixed margin amount for {provider} must be non-negative"
                    )
            if not margin_value:  # Empty dict
                raise HTTPException(
                    status_code=400,
                    detail=f"Margin config for {provider} cannot be empty. Must include 'percentage' and/or 'fixed_amount'"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Margin for {provider} must be a number (percentage) or dict with 'percentage' and/or 'fixed_amount'"
            )
    
    try:
        # Load existing config
        config = await proxy_config.get_config()
        
        # Ensure litellm_settings exists
        if "litellm_settings" not in config:
            config["litellm_settings"] = {}
        
        # Update cost_margin_config
        config["litellm_settings"]["cost_margin_config"] = cost_margin_config
        
        # Save the updated config to DB
        await proxy_config.save_config(new_config=config)
        
        # Update in-memory litellm.cost_margin_config
        litellm.cost_margin_config = cost_margin_config
        
        verbose_proxy_logger.info(
            f"Updated cost_margin_config: {cost_margin_config}"
        )
        
        return {
            "message": "Cost margin configuration updated successfully",
            "status": "success",
            "values": cost_margin_config
        }
    except Exception as e:
        verbose_proxy_logger.error(
            f"Error updating cost margin config: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to update cost margin config: {str(e)}"}
        )


@router.post(
    "/cost/estimate",
    tags=["Cost Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CostEstimateResponse,
)
async def estimate_cost(
    request: CostEstimateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> CostEstimateResponse:
    """
    Estimate cost for a given model and token counts.

    This endpoint uses the same cost calculation logic as actual requests,
    including any configured margins and discounts.

    Parameters:
    - model: Model name from /model_group/info (e.g., "gpt-4", "claude-3-opus")
    - input_tokens: Expected input tokens per request
    - output_tokens: Expected output tokens per request
    - num_requests: Number of requests (default: 1)

    Returns cost breakdown including:
    - Input token cost
    - Output token cost
    - Margin/fee cost
    - Total cost

    Example:
    ```json
    {
        "model": "gpt-4",
        "input_tokens": 1000,
        "output_tokens": 500,
        "num_requests": 100
    }
    ```
    """
    from litellm.cost_calculator import _apply_cost_margin
    from litellm.proxy.proxy_server import llm_router
    from litellm.types.utils import Usage

    if llm_router is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "Router not initialized. No models configured."},
        )

    # Get model group info from router to resolve pricing
    model_group_info = llm_router.get_model_group_info(model_group=request.model)

    if model_group_info is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Model '{request.model}' not found. Use /model_group/info to see available models."
            },
        )

    # Get the provider from the model group
    providers: List[str] = model_group_info.providers or []
    custom_llm_provider: Optional[str] = providers[0] if providers else None

    # Get cost per token from model group info
    input_cost_per_token = model_group_info.input_cost_per_token or 0.0
    output_cost_per_token = model_group_info.output_cost_per_token or 0.0

    # Calculate base costs (before margin)
    input_cost = input_cost_per_token * request.input_tokens
    output_cost = output_cost_per_token * request.output_tokens
    base_cost = input_cost + output_cost

    # Apply margin using the same function as completion_cost
    (
        cost_with_margin,
        _margin_percent,
        _margin_fixed_amount,
        margin_cost,
    ) = _apply_cost_margin(
        base_cost=base_cost,
        custom_llm_provider=custom_llm_provider,
    )

    cost_per_request = cost_with_margin

    # Calculate totals based on number of requests
    total_cost = cost_per_request * request.num_requests
    total_input_cost = input_cost * request.num_requests
    total_output_cost = output_cost * request.num_requests
    total_margin_cost = margin_cost * request.num_requests

    return CostEstimateResponse(
        model=request.model,
        input_tokens=request.input_tokens,
        output_tokens=request.output_tokens,
        num_requests=request.num_requests,
        cost_per_request=cost_per_request,
        input_cost_per_request=input_cost,
        output_cost_per_request=output_cost,
        margin_cost_per_request=margin_cost,
        total_cost=total_cost,
        total_input_cost=total_input_cost,
        total_output_cost=total_output_cost,
        total_margin_cost=total_margin_cost,
        input_cost_per_token=input_cost_per_token,
        output_cost_per_token=output_cost_per_token,
        provider=custom_llm_provider,
    )

