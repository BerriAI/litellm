"""
COST TRACKING SETTINGS MANAGEMENT

Endpoints for managing cost discount and margin configuration

GET /config/cost_discount_config - Get current cost discount configuration
PATCH /config/cost_discount_config - Update cost discount configuration
GET /config/cost_margin_config - Get current cost margin configuration
PATCH /config/cost_margin_config - Update cost margin configuration
"""

from typing import Dict, Union

from fastapi import APIRouter, Depends, HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
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

