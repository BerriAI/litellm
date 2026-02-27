"""
COST TRACKING SETTINGS MANAGEMENT

Endpoints for managing cost discount and margin configuration

GET /config/cost_discount_config - Get current cost discount configuration
PATCH /config/cost_discount_config - Update cost discount configuration
GET /config/cost_margin_config - Get current cost margin configuration
PATCH /config/cost_margin_config - Update cost margin configuration
POST /cost/estimate - Estimate cost for a given model and token counts
"""

from typing import Dict, Optional, Tuple, Union

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


def _resolve_model_for_cost_lookup(model: str) -> Tuple[str, Optional[str]]:
    """
    Resolve a model name (which may be a router alias/model_group) to the
    underlying litellm model name for cost lookup.

    Args:
        model: The model name from the request (could be a router alias like 'e-model-router'
               or an actual model name like 'azure_ai/gpt-4')

    Returns:
        Tuple of (resolved_model_name, custom_llm_provider)
        - resolved_model_name: The actual model name to use for cost lookup
        - custom_llm_provider: The provider if resolved from router, None otherwise
    """
    from litellm.proxy.proxy_server import llm_router

    custom_llm_provider: Optional[str] = None

    # Try to resolve from router if available
    if llm_router is not None:
        try:
            # Get deployments for this model name (handles aliases, wildcards, etc.)
            deployments = llm_router.get_model_list(model_name=model)

            if deployments and len(deployments) > 0:
                # Get the first deployment's litellm model
                first_deployment = deployments[0]
                litellm_params = first_deployment.get("litellm_params", {})
                resolved_model = litellm_params.get("model")

                if resolved_model:
                    verbose_proxy_logger.debug(
                        f"Resolved model '{model}' to '{resolved_model}' from router"
                    )
                    # Extract custom_llm_provider if present
                    custom_llm_provider = litellm_params.get("custom_llm_provider")
                    return resolved_model, custom_llm_provider
        except Exception as e:
            verbose_proxy_logger.debug(
                f"Could not resolve model '{model}' from router: {e}"
            )

    # Return original model if not resolved
    return model, custom_llm_provider


def _calculate_period_costs(
    num_requests, cost_per_request, input_cost, output_cost, margin_cost
):
    """
    Calculate costs for a given number of requests.

    Returns tuple of (total_cost, input_cost, output_cost, margin_cost) or all None if num_requests is None/0.
    """
    if not num_requests:
        return None, None, None, None
    return (
        cost_per_request * num_requests,
        input_cost * num_requests,
        output_cost * num_requests,
        margin_cost * num_requests,
    )


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
    - model: Model name (e.g., "gpt-4", "claude-3-opus")
    - input_tokens: Expected input tokens per request
    - output_tokens: Expected output tokens per request
    - num_requests_per_day: Number of requests per day (optional)
    - num_requests_per_month: Number of requests per month (optional)

    Returns cost breakdown including:
    - Per-request costs (input, output, margin)
    - Daily costs (if num_requests_per_day provided)
    - Monthly costs (if num_requests_per_month provided)

    Example:
    ```json
    {
        "model": "gpt-4",
        "input_tokens": 1000,
        "output_tokens": 500,
        "num_requests_per_day": 100,
        "num_requests_per_month": 3000
    }
    ```
    """
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import ModelResponse, Usage

    # Resolve model name (handles router aliases like 'e-model-router' -> 'azure_ai/gpt-4')
    resolved_model, resolved_provider = _resolve_model_for_cost_lookup(request.model)

    verbose_proxy_logger.debug(
        f"Cost estimate: request.model='{request.model}' resolved to '{resolved_model}'"
    )

    # Create a mock response with usage for completion_cost
    mock_response = ModelResponse(
        model=resolved_model,
        usage=Usage(
            prompt_tokens=request.input_tokens,
            completion_tokens=request.output_tokens,
            total_tokens=request.input_tokens + request.output_tokens,
        ),
    )

    # Create a logging object to capture cost breakdown
    litellm_logging_obj = LiteLLMLoggingObj(
        model=resolved_model,
        messages=[],
        stream=False,
        call_type="completion",
        start_time=None,
        litellm_call_id="cost-estimate",
        function_id="cost-estimate",
    )

    # Use completion_cost which handles all the logic including margins/discounts
    try:
        cost_per_request = completion_cost(
            completion_response=mock_response,
            model=resolved_model,
            litellm_logging_obj=litellm_logging_obj,
        )
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Could not calculate cost for model '{request.model}' (resolved to '{resolved_model}'): {str(e)}"
            },
        )

    # Get cost breakdown from the logging object
    cost_breakdown = litellm_logging_obj.cost_breakdown

    input_cost = cost_breakdown.get("input_cost", 0.0) if cost_breakdown else 0.0
    output_cost = cost_breakdown.get("output_cost", 0.0) if cost_breakdown else 0.0
    margin_cost = cost_breakdown.get("margin_total_amount", 0.0) if cost_breakdown else 0.0

    # Get model info for per-token pricing display
    try:
        model_info = litellm.get_model_info(model=resolved_model)
        input_cost_per_token = model_info.get("input_cost_per_token")
        output_cost_per_token = model_info.get("output_cost_per_token")
        custom_llm_provider = model_info.get("litellm_provider")
    except Exception:
        input_cost_per_token = None
        output_cost_per_token = None
        custom_llm_provider = None

    # Use provider from router resolution if not found in model_info
    if custom_llm_provider is None and resolved_provider is not None:
        custom_llm_provider = resolved_provider

    # Calculate daily and monthly costs
    daily_cost, daily_input_cost, daily_output_cost, daily_margin_cost = (
        _calculate_period_costs(
            num_requests=request.num_requests_per_day,
            cost_per_request=cost_per_request,
            input_cost=input_cost,
            output_cost=output_cost,
            margin_cost=margin_cost,
        )
    )
    monthly_cost, monthly_input_cost, monthly_output_cost, monthly_margin_cost = (
        _calculate_period_costs(
            num_requests=request.num_requests_per_month,
            cost_per_request=cost_per_request,
            input_cost=input_cost,
            output_cost=output_cost,
            margin_cost=margin_cost,
        )
    )

    return CostEstimateResponse(
        model=request.model,
        input_tokens=request.input_tokens,
        output_tokens=request.output_tokens,
        num_requests_per_day=request.num_requests_per_day,
        num_requests_per_month=request.num_requests_per_month,
        cost_per_request=cost_per_request,
        input_cost_per_request=input_cost,
        output_cost_per_request=output_cost,
        margin_cost_per_request=margin_cost,
        daily_cost=daily_cost,
        daily_input_cost=daily_input_cost,
        daily_output_cost=daily_output_cost,
        daily_margin_cost=daily_margin_cost,
        monthly_cost=monthly_cost,
        monthly_input_cost=monthly_input_cost,
        monthly_output_cost=monthly_output_cost,
        monthly_margin_cost=monthly_margin_cost,
        input_cost_per_token=input_cost_per_token,
        output_cost_per_token=output_cost_per_token,
        provider=custom_llm_provider,
    )

