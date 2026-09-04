"""
COST TRACKING SETTINGS MANAGEMENT

Endpoints for managing cost discount and margin configuration

GET /config/cost_discount_config - Get current cost discount configuration
PATCH /config/cost_discount_config - Update cost discount configuration
GET /config/cost_margin_config - Get current cost margin configuration
PATCH /config/cost_margin_config - Update cost margin configuration
POST /cost/estimate - Estimate cost for a given model and token counts
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

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
from litellm.types.utils import CostPerToken, LlmProvidersSet, ModelInfo

router: Final = APIRouter()


@dataclass(frozen=True, slots=True)
class ResolvedCostModel:
    model: str
    provider: str | None
    custom_cost_per_token: CostPerToken | None


def _configured_price(key: str, sources: tuple[Mapping[str, object], ...]) -> float | None:
    values: Final = (source.get(key) for source in sources)
    numeric: Final = (float(value) for value in values if isinstance(value, (int, float)))
    return next(numeric, None)


def _extract_custom_pricing(
    litellm_params: Mapping[str, object], model_info: Mapping[str, object]
) -> CostPerToken | None:
    """
    Pull per-token pricing configured on a deployment so on-prem / self-hosted
    models (absent from the public cost map) still estimate a real cost.
    Pricing may live on ``litellm_params`` or ``model_info``; ``litellm_params``
    wins, matching the router's cost-map registration precedence.
    """
    sources: Final = (litellm_params, model_info)
    input_price: Final = _configured_price("input_cost_per_token", sources)
    output_price: Final = _configured_price("output_cost_per_token", sources)

    if input_price is None and output_price is None:
        return None

    return CostPerToken(
        input_cost_per_token=input_price or 0.0,
        output_cost_per_token=output_price or 0.0,
    )


def _lookup_model_info(model: str) -> ModelInfo | None:
    try:
        return litellm.get_model_info(model=model)
    except Exception:
        return None


def _resolve_model_for_cost_lookup(model: str) -> ResolvedCostModel:
    """
    Resolve a model name (which may be a router alias/model_group) to the
    underlying litellm model name, provider, and any deployment-configured
    pricing used for cost lookup.

    Args:
        model: The model name from the request (could be a router alias like 'e-model-router'
               or an actual model name like 'azure_ai/gpt-4')
    """
    from litellm.proxy.proxy_server import llm_router

    # Try to resolve from router if available
    if llm_router is not None:
        try:
            # Get deployments for this model name (handles aliases, wildcards, etc.)
            deployments: Final = llm_router.get_model_list(model_name=model)

            if deployments and len(deployments) > 0:
                first_deployment: Final = deployments[0]
                litellm_params: Final = first_deployment.get("litellm_params", {})
                model_info: Final = first_deployment.get("model_info", {})
                custom_llm_provider: Final = litellm_params.get("custom_llm_provider")
                provider: Final = str(custom_llm_provider) if custom_llm_provider is not None else None
                custom_cost_per_token: Final = _extract_custom_pricing(litellm_params, model_info)

                # Check base_model first (needed for Azure custom deployment names)
                base_model: Final = model_info.get("base_model") or litellm_params.get("base_model")
                if base_model:
                    verbose_proxy_logger.debug("Resolved model '%s' to base_model '%s' from router", model, base_model)
                    return ResolvedCostModel(str(base_model), provider, custom_cost_per_token)

                resolved_model: Final = litellm_params.get("model")
                if resolved_model:
                    verbose_proxy_logger.debug("Resolved model '%s' to '%s' from router", model, resolved_model)
                    return ResolvedCostModel(str(resolved_model), provider, custom_cost_per_token)
        except Exception as e:
            verbose_proxy_logger.debug("Could not resolve model '%s' from router: %s", model, e)

    # Return original model if not resolved
    return ResolvedCostModel(model, None, None)


def _calculate_period_costs(num_requests, cost_per_request, input_cost, output_cost, margin_cost):
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
        config: Final = await proxy_config.get_config()

        # Get cost_discount_config from litellm_settings
        litellm_settings: Final = config.get("litellm_settings", {})
        cost_discount_config: Final = litellm_settings.get("cost_discount_config", {})

        return {"values": cost_discount_config}
    except Exception as e:
        verbose_proxy_logger.error("Error fetching cost discount config: %s", e)
        return {"values": {}}


@router.patch(
    "/config/cost_discount_config",
    tags=["Cost Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_cost_discount_config(
    cost_discount_config: dict[str, float],
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
            detail={"error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."},
        )

    # Validate that all providers are valid LiteLLM providers
    invalid_providers: Final = []
    for provider in cost_discount_config:
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
            raise HTTPException(status_code=400, detail=f"Discount for {provider} must be a number")
        if not (0 <= discount <= 1):
            raise HTTPException(
                status_code=400,
                detail=f"Discount for {provider} must be between 0 and 1 (0% to 100%)",
            )

    try:
        # Load existing config
        config: Final = await proxy_config.get_config()

        # Ensure litellm_settings exists
        if "litellm_settings" not in config:
            config["litellm_settings"] = {}

        # Update cost_discount_config
        config["litellm_settings"]["cost_discount_config"] = cost_discount_config

        # Save the updated config to DB
        await proxy_config.save_config(new_config=config)

        # Update in-memory litellm.cost_discount_config
        litellm.cost_discount_config = cost_discount_config

        verbose_proxy_logger.info("Updated cost_discount_config: %s", cost_discount_config)

        return {
            "message": "Cost discount configuration updated successfully",
            "status": "success",
            "values": cost_discount_config,
        }
    except Exception as e:
        verbose_proxy_logger.error("Error updating cost discount config: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to update cost discount config: {e}"},
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
        config: Final = await proxy_config.get_config()

        # Get cost_margin_config from litellm_settings
        litellm_settings: Final = config.get("litellm_settings", {})
        cost_margin_config: Final = litellm_settings.get("cost_margin_config", {})

        return {"values": cost_margin_config}
    except Exception as e:
        verbose_proxy_logger.error("Error fetching cost margin config: %s", e)
        return {"values": {}}


@router.patch(
    "/config/cost_margin_config",
    tags=["Cost Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_cost_margin_config(
    cost_margin_config: dict[str, float | dict[str, float]],
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
            detail={"error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."},
        )

    # Validate that all providers are valid LiteLLM providers (except "global")
    invalid_providers: Final = []
    for provider in cost_margin_config:
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
                    detail=f"Margin percentage for {provider} must be between 0 and 10 (0% to 1000%)",
                )
        elif isinstance(margin_value, dict):
            # Complex format: {"percentage": 0.08, "fixed_amount": 0.0005}
            if "percentage" in margin_value:
                percentage = margin_value["percentage"]
                if not isinstance(percentage, (int, float)):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Margin percentage for {provider} must be a number",
                    )
                if not (0 <= percentage <= 10):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Margin percentage for {provider} must be between 0 and 10 (0% to 1000%)",
                    )
            if "fixed_amount" in margin_value:
                fixed_amount = margin_value["fixed_amount"]
                if not isinstance(fixed_amount, (int, float)):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Fixed margin amount for {provider} must be a number",
                    )
                if fixed_amount < 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Fixed margin amount for {provider} must be non-negative",
                    )
            if not margin_value:  # Empty dict
                raise HTTPException(
                    status_code=400,
                    detail=f"Margin config for {provider} cannot be empty. Must include 'percentage' and/or 'fixed_amount'",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Margin for {provider} must be a number (percentage) or dict with 'percentage' and/or 'fixed_amount'",
            )

    try:
        # Load existing config
        config: Final = await proxy_config.get_config()

        # Ensure litellm_settings exists
        if "litellm_settings" not in config:
            config["litellm_settings"] = {}

        # Update cost_margin_config
        config["litellm_settings"]["cost_margin_config"] = cost_margin_config

        # Save the updated config to DB
        await proxy_config.save_config(new_config=config)

        # Update in-memory litellm.cost_margin_config
        litellm.cost_margin_config = cost_margin_config

        verbose_proxy_logger.info("Updated cost_margin_config: %s", cost_margin_config)

        return {
            "message": "Cost margin configuration updated successfully",
            "status": "success",
            "values": cost_margin_config,
        }
    except Exception as e:
        verbose_proxy_logger.error("Error updating cost margin config: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to update cost margin config: {e}"},
        )


class BlockUnpricedModelsRequest(BaseModel):
    enabled: bool


class BlockUnpricedModelsResponse(BaseModel):
    enabled: bool


@router.get(
    "/config/block_requests_for_models_without_pricing",
    tags=("Cost Tracking",),
    dependencies=(Depends(user_api_key_auth),),
    response_model=BlockUnpricedModelsResponse,
)
async def get_block_requests_for_models_without_pricing() -> BlockUnpricedModelsResponse:
    return BlockUnpricedModelsResponse(enabled=bool(litellm.block_requests_for_models_without_pricing))


@router.patch(
    "/config/block_requests_for_models_without_pricing",
    tags=("Cost Tracking",),
    dependencies=(Depends(user_api_key_auth),),
    response_model=BlockUnpricedModelsResponse,
)
async def update_block_requests_for_models_without_pricing(
    request: BlockUnpricedModelsRequest,
) -> BlockUnpricedModelsResponse:
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_config,
        store_model_in_db,
    )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={  # mutable-ok: HTTPException detail must be a plain mapping
                "error": CommonProxyErrors.db_not_connected_error.value
            },
        )

    if store_model_in_db is not True:
        raise HTTPException(
            status_code=500,
            detail={  # mutable-ok: HTTPException detail must be a plain mapping
                "error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."
            },
        )

    try:
        config = await proxy_config.get_config()
        if "litellm_settings" not in config:
            config["litellm_settings"] = {}  # mutable-ok: config is a plain-dict payload for save_config
        config["litellm_settings"]["block_requests_for_models_without_pricing"] = request.enabled
        await proxy_config.save_config(new_config=config)

        litellm.block_requests_for_models_without_pricing = request.enabled
        verbose_proxy_logger.info("Updated block_requests_for_models_without_pricing: %s", request.enabled)

        return BlockUnpricedModelsResponse(enabled=request.enabled)
    except Exception as e:  # noqa: BLE001  # any config persistence failure must surface as a 500 response, not a crash
        verbose_proxy_logger.error("Error updating block_requests_for_models_without_pricing: %s", e)
        raise HTTPException(
            status_code=500,
            detail={  # mutable-ok: HTTPException detail must be a plain mapping
                "error": f"Failed to update setting: {e!s}"
            },
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
    resolved: Final = _resolve_model_for_cost_lookup(request.model)
    resolved_model: Final = resolved.model
    resolved_provider: Final = resolved.provider

    verbose_proxy_logger.debug("Cost estimate: request.model='%s' resolved to '%s'", request.model, resolved_model)

    # Create a mock response with usage for completion_cost
    mock_response: Final = ModelResponse(
        model=resolved_model,
        usage=Usage(
            prompt_tokens=request.input_tokens,
            completion_tokens=request.output_tokens,
            total_tokens=request.input_tokens + request.output_tokens,
        ),
    )

    # Create a logging object to capture cost breakdown
    litellm_logging_obj: Final = LiteLLMLoggingObj(
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
        cost_per_request: Final = completion_cost(
            completion_response=mock_response,
            model=resolved_model,
            custom_llm_provider=resolved_provider,
            custom_cost_per_token=resolved.custom_cost_per_token,
            litellm_logging_obj=litellm_logging_obj,
        )
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Could not calculate cost for model '{request.model}' (resolved to '{resolved_model}'): {e}"
            },
        )

    # Get cost breakdown from the logging object
    cost_breakdown: Final = litellm_logging_obj.cost_breakdown

    input_cost: Final = cost_breakdown.get("input_cost", 0.0) if cost_breakdown else 0.0
    output_cost: Final = cost_breakdown.get("output_cost", 0.0) if cost_breakdown else 0.0
    margin_cost: Final = cost_breakdown.get("margin_total_amount", 0.0) if cost_breakdown else 0.0

    model_info: Final = _lookup_model_info(resolved_model)
    mapped_input_price: Final = model_info.get("input_cost_per_token") if model_info is not None else None
    mapped_output_price: Final = model_info.get("output_cost_per_token") if model_info is not None else None
    mapped_provider: Final = model_info.get("litellm_provider") if model_info is not None else None

    input_cost_per_token: Final = (
        resolved.custom_cost_per_token["input_cost_per_token"]
        if resolved.custom_cost_per_token is not None
        else mapped_input_price
    )
    output_cost_per_token: Final = (
        resolved.custom_cost_per_token["output_cost_per_token"]
        if resolved.custom_cost_per_token is not None
        else mapped_output_price
    )
    custom_llm_provider: Final = mapped_provider if mapped_provider is not None else resolved_provider

    # Calculate daily and monthly costs
    (
        daily_cost,
        daily_input_cost,
        daily_output_cost,
        daily_margin_cost,
    ) = _calculate_period_costs(
        num_requests=request.num_requests_per_day,
        cost_per_request=cost_per_request,
        input_cost=input_cost,
        output_cost=output_cost,
        margin_cost=margin_cost,
    )
    (
        monthly_cost,
        monthly_input_cost,
        monthly_output_cost,
        monthly_margin_cost,
    ) = _calculate_period_costs(
        num_requests=request.num_requests_per_month,
        cost_per_request=cost_per_request,
        input_cost=input_cost,
        output_cost=output_cost,
        margin_cost=margin_cost,
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
