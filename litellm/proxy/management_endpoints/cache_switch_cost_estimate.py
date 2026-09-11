import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Final

from fastapi import APIRouter, Depends, HTTPException

import litellm
from litellm._internal_context import current_billing_time
from litellm.litellm_core_utils.llm_cost_calc.utils import get_billed_token_rates
from litellm.litellm_core_utils.prompt_cache_prediction import (
    Prediction,
    PromptCacheTokenBudget,
    make_plan,
    predict,
)
from litellm.litellm_core_utils.token_counter import offload_token_count
from litellm.proxy._types import (
    CacheSwitchCostEstimateArmRequest,
    CacheSwitchCostEstimateArmResponse,
    CacheSwitchCostEstimateRequest,
    CacheSwitchCostEstimateResponse,
    CacheSwitchPredictionArm,
    CacheSwitchPredictionRequest,
    CacheSwitchPredictionResponse,
    CacheSwitchPredictionTarget,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import can_key_call_resolved_model
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.utils import CacheCreationTokenDetails, PromptTokensDetailsWrapper, Usage

router: Final = APIRouter()
_USER_API_KEY_AUTH_DEPENDENCY: Final = Depends(user_api_key_auth)


@dataclass(frozen=True, slots=True)
class _ResolvedModel:
    model: str
    model_id: str | None
    provider: str | None


def _resolved_direct_model(model: str) -> _ResolvedModel:
    try:
        resolved_model, provider, _, _ = litellm.get_llm_provider(model=model)
    except Exception as error:
        raise HTTPException(
            status_code=404,
            detail={  # mutable-ok: HTTPException detail is a public response mapping
                "error": f"Could not price model '{model}'"
            },
        ) from error
    return _ResolvedModel(model=resolved_model, model_id=None, provider=provider)


def _resolved_deployment(model: str, model_id: str | None) -> _ResolvedModel:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        if model_id is not None:
            raise HTTPException(
                status_code=422,
                detail={  # mutable-ok: HTTPException detail is a public response mapping
                    "error": f"Unknown model_id '{model_id}'"
                },
            )
        return _resolved_direct_model(model)
    deployments: Final = tuple(llm_router.get_model_list(model_name=model) or ())
    if model_id is None and len(deployments) > 1:
        raise HTTPException(
            status_code=422,
            detail={  # mutable-ok: HTTPException detail is a public response mapping
                "error": f"Model '{model}' resolves to multiple deployments; provide model_id"
            },
        )
    matches: Final = (
        deployments
        if model_id is None
        else tuple(
            deployment
            for deployment in deployments
            if (deployment.get("model_info") or {}).get("id")  # mutable-ok: router metadata mapping
            == model_id
        )
    )
    if model_id is not None and len(matches) != 1:
        raise HTTPException(
            status_code=422,
            detail={  # mutable-ok: HTTPException detail is a public response mapping
                "error": f"model_id '{model_id}' does not uniquely identify a deployment for model '{model}'"
            },
        )
    if not matches:
        return _resolved_direct_model(model)
    deployment: Final = matches[0]
    params: Final = (
        deployment.get("litellm_params") or {}  # mutable-ok: router params mapping
    )
    info: Final = deployment.get("model_info") or {}  # mutable-ok: deployment info default is a read-only local view
    backend: Final = info.get("base_model") or params.get("base_model") or params.get("model")
    if not isinstance(backend, str):
        raise HTTPException(
            status_code=422,
            detail={  # mutable-ok: HTTPException detail is a public response mapping
                "error": f"Could not resolve model '{model}'"
            },
        )
    resolved_model, provider, _, _ = litellm.get_llm_provider(
        model=backend,
        custom_llm_provider=params.get("custom_llm_provider"),
    )
    resolved_id: Final = info.get("id")
    return _ResolvedModel(
        model=resolved_model,
        model_id=resolved_id if isinstance(resolved_id, str) else None,
        provider=provider,
    )


def _usage(request: CacheSwitchCostEstimateArmRequest) -> Usage:
    creation: Final = request.cache_creation_input_tokens_5m + request.cache_creation_input_tokens_1h
    total: Final = request.uncached_input_tokens + request.cache_read_input_tokens + creation
    return Usage(
        prompt_tokens=total,
        completion_tokens=0,
        total_tokens=total,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=request.uncached_input_tokens,
            cached_tokens=request.cache_read_input_tokens,
            cache_creation_tokens=creation,
            cache_creation_token_details=CacheCreationTokenDetails(
                ephemeral_5m_input_tokens=request.cache_creation_input_tokens_5m,
                ephemeral_1h_input_tokens=request.cache_creation_input_tokens_1h,
            ),
        ),
    )


def _estimate_arm(
    request: CacheSwitchCostEstimateArmRequest, billing_time: datetime
) -> CacheSwitchCostEstimateArmResponse:
    from litellm.proxy.proxy_server import llm_router

    resolved: Final = _resolved_deployment(request.model, request.model_id)
    usage: Final = _usage(request)
    model_info: Final = (
        llm_router.get_deployment_model_info(resolved.model_id, resolved.model)
        if llm_router is not None and resolved.model_id is not None
        else None
    )
    rates: Final = get_billed_token_rates(
        model=resolved.model,
        custom_llm_provider=resolved.provider,
        usage=usage,
        current_time=billing_time,
        model_info=model_info,
    )
    if rates is None:
        raise HTTPException(
            status_code=404,
            detail={  # mutable-ok: HTTPException detail is a public response mapping
                "error": f"Could not price model '{request.model}'"
            },
        )
    effective_info: Final = model_info or litellm.get_model_info(
        model=resolved.model,
        custom_llm_provider=resolved.provider,
    )
    missing_rate: Final = next(
        (
            label
            for tokens, label, key in (
                (request.uncached_input_tokens, "input", "input_cost_per_token"),
                (request.cache_read_input_tokens, "cache read", "cache_read_input_token_cost"),
                (request.cache_creation_input_tokens_5m, "5-minute cache write", "cache_creation_input_token_cost"),
                (
                    request.cache_creation_input_tokens_1h,
                    "1-hour cache write",
                    "cache_creation_input_token_cost_above_1hr",
                ),
            )
            if tokens > 0 and effective_info.get(key) is None
        ),
        None,
    )
    if missing_rate is not None:
        raise HTTPException(
            status_code=422,
            detail={  # mutable-ok: HTTPException detail is a public response mapping
                "error": f"Model '{request.model}' has no {missing_rate} rate"
            },
        )
    uncached: Final = request.uncached_input_tokens * rates.input_cost_per_token
    cache_read: Final = request.cache_read_input_tokens * rates.cache_read_input_token_cost
    write_5m: Final = request.cache_creation_input_tokens_5m * rates.cache_creation_input_token_cost
    write_1h: Final = request.cache_creation_input_tokens_1h * rates.cache_creation_input_token_cost_above_1hr
    return CacheSwitchCostEstimateArmResponse(
        **request.model_dump(),
        resolved_model=resolved.model,
        resolved_model_id=resolved.model_id,
        provider=resolved.provider,
        input_tokens=usage.prompt_tokens,
        input_cost=uncached + cache_read + write_5m + write_1h,
        uncached_input_cost=uncached,
        cache_read_input_cost=cache_read,
        cache_creation_input_cost_5m=write_5m,
        cache_creation_input_cost_1h=write_1h,
        input_cost_per_token=rates.input_cost_per_token,
        cache_read_input_token_cost=rates.cache_read_input_token_cost,
        cache_creation_input_token_cost_5m=rates.cache_creation_input_token_cost,
        cache_creation_input_token_cost_1h=rates.cache_creation_input_token_cost_above_1hr,
    )


@router.post(
    "/cost/estimate/cache-switch",
    tags=("Cost Tracking",),
    dependencies=(Depends(user_api_key_auth),),
    response_model=CacheSwitchCostEstimateResponse,
)
async def estimate_cache_switch_cost(
    request: CacheSwitchCostEstimateRequest,
    user_api_key_dict: UserAPIKeyAuth = _USER_API_KEY_AUTH_DEPENDENCY,
) -> CacheSwitchCostEstimateResponse:
    billing_time: Final = current_billing_time()
    stay: Final = _estimate_arm(request.stay, billing_time)
    switch: Final = _estimate_arm(request.switch, billing_time)
    return CacheSwitchCostEstimateResponse(
        stay=stay,
        switch=switch,
        switch_cost_delta=switch.input_cost - stay.input_cost,
    )


def _prediction_prompt(target: CacheSwitchPredictionTarget, resolved: _ResolvedModel) -> Mapping[str, object]:
    return {**target.prompt, "model": resolved.model}  # mutable-ok: native JSON body passed to the serializer


def _prediction_arm(
    target: CacheSwitchPredictionTarget,
    resolved: _ResolvedModel,
    prediction: Prediction,
    cold: CacheSwitchCostEstimateArmResponse | None,
    warm: CacheSwitchCostEstimateArmResponse | None,
    estimated: CacheSwitchCostEstimateArmResponse | None,
) -> CacheSwitchPredictionArm:
    return CacheSwitchPredictionArm(
        model=resolved.model,
        model_id=resolved.model_id,
        cache_state=prediction.state,
        reason=prediction.reason,
        observed_at=prediction.as_of,
        expires_at=prediction.expires_at,
        estimated_input_cost=estimated.input_cost if estimated is not None else None,
        cold_input_cost=cold.input_cost if cold is not None else None,
        warm_input_cost=warm.input_cost if warm is not None else None,
        cache_penalty=(estimated.input_cost - warm.input_cost if estimated is not None and warm is not None else None),
        cache_read_input_tokens=prediction.cache_read_input_tokens,
        cache_creation_input_tokens_5m=prediction.cache_creation_input_tokens_5m,
        cache_creation_input_tokens_1h=prediction.cache_creation_input_tokens_1h,
        uncached_input_tokens=prediction.uncached_input_tokens,
    )


async def _predict_target(
    target: CacheSwitchPredictionTarget, scope: str | None, now: float
) -> CacheSwitchPredictionArm:
    from litellm.proxy.proxy_server import llm_router

    resolved: Final = _resolved_deployment(target.model, target.model_id)
    if llm_router is None or resolved.model_id is None or scope is None:
        return CacheSwitchPredictionArm(
            model=resolved.model,
            model_id=resolved.model_id,
            cache_state="unknown",
            reason="concrete_router_deployment_required",
        )
    plan: Final = await offload_token_count(make_plan)(_prediction_prompt(target, resolved))
    if plan is None:
        return CacheSwitchPredictionArm(
            model=resolved.model,
            model_id=resolved.model_id,
            cache_state="unknown",
            reason="unsupported_or_ambiguous_prompt_shape",
        )
    prediction: Final = await predict(llm_router.cache, scope, resolved.model_id, plan, now)
    predicted_cacheable: Final = (
        prediction.cache_read_input_tokens
        + prediction.cache_creation_input_tokens_5m
        + prediction.cache_creation_input_tokens_1h
    )
    full_cacheable: Final = (
        plan.final_boundary.estimated_tokens if prediction.state == "unknown" else predicted_cacheable
    )
    predicted_uncached: Final = prediction.uncached_input_tokens
    cold_budget: Final = PromptCacheTokenBudget(
        0,
        full_cacheable if plan.final_ttl_seconds == 300 else 0,
        full_cacheable if plan.final_ttl_seconds == 3600 else 0,
        predicted_uncached,
    )
    warm_budget: Final = PromptCacheTokenBudget(full_cacheable, 0, 0, predicted_uncached)
    cold_request: Final = CacheSwitchCostEstimateArmRequest(
        model=target.model,
        model_id=resolved.model_id,
        cache_read_input_tokens=cold_budget.cache_read_input_tokens,
        cache_creation_input_tokens_5m=cold_budget.cache_creation_input_tokens_5m,
        cache_creation_input_tokens_1h=cold_budget.cache_creation_input_tokens_1h,
        uncached_input_tokens=cold_budget.uncached_input_tokens,
        assumed_cache_state="cold",
    )
    warm_request: Final = CacheSwitchCostEstimateArmRequest(
        model=target.model,
        model_id=resolved.model_id,
        cache_read_input_tokens=warm_budget.cache_read_input_tokens,
        cache_creation_input_tokens_5m=warm_budget.cache_creation_input_tokens_5m,
        cache_creation_input_tokens_1h=warm_budget.cache_creation_input_tokens_1h,
        uncached_input_tokens=warm_budget.uncached_input_tokens,
        assumed_cache_state="warm",
    )
    predicted_request: Final = CacheSwitchCostEstimateArmRequest(
        model=target.model,
        model_id=resolved.model_id,
        cache_read_input_tokens=prediction.cache_read_input_tokens,
        cache_creation_input_tokens_5m=prediction.cache_creation_input_tokens_5m,
        cache_creation_input_tokens_1h=prediction.cache_creation_input_tokens_1h,
        uncached_input_tokens=prediction.uncached_input_tokens,
        assumed_cache_state=(
            "warm" if prediction.state in ("warm", "partial") else "stale" if prediction.state == "stale" else "unknown"
        ),
    )
    billing_time: Final = current_billing_time()
    cold: Final = _estimate_arm(cold_request, billing_time)
    warm: Final = _estimate_arm(warm_request, billing_time)
    estimated: Final = None if prediction.state == "unknown" else _estimate_arm(predicted_request, billing_time)
    return _prediction_arm(target, resolved, prediction, cold, warm, estimated)


@router.post(
    "/cost/estimate/cache-switch/predict",
    tags=("Cost Tracking",),
    dependencies=(Depends(user_api_key_auth),),
    response_model=CacheSwitchPredictionResponse,
)
async def predict_cache_switch_cost(
    request: CacheSwitchPredictionRequest,
    user_api_key_dict: UserAPIKeyAuth = _USER_API_KEY_AUTH_DEPENDENCY,
) -> CacheSwitchPredictionResponse:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        raise HTTPException(
            status_code=422,
            detail=MappingProxyType({"error": "A configured Router is required for prediction"}),
        )
    for target in (request.stay, request.switch):
        await can_key_call_resolved_model(
            model=target.model,
            llm_model_list=llm_router.get_model_list(model_name=target.model),
            valid_token=user_api_key_dict,
            llm_router=llm_router,
        )
    now: Final = current_billing_time().timestamp()
    scope: Final = user_api_key_dict.api_key
    stay, switch = await asyncio.gather(
        _predict_target(request.stay, scope, now),
        _predict_target(request.switch, scope, now),
    )
    delta: Final = (
        switch.estimated_input_cost - stay.estimated_input_cost
        if switch.estimated_input_cost is not None and stay.estimated_input_cost is not None
        else None
    )
    return CacheSwitchPredictionResponse(stay=stay, switch=switch, switch_cost_delta=delta)
