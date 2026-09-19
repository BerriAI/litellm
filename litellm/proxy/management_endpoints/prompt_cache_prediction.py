import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, JsonValue, TypeAdapter

import litellm
from litellm._internal_context import current_billing_time, pinned_billing_time
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.anthropic.prompt_cache_prediction import (
    PromptPrefix,
    TokenCounter,
    UnsupportedPredictionTarget,
    cache_scope,
    count_prompt_tokens,
    parse_prompt,
    resolve_prediction_target,
    supported_prediction_headers,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import can_key_call_resolved_model
from litellm.proxy.auth.auth_utils import get_cache_prediction_deployments
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import (
    _read_request_body,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # canonical parsed-body owner; validate its legacy result at the endpoint boundary
)
from litellm.proxy.common_utils.prompt_cache_pricing import price_cache_tokens
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,  # pyright: ignore[reportPrivateUsage]  # use the configured proxy limiter's shared capacity owner
)
from litellm.proxy.hooks.prompt_cache_prediction import lookup
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.types.management_endpoints.prompt_cache_prediction import (
    CacheCostScenario,
    CacheEvidence,
    CachePredictionArm,
    CachePredictionRequest,
    CachePredictionResponse,
    CacheTokenBuckets,
)
from litellm.types.router import Deployment
from litellm.utils import get_prompt_cache_min_tokens

router: Final = APIRouter()
_REQUEST_DATA: Final = TypeAdapter(Mapping[str, object])


class _CallerSettings(BaseModel):
    config: Mapping[str, object] | None = None


def has_request_transforms() -> bool:
    from litellm.proxy.hooks import PROXY_HOOKS

    builtins: Final = frozenset(PROXY_HOOKS.values())
    hooks: Final = ("async_pre_call_hook", "async_pre_request_hook", "async_pre_call_deployment_hook")
    callbacks: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(callback_type=CustomLogger)
    return any(
        type(callback) not in builtins
        and any(getattr(type(callback), hook) is not getattr(CustomLogger, hook) for hook in hooks)
        for callback in callbacks
    )


def _buckets(prefix_tokens: int, suffix_tokens: int, read_tokens: int, ttl_seconds: int) -> CacheTokenBuckets:
    return CacheTokenBuckets(
        uncached_input_tokens=suffix_tokens,
        cache_read_input_tokens=read_tokens,
        cache_creation_5m_input_tokens=prefix_tokens - read_tokens if ttl_seconds == 300 else 0,
        cache_creation_1h_input_tokens=prefix_tokens - read_tokens if ttl_seconds == 3600 else 0,
    )


def _scenario(model: str, deployment_id: str, tokens: CacheTokenBuckets) -> CacheCostScenario | None:
    cost: Final = price_cache_tokens(model=model, deployment_id=deployment_id, tokens=tokens)
    return CacheCostScenario(tokens=tokens, input_cost=cost) if cost is not None else None


def _capacity_counter(
    limiter: _PROXY_MaxParallelRequestsHandler_v3,
    caller: UserAPIKeyAuth,
    model_name: str,
    request_data: Mapping[str, object],
) -> TokenCounter:
    async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
        async with limiter.request_capacity(caller, model_name, request_data=request_data):
            return await count_prompt_tokens(model, api_key, body)

    return count


def _capacity_request_data(
    http_request: Request, caller: UserAPIKeyAuth, request_data: Mapping[str, object]
) -> Mapping[str, object]:
    # The parsed-body cache retains only original top-level keys. Replay the
    # shared idempotent tag merges on limiter-only data when auth added metadata.
    data: Final = dict(request_data)  # mutable-ok: the existing tag merge owners accept a dictionary out-param
    LiteLLMProxyRequestSetup.apply_client_tag_policy_pre_auth(http_request, data, caller)  # pyright: ignore[reportUnknownMemberType]  # legacy tag owner takes the validated capacity dictionary
    LiteLLMProxyRequestSetup.apply_key_tags_pre_auth(data, caller)  # pyright: ignore[reportUnknownMemberType]  # legacy tag owner merges trusted key tags into capacity metadata
    return MappingProxyType(data)


async def predict_arm(
    deployment: Deployment,
    body: Mapping[str, JsonValue],
    prefix: PromptPrefix,
    caller_key_hash: str,
    cache: DualCache,
    token_counter: TokenCounter,
) -> CachePredictionArm:
    deployment_id: Final = deployment.model_info.id or ""
    params: Final = deployment.litellm_params
    unknown: Final = CachePredictionArm(deployment_id=deployment_id, model=params.model)
    if deployment.model_info.blocked:
        return unknown.model_copy(update=MappingProxyType({"reason": "unsupported_deployment_configuration"}))
    target: Final = resolve_prediction_target(params)
    if isinstance(target, UnsupportedPredictionTarget):
        return unknown.model_copy(update=MappingProxyType({"reason": target.reason}))
    model: Final = target.model
    api_key: Final = target.api_key
    total_count: Final = await token_counter(model, api_key, body)
    prefix_count: Final = await token_counter(model, api_key, prefix.prefix_body)
    if total_count is None or prefix_count is None or total_count < prefix_count:
        return unknown.model_copy(update=MappingProxyType({"reason": "token_count_unavailable"}))
    scope: Final = cache_scope(caller_key_hash, deployment_id, api_key, model)
    observation: Final = await lookup(cache, scope, prefix)
    exact: Final = observation is not None and observation.fingerprint == prefix.fingerprint
    cacheable: Final = observation.cached_tokens if exact and observation is not None else prefix_count
    if cacheable > total_count or (observation is not None and observation.cached_tokens > cacheable):
        return unknown.model_copy(update=MappingProxyType({"reason": "inconsistent_prefix_token_count"}))
    suffix: Final = total_count - cacheable
    evidence: Final = (
        CacheEvidence(observed_at=observation.observed_at, expires_at=observation.expires_at)
        if observation is not None
        else None
    )
    if cacheable < get_prompt_cache_min_tokens(params.model):
        disabled: Final = _scenario(model, deployment_id, CacheTokenBuckets(uncached_input_tokens=total_count))
        if disabled is None:
            return unknown.model_copy(update=MappingProxyType({"reason": "pricing_unavailable"}))
        return CachePredictionArm(
            deployment_id=deployment_id,
            model=model,
            cache_state="disabled",
            reason="below_cache_minimum",
            estimate=disabled,
            cold=disabled,
            warm=disabled,
            token_count_source="anthropic_count_tokens",
        )
    fresh: Final = observation is not None and observation.expires_at > time.time()
    read: Final = observation.cached_tokens if fresh and observation is not None else 0
    with pinned_billing_time(current_billing_time()):
        cold: Final = _scenario(model, deployment_id, _buckets(cacheable, suffix, 0, prefix.ttl_seconds))
        warm: Final = _scenario(model, deployment_id, _buckets(cacheable, suffix, cacheable, prefix.ttl_seconds))
        estimate: Final = _scenario(model, deployment_id, _buckets(cacheable, suffix, read, prefix.ttl_seconds))
    if cold is None or warm is None or estimate is None:
        return unknown.model_copy(update=MappingProxyType({"reason": "pricing_unavailable"}))
    return CachePredictionArm(
        deployment_id=deployment_id,
        model=model,
        cache_state="warm" if fresh and exact else "partial" if fresh else "stale" if observation else "unknown",
        reason=None if fresh else "observation_expired" if observation else "no_compatible_observation",
        estimate=estimate,
        cold=cold,
        warm=warm,
        evidence=evidence,
        token_count_source="anthropic_count_tokens",
    )


@router.post(
    "/cost/predict-cache",
    tags=["Cost Tracking"],  # mutable-ok: FastAPI requires a list for OpenAPI tags
    response_model=CachePredictionResponse,
)
async def predict_cache_cost(
    request: CachePredictionRequest,
    http_request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> CachePredictionResponse:
    """Compare the next native Anthropic request on two configured deployment IDs.

    Estimates use provider token counting and recent successful cache telemetry for this key.
    Unknown cache state uses the cold scenario when prices/counts are available. Cache observations
    do not guarantee retention. v0 supports one message-content breakpoint, text and client tools;
    system/tool-only breakpoints, thinking, images, nondefault Anthropic versions, beta headers and
    request transforms are unknown.
    Each provider count consumes one RPM unit and holds concurrency capacity; a comparison uses
    up to four counts. The legacy rate limiter returns unknown without contacting the provider.
    This endpoint does not generate tokens, prewarm caches, choose a model or alter routing.
    """
    from litellm.proxy.proxy_server import llm_router, proxy_logging_obj

    if llm_router is None:
        raise HTTPException(status_code=503, detail="Model router is unavailable")
    deployments: Final = get_cache_prediction_deployments(
        current_deployment_id=request.current_deployment_id,
        candidate_deployment_id=request.candidate_deployment_id,
        llm_router=llm_router,
        team_id=user_api_key_dict.team_id,
    )
    if deployments is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    current, candidate = deployments
    for deployment in (current, candidate):
        await can_key_call_resolved_model(
            model=deployment.model_name,
            llm_model_list=llm_router.get_model_list(),
            valid_token=user_api_key_dict,
            llm_router=llm_router,
        )
    prefix: Final = parse_prompt(request.request)
    caller: Final = user_api_key_dict.api_key
    caller_settings: Final = _CallerSettings.model_validate(user_api_key_dict, from_attributes=True)
    unsupported_transform: Final = bool(caller_settings.config) or has_request_transforms()
    unsupported_headers: Final = not supported_prediction_headers(http_request.headers)
    limiter: Final = proxy_logging_obj.get_proxy_hook("parallel_request_limiter")
    if (
        prefix is None
        or not caller
        or unsupported_transform
        or unsupported_headers
        or not isinstance(limiter, _PROXY_MaxParallelRequestsHandler_v3)
    ):
        reason: Final = (
            "unsupported_provider_headers"
            if unsupported_headers
            else "unsupported_request_transform"
            if unsupported_transform
            else "unsupported_prompt_shape"
            if prefix is None
            else "caller_identity_unavailable"
            if not caller
            else "limiter_unavailable"
        )
        return CachePredictionResponse(
            stay=CachePredictionArm(deployment_id=request.current_deployment_id, reason=reason),
            switch=CachePredictionArm(deployment_id=request.candidate_deployment_id, reason=reason),
            switch_delta=None,
            cache_rebuild_penalty=None,
        )
    request_data: Final = _capacity_request_data(
        http_request, user_api_key_dict, _REQUEST_DATA.validate_python(await _read_request_body(http_request))
    )
    stay: Final = await predict_arm(
        current,
        request.request,
        prefix,
        caller,
        proxy_logging_obj.internal_usage_cache.dual_cache,
        _capacity_counter(limiter, user_api_key_dict, current.model_name, request_data),
    )
    switch: Final = (
        stay
        if current.model_info.id == candidate.model_info.id
        else await predict_arm(
            candidate,
            request.request,
            prefix,
            caller,
            proxy_logging_obj.internal_usage_cache.dual_cache,
            _capacity_counter(limiter, user_api_key_dict, candidate.model_name, request_data),
        )
    )
    return CachePredictionResponse(
        stay=stay,
        switch=switch,
        switch_delta=(switch.estimate.input_cost - stay.estimate.input_cost)
        if switch.estimate is not None and stay.estimate is not None
        else None,
        cache_rebuild_penalty=(switch.estimate.input_cost - switch.warm.input_cost)
        if switch.estimate is not None and switch.warm is not None
        else None,
    )
