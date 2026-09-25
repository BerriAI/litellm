from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, JsonValue, TypeAdapter

from litellm.llms.anthropic.prompt_cache_prediction import (
    TokenCounter,
    count_prompt_tokens,
    parse_prompt,
    supported_prediction_headers,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import can_key_call_resolved_model
from litellm.proxy.auth.auth_utils import get_cache_prediction_deployments
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import (
    _read_request_body,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # canonical parsed-body owner; validate its legacy result at the endpoint boundary
)
from litellm.proxy.common_utils.prompt_cache_prediction import has_request_transforms, predict_arm
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,  # pyright: ignore[reportPrivateUsage]  # use the configured proxy limiter's shared capacity owner
)
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.types.management_endpoints.prompt_cache_prediction import (
    CachePredictionArm,
    CachePredictionRequest,
    CachePredictionResponse,
)

router: Final = APIRouter()
_REQUEST_DATA: Final = TypeAdapter(Mapping[str, object])


class _CallerSettings(BaseModel):
    config: Mapping[str, object] | None = None


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
