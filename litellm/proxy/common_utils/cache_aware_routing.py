from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs
from litellm.llms.anthropic.prompt_cache_prediction import (
    NativePredictionTarget,
    PromptPrefix,
    TokenCounter,
    cache_scope,
    count_prompt_tokens,
    parse_prompt,
    resolve_prediction_target,
    supported_prediction_headers,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import can_key_call_resolved_model
from litellm.proxy.common_utils.prompt_cache_prediction import predict_arm
from litellm.proxy.common_utils.prompt_cache_pricing import price_cache_tokens
from litellm.proxy.hooks.prompt_cache_prediction import lookup
from litellm.router_strategy.complexity_router.config import ComplexityRouterConfig
from litellm.types.router import Deployment, PreRoutingHookResponse
from litellm.types.utils import StandardLoggingRoutingDecision

if TYPE_CHECKING:
    from litellm.router import Router

_JSON: Final = TypeAdapter(Mapping[str, JsonValue])
_MESSAGES: Final = TypeAdapter(list[Mapping[str, object]])
_MAPPING: Final = TypeAdapter(Mapping[str, object])
_DEPLOYMENTS: Final[TypeAdapter[tuple[Deployment, ...] | Deployment]] = TypeAdapter(tuple[Deployment, ...] | Deployment)
_NATIVE_OPTIONS: Final = frozenset(
    (
        "max_tokens",
        "system",
        "tools",
        "tool_choice",
        "thinking",
        "output_config",
        "cache_control",
        "speed",
        "service_tier",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "stream",
    )
)
_MARKER_OPTIONS: Final = frozenset(
    ("model", "complexity_router_config", "rpm", "tpm", "tags", "timeout", "stream_timeout", "num_retries")
)
_CLASSIFIED_CAUSES: Final = frozenset(
    {
        "heuristic_scorer",
        "heuristic_v2",
        "reasoning_override",
        "llm_classifier",
        "llm_v2_classifier",
        "jev_classifier",
        "capability_classifier",
        "heuristic_first_short_circuit",
        "hybrid_short_circuit",
        "classifier_plugin",
    }
)


class _ProxyRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    url: str
    body: Mapping[str, JsonValue]
    headers: Mapping[str, str]


class _CallerSettings(BaseModel):
    config: Mapping[str, object] | None = None


class _ModelLimits(BaseModel):
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class CacheAwareChoice:
    model: str
    tier: str
    deployment_id: str
    original_cost: float
    estimated_cost: float


@dataclass(frozen=True, slots=True)
class _Candidate:
    model: str
    tier: str
    deployment: Deployment
    target: NativePredictionTarget


def eligible_models(
    config: ComplexityRouterConfig, decision: StandardLoggingRoutingDecision
) -> tuple[tuple[str, str], ...]:
    tier: Final = decision.get("tier")
    order: Final = config.tier_names()
    tier_entries: Final = chain.from_iterable(config.tier_model_configs.values())
    if (
        tier is None
        or tier not in order
        or decision.get("cause") not in _CLASSIFIED_CAUSES
        or config.has_custom_tiers
        or config.plugins
        or config.adaptive
        or config.session_affinity
        or config.classification_mode != "every_request"
        or any(entry.litellm_params for entry in tier_entries)
        or any(not isinstance(model, str) for model in config.tiers.values())
    ):
        return ()
    floor: Final = order.index(tier)
    eligible: Final = tuple(
        (name, model) for name, model in config.tiers.items() if isinstance(model, str) and name in order[floor:]
    )
    return tuple(entry for index, entry in enumerate(eligible) if entry[1] not in tuple(m for _, m in eligible[:index]))


async def _candidate(
    router: Router,
    tier: str,
    model: str,
    caller: UserAPIKeyAuth,
    request_kwargs: Mapping[str, object],
    messages: Sequence[Mapping[str, object]] | None,
) -> _Candidate | None:
    deployments: Final = router.deployments_for_request(model, request_kwargs)
    if len(deployments) != 1:
        return None
    deployment: Final = Deployment.model_validate(deployments[0])
    target: Final = resolve_prediction_target(deployment.litellm_params)
    if deployment.model_info.blocked or not deployment.model_info.id or not isinstance(target, NativePredictionTarget):
        return None
    try:
        await can_key_call_resolved_model(
            model=model, llm_model_list=router.get_model_list(), valid_token=caller, llm_router=router
        )
        healthy: Final = _DEPLOYMENTS.validate_python(
            await router.async_get_healthy_deployments(  # pyright: ignore[reportUnknownMemberType]  # legacy router results are validated at this boundary
                model=model,
                messages=_MESSAGES.validate_python(messages) if messages else None,  # pyright: ignore[reportArgumentType]  # router annotations predate structured native messages
                request_kwargs=dict(request_kwargs),  # mutable-ok: Router's filtering API accepts a request dictionary
            )
        )
    except Exception:  # noqa: BLE001  # an unavailable optional candidate must not fail the originally selected route
        return None
    available: Final = (healthy,) if isinstance(healthy, Deployment) else healthy
    return (
        _Candidate(model, tier, deployment, target)
        if any(entry.model_info.id == deployment.model_info.id for entry in available)
        else None
    )


async def _warm(candidate: _Candidate, caller: str, cache: DualCache, prefix: PromptPrefix, now: float) -> bool:
    scope: Final = cache_scope(
        caller, candidate.deployment.model_info.id or "", candidate.target.api_key, candidate.target.model
    )
    observation: Final = await lookup(cache, scope, prefix, now=now)
    return observation is not None and observation.expires_at > now


def _fits(candidate: _Candidate, input_tokens: int, output_tokens: int) -> bool:
    limits: Final = _ModelLimits.model_validate(
        MappingProxyType(
            {
                **litellm.get_model_info(candidate.target.model, custom_llm_provider="anthropic"),
                **candidate.deployment.model_info.model_dump(exclude_none=True),
            }
        )
    )
    return (
        limits.max_input_tokens is not None
        and input_tokens + output_tokens <= limits.max_input_tokens
        and limits.max_output_tokens is not None
        and output_tokens <= limits.max_output_tokens
    )


def supported_router_marker(router: Router, alias: str, request_kwargs: Mapping[str, object]) -> bool:
    markers: Final = tuple(
        Deployment.model_validate(entry) for entry in router.deployments_for_request(alias, request_kwargs)
    )
    return bool(markers) and all(
        marker.litellm_params.model == "auto_router/complexity_router"
        and not frozenset(marker.litellm_params.model_dump(exclude_defaults=True, exclude_none=True)) - _MARKER_OPTIONS
        for marker in markers
    )


async def select_cached_model(
    *,
    router: Router,
    config: ComplexityRouterConfig,
    response: PreRoutingHookResponse,
    body: Mapping[str, JsonValue],
    request_kwargs: Mapping[str, object],
    messages: Sequence[Mapping[str, object]] | None,
    caller: UserAPIKeyAuth,
    cache: DualCache,
    counter_for_model: Callable[[str], TokenCounter],
    now: float | None = None,
) -> CacheAwareChoice | None:
    checked_at: Final = time.time() if now is None else now
    decision: Final = response.routing_decision
    prefix: Final = parse_prompt(body)
    requested_limit: Final = body.get("max_tokens")
    if (
        not config.cache_aware_routing
        or decision is None
        or prefix is None
        or not caller.api_key
        or not isinstance(requested_limit, int)
        or isinstance(requested_limit, bool)
        or requested_limit <= 0
    ):
        return None
    names: Final = eligible_models(config, decision)
    if not names or response.model not in tuple(model for _, model in names):
        return None
    candidates: Final = tuple(
        candidate
        for candidate in await asyncio.gather(
            *(_candidate(router, tier, model, caller, request_kwargs, messages) for tier, model in names)
        )
        if candidate is not None
    )
    original: Final = next((candidate for candidate in candidates if candidate.model == response.model), None)
    if original is None:
        return None
    alternatives: Final = tuple(candidate for candidate in candidates if candidate.model != original.model)
    warm_flags: Final = await asyncio.gather(
        *(_warm(candidate, caller.api_key, cache, prefix, checked_at) for candidate in alternatives)
    )
    warm: Final = tuple(candidate for candidate, fresh in zip(alternatives, warm_flags) if fresh)
    if not warm:
        return None
    compared: Final = (original, *warm)
    arms: Final = await asyncio.gather(
        *(
            predict_arm(
                candidate.deployment,
                body,
                prefix,
                caller.api_key,
                cache,
                counter_for_model(candidate.model),
                now=now,
            )
            for candidate in compared
        )
    )
    output_tokens: Final = min(config.cache_aware_routing_output_tokens, requested_limit)
    costs: Final = tuple(
        price_cache_tokens(arm.model or "", arm.deployment_id, arm.estimate.tokens, output_tokens)
        if arm.estimate is not None
        else None
        for arm in arms
    )
    original_cost: Final = costs[0]
    if original_cost is None:
        return None
    finished_at: Final = time.time() if now is None else now
    qualifying: Final = tuple(
        CacheAwareChoice(candidate.model, candidate.tier, arm.deployment_id, original_cost, cost)
        for candidate, arm, cost in zip(warm, arms[1:], costs[1:])
        if cost is not None
        and cost < original_cost
        and arm.cache_state in ("warm", "partial")
        and arm.evidence is not None
        and arm.evidence.expires_at > finished_at
        and arm.estimate is not None
        and _fits(candidate, arm.estimate.tokens.total_tokens, requested_limit)
    )
    return min(qualifying, key=lambda choice: choice.estimated_cost, default=None)


async def _choose_cached_model(
    *,
    router: Router,
    config: ComplexityRouterConfig,
    response: PreRoutingHookResponse | None,
    request_kwargs: Mapping[str, object],
    messages: Sequence[Mapping[str, object]] | None,
) -> CacheAwareChoice | None:
    if not config.cache_aware_routing or response is None or response.routing_decision is None:
        return None
    if not eligible_models(config, response.routing_decision):
        return None
    from litellm.proxy import proxy_server
    from litellm.proxy.common_utils.prompt_cache_prediction import has_request_transforms
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        _PROXY_MaxParallelRequestsHandler_v3,  # pyright: ignore[reportPrivateUsage]  # use the configured proxy limiter's shared capacity owner
    )
    from litellm.router_strategy.complexity_router.context_compaction import compaction_pending

    if (
        proxy_server.llm_router is not router
        or router.routing_plugins
        or has_request_transforms()
        or compaction_pending(request_kwargs)
        or not supported_router_marker(router, response.routing_decision.get("router_model_name") or "", request_kwargs)
    ):
        return None
    metadata: Final = _MAPPING.validate_python(
        request_kwargs.get(get_metadata_variable_name_from_kwargs(request_kwargs)) or MappingProxyType({})
    )
    caller: Final = metadata.get("user_api_key_auth")
    if not isinstance(caller, UserAPIKeyAuth):
        return None
    settings: Final = _CallerSettings.model_validate(caller, from_attributes=True)
    if settings.config:
        return None
    try:
        incoming: Final = _ProxyRequest.model_validate(request_kwargs.get("proxy_server_request"))
    except ValidationError:
        return None
    if not urlparse(incoming.url).path.endswith("/v1/messages") or not supported_prediction_headers(incoming.headers):
        return None
    if any(
        request_kwargs.get(key)
        for key in (
            "guardrails",
            "cache_control_injection_points",
            "api_key",
            "api_base",
            "extra_headers",
            "prompt_id",
            "mock_response",
            "model_info",
            "custom_llm_provider",
        )
    ):
        return None
    body: Final = _JSON.validate_python(
        MappingProxyType(
            {
                **incoming.body,
                **MappingProxyType({key: request_kwargs[key] for key in _NATIVE_OPTIONS if key in request_kwargs}),
                "messages": messages,
            }
        )
    )
    limiter: Final = proxy_server.proxy_logging_obj.get_proxy_hook("parallel_request_limiter")
    if not isinstance(limiter, _PROXY_MaxParallelRequestsHandler_v3):
        return None

    def counter_for_model(model_name: str) -> TokenCounter:
        async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
            try:
                async with limiter.request_capacity(caller, model_name, request_data=request_kwargs):
                    return await count_prompt_tokens(model, api_key, body)
            except Exception:  # noqa: BLE001  # an optional prediction denied capacity is an unavailable estimate
                return None

        return count

    return await select_cached_model(
        router=router,
        config=config,
        response=response,
        body=body,
        request_kwargs=request_kwargs,
        messages=messages,
        caller=caller,
        cache=proxy_server.proxy_logging_obj.internal_usage_cache.dual_cache,
        counter_for_model=counter_for_model,
    )


async def choose_cached_model(
    *,
    router: Router,
    config: ComplexityRouterConfig,
    response: PreRoutingHookResponse | None,
    request_kwargs: Mapping[str, object],
    messages: Sequence[Mapping[str, object]] | None,
) -> CacheAwareChoice | None:
    if not config.cache_aware_routing:
        return None
    try:
        return await asyncio.wait_for(
            _choose_cached_model(
                router=router, config=config, response=response, request_kwargs=request_kwargs, messages=messages
            ),
            timeout=config.cache_aware_routing_timeout_ms / 1000,
        )
    except Exception:  # noqa: BLE001  # cache prediction is optional and must preserve normal routing on failure
        verbose_router_logger.debug("Cache-aware routing unavailable; keeping the classified model")
        return None
