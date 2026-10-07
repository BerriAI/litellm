from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
import pydantic
from pydantic import ConfigDict, Field, JsonValue, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs
from litellm.litellm_core_utils.logging_worker import optional_callback_budget
from litellm.llms.anthropic.prompt_cache_prediction import (
    CountedPromptCachePlan,
    NativePredictionTarget,
    TokenCounter,
    UnsupportedCachePlan,
    UnsupportedPredictionTarget,
    count_cache_plan,
    count_prompt_tokens,
    parse_cache_plan,
    prepare_native_baseline_body,
    resolve_baseline_prediction_target,
    supported_baseline_recipient,
    supported_prediction_headers,
)
from litellm.llms.prompt_cache_estimation import (
    EstimatedCachePlan,
    count_prefix_tokens,
    estimate_cache_plan,
    normalize_cache_usage,
    prepare_baseline_usage,
    prepare_cache_request,
)
from litellm.proxy.spend_tracking.baseline_accounting import BaselineObservation
from litellm.proxy.spend_tracking.savings import (
    _cost_of_usage,  # pyright: ignore[reportPrivateUsage]  # shared token-pricing owner
    _effective_model_info,  # pyright: ignore[reportPrivateUsage]  # existing deployment-price owner
    _model_info,  # pyright: ignore[reportPrivateUsage]  # shared public-price lookup owner
    _pricing_basis,  # pyright: ignore[reportPrivateUsage]  # shared billing-tier owner
    _proxy_llm_router,  # pyright: ignore[reportPrivateUsage]  # existing optional proxy-router owner
    _resolve_model,  # pyright: ignore[reportPrivateUsage]  # shared model identity resolver
)
from litellm.router_strategy.complexity_router.context_compaction import compaction_applied
from litellm.router_utils.baseline_request import baseline_request
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.router import BaselineRouteStamp
from litellm.types.utils import CallTypes, LlmProviders, ModelInfo, Usage
from litellm.utils import ProviderConfigManager, get_prompt_cache_min_tokens

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router

_METADATA: Final = TypeAdapter(Mapping[str, object])
_PRICES: Final[TypeAdapter[ModelInfo | None]] = TypeAdapter(ModelInfo | None)
_JSON_BODY: Final = TypeAdapter(dict[str, JsonValue])
_COUNT_TIMEOUT: Final = 3.0
_MAX_COUNTS: Final = 4096


class CapturedBaselineObservation(LiteLLMBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scope: str
    api_key: str
    session_id: str
    router_name: str
    baseline_model: str
    model: str
    provider: str = "anthropic"
    actual_token_cost: float | None = None
    prices: ModelInfo | None
    observation: BaselineObservation

    def with_observation(self, observation: BaselineObservation) -> CapturedBaselineObservation:
        return self.model_copy(update={"observation": observation})


@dataclass(frozen=True, slots=True)
class BaselineCacheContext:
    collector: AutoRouterBaselineCache
    capture: CapturedBaselineObservation
    target: NativePredictionTarget | UnsupportedPredictionTarget
    baseline_deployment_id: str | None
    estimated_request: dict[str, JsonValue] | None = field(default=None, repr=False)
    estimated: bool = False
    baseline_body: Mapping[str, JsonValue] | None = field(default=None, repr=False)
    selected_body_digest: str | None = field(default=None, repr=False)
    invalidated: str | None = None
    finalization: asyncio.Task[CapturedBaselineObservation] | None = field(default=None, repr=False, compare=False)


class _Metadata(LiteLLMBaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)
    route: BaselineRouteStamp = Field(alias="_autorouter_baseline_route")
    user_api_key_hash: str = Field(min_length=1)
    session_id: str | None = None


class _WireEvent(LiteLLMBaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)
    httpx_response: httpx.Response
    api_call_start_time: datetime
    completion_start_time: datetime
    custom_llm_provider: str
    stream: bool = False
    prompt_cache_response_complete: bool = False


class _ResponseUsage(LiteLLMBaseModel):
    model_config = ConfigDict(strict=True, from_attributes=True)
    usage: Usage | None = None


class _UsageContainer(pydantic.BaseModel):
    model_config = ConfigDict(strict=True, from_attributes=True)
    usage: object | None = None


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _native_body_digest(body: Mapping[str, JsonValue]) -> str:
    return _digest({key: value for key, value in body.items() if key not in ("metadata", "stream")})


def _uses_messages_adapter(model: str, provider: str, model_info: Mapping[str, object]) -> bool:
    from litellm.llms.anthropic.pass_through.messages.handler import (
        _deployment_passes_through_anthropic_messages,  # pyright: ignore[reportPrivateUsage]  # reuse the native dispatch opt-in owner
    )

    provider_id: Final = next((candidate for candidate in LlmProviders if candidate.value == provider), None)
    return not _deployment_passes_through_anthropic_messages(dict(model_info)) and (
        provider_id is None or ProviderConfigManager.get_provider_anthropic_messages_config(model, provider_id) is None
    )


class AutoRouterBaselineCache(CustomLogger):
    def __init__(
        self,
        prisma_client: PrismaClient | None,
        router: Callable[[], Router | None] = _proxy_llm_router,
        token_counter: TokenCounter | None = None,
        clock: Callable[[], float] = time.time,
        prefix_token_counter: Callable[[str, str], int] = count_prefix_tokens,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # legacy callback constructor
        self.router: Final = router
        self.token_counter: Final = token_counter
        self.clock: Final = clock
        self.prefix_token_counter: Final = prefix_token_counter
        self.count_slots: Final = asyncio.Semaphore(8)
        self.estimate_slots: Final = asyncio.Semaphore(8)
        self.estimate_workers: Final = asyncio.Semaphore(2)
        self.counts: Mapping[str, tuple[int, float]] = MappingProxyType({})

    async def async_pre_call_deployment_hook(self, kwargs: Mapping[str, object], call_type: CallTypes | None) -> None:
        from litellm.litellm_core_utils.litellm_logging import Logging

        logging_obj: Final = kwargs.get("litellm_logging_obj")
        if (
            not isinstance(logging_obj, Logging)
            or call_type is None
            or call_type
            not in (
                CallTypes.anthropic_messages,
                CallTypes.completion,
                CallTypes.acompletion,
                CallTypes.responses,
                CallTypes.aresponses,
            )
        ):
            return
        try:
            raw_metadata: Final = kwargs.get(get_metadata_variable_name_from_kwargs(kwargs))
            metadata: Final = _METADATA.validate_python(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
            if metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY):
                return
            if logging_obj.baseline_cache_context is not None:
                if call_type.value != logging_obj.call_type:
                    return
                await invalidate_baseline_cache(logging_obj, "retried_request")
                return
            if not isinstance(metadata.get("_autorouter_baseline_route"), BaselineRouteStamp):
                return
            request: Final = _Metadata.model_validate(metadata)
            session: Final = kwargs.get("litellm_session_id") or request.session_id or logging_obj.litellm_session_id
            if not isinstance(session, str) or not session or len(session) > 256:
                return
            router: Final = self.router()
            deployment: Final = (
                router.get_deployment(request.route.baseline_deployment_id)
                if router and request.route.baseline_deployment_id
                else None
            )
            if request.route.baseline_deployment_id and deployment is None:
                return
            identity: Final = _resolve_model(
                deployment.litellm_params.model if deployment else request.route.baseline_model,
                deployment.litellm_params.custom_llm_provider if deployment else None,
            )
            if identity is None:
                return
            target: Final = (
                resolve_baseline_prediction_target(deployment.litellm_params)
                if deployment
                else UnsupportedPredictionTarget("direct_model_baseline")
            )
            prices: Final = _PRICES.validate_python(
                _effective_model_info(router, request.route.baseline_deployment_id, request.route.baseline_model)
                or _model_info(identity)
            )
            params: Final = (
                _METADATA.validate_python(deployment.litellm_params.model_dump(mode="json")) if deployment else {}
            )
            selected_model: Final = kwargs.get("model")
            selected_provider: Final = kwargs.get("custom_llm_provider")
            selected: Final = _resolve_model(
                selected_model if isinstance(selected_model, str) else logging_obj.model,
                selected_provider if isinstance(selected_provider, str) else None,
            )
            estimated: Final = (
                call_type != CallTypes.anthropic_messages
                or not isinstance(target, NativePredictionTarget)
                or selected is None
                or selected.provider != "anthropic"
            )
            native: Final = call_type == CallTypes.anthropic_messages and not _uses_messages_adapter(
                identity.model,
                identity.provider,
                _METADATA.validate_python(deployment.model_info.model_dump()) if deployment else {},
            )
            projected: Final = (
                baseline_request(
                    kwargs,
                    request.route.request_parameters,
                    params,
                    include_extra_body=not native,
                )
                if request.route.request_parameters is not None
                else None
            )
            estimated_request: Final = (
                prepare_cache_request(projected, identity.model, identity.provider, native=native)
                if estimated and projected is not None
                else None
            )
            scope: Final = "autorouter-baseline:v3:" + _digest(
                (
                    "baseline_request_v3" if estimated else "baseline_request_v4",
                    request.user_api_key_hash,
                    session,
                    request.route.router_name,
                    request.route.baseline_deployment_id,
                    params,
                    prices,
                    *((identity, "estimated_prefixes_v5") if estimated else ()),
                )
            )
            started: Final = logging_obj.start_time.timestamp()
            capture: Final = CapturedBaselineObservation(
                scope=scope,
                api_key=request.user_api_key_hash,
                session_id=session,
                router_name=request.route.router_name,
                baseline_model=request.route.baseline_model,
                model=identity.model,
                provider=identity.provider,
                prices=prices,
                observation=BaselineObservation(
                    request_id=logging_obj.litellm_call_id,
                    started_at=started,
                    available_at=started,
                    outcome="uncertain",
                    baseline_equivalent=False,
                    reason="incomplete_response",
                    cache_policy="estimated" if estimated else "anthropic",
                    assumptions=(),
                    cache_write_pricing="standard"
                    if estimated and (prices is None or prices.get("cache_creation_input_token_cost_above_1hr") is None)
                    else "duration",
                ),
            )
            selected_body: Final = (
                prepare_native_baseline_body(
                    kwargs, selected_model if isinstance(selected_model, str) else logging_obj.model
                )
                if not estimated
                else None
            )
            logging_obj.baseline_cache_context = BaselineCacheContext(
                self,
                capture,
                target,
                request.route.baseline_deployment_id,
                estimated_request,
                estimated,
                prepare_native_baseline_body(projected, target.model)
                if not estimated and projected is not None and isinstance(target, NativePredictionTarget)
                else None,
                _native_body_digest(selected_body)
                if selected_body is not None and not compaction_applied(kwargs)
                else None,
            )
        except Exception:  # noqa: BLE001  # optional observation cannot fail inference
            verbose_proxy_logger.warning("Auto-router baseline observation could not be initialized")

    async def estimate(self, context: BaselineCacheContext, usage: Usage) -> EstimatedCachePlan | None:
        if self.estimate_slots.locked() or context.estimated_request is None:
            return None
        await self.estimate_slots.acquire()

        async def count() -> EstimatedCachePlan | None:
            async with self.estimate_workers:
                return await asyncio.to_thread(
                    estimate_cache_plan,
                    context.estimated_request,
                    context.capture.model,
                    context.capture.provider,
                    context.capture.prices,
                    usage,
                    self.prefix_token_counter,
                )

        task: Final = asyncio.create_task(count())
        task.add_done_callback(self._release_estimate_slot)
        return await asyncio.shield(task)

    def _release_estimate_slot(self, task: asyncio.Task[EstimatedCachePlan | None]) -> None:
        self.estimate_slots.release()
        if not task.cancelled():
            task.exception()

    async def _count(self, target: NativePredictionTarget, body: Mapping[str, JsonValue]) -> int | None:
        key: Final = _digest((target.model, target.api_key, target.api_base, _JSON_BODY.validate_python(body)))
        now: Final = self.clock()
        cached: Final = self.counts.get(key)
        if cached is not None and cached[1] > now:
            return cached[0]
        async with self.count_slots:
            tokens: Final = (
                await self.token_counter(target.model, target.api_key, body)
                if self.token_counter is not None
                else await count_prompt_tokens(target.model, target.api_key, body, api_base=target.api_base)
            )
        if tokens is None or tokens < 0:
            return None
        retained: Final = tuple((k, v) for k, v in self.counts.items() if v[1] > now and k != key)[-(_MAX_COUNTS - 1) :]
        self.counts = MappingProxyType(dict((*retained, (key, (tokens, now + 3600)))))
        return tokens

    async def plan(
        self, target: NativePredictionTarget, wire: httpx.Request, body: Mapping[str, JsonValue], usage: Usage | None
    ) -> tuple[CountedPromptCachePlan | None, str | None]:
        deadline: Final = asyncio.get_running_loop().time() + optional_callback_budget(_COUNT_TIMEOUT, fraction=0.75)
        if not supported_prediction_headers(wire.headers):
            return None, "unsupported_request_headers"
        plan: Final = parse_cache_plan(body)
        if isinstance(plan, UnsupportedCachePlan):
            return None, plan.reason
        details: Final = usage.prompt_tokens_details if usage is not None else None
        selected: Final = parse_cache_plan(_JSON_BODY.validate_json(wire.content))
        if (
            (isinstance(selected, UnsupportedCachePlan) or not selected.breakpoints)
            and details is not None
            and ((details.cached_tokens or 0) + (details.cache_creation_tokens or 0))
        ):
            return None, "implicit_cache_without_breakpoints"

        async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
            return await self._count(target, body)

        try:
            counted: Final = await asyncio.wait_for(
                count_cache_plan(target.model, target.api_key, plan, token_counter=count),
                timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
            )
            return (None, counted.reason) if isinstance(counted, UnsupportedCachePlan) else (counted, None)
        except TimeoutError:
            return None, "token_count_timeout"
        except Exception:  # noqa: BLE001  # token counting cannot fail a completed request
            return None, "token_count_unavailable"


async def invalidate_baseline_cache(logging_obj: Logging, reason: str, *, completed: bool = False) -> None:
    context: Final = logging_obj.baseline_cache_context
    if context is not None:
        logging_obj.baseline_cache_context = replace(context, invalidated=reason)
        logging_obj.baseline_observation = context.capture.model_copy(
            update={
                "observation": context.capture.observation.model_copy(
                    update={
                        "available_at": max(context.capture.observation.started_at, context.collector.clock()),
                        "reason": reason,
                    }
                ),
            }
        )


async def finalize_baseline_cache(logging_obj: Logging, response_obj: object) -> None:
    context: Final = logging_obj.baseline_cache_context
    if context is None or logging_obj.baseline_observation is not None:
        return
    budget: Final = optional_callback_budget(_COUNT_TIMEOUT)
    task: Final = context.finalization or asyncio.create_task(_capture(context, logging_obj, response_obj))
    active: Final = context if context.finalization is not None else replace(context, finalization=task)
    if context.finalization is None:
        task.add_done_callback(_consume_finalization)
    logging_obj.baseline_cache_context = active
    try:
        capture: Final = (
            await asyncio.wait_for(asyncio.shield(task), timeout=budget)
            if context.estimated
            else await asyncio.shield(task)
        )
        if logging_obj.baseline_cache_context is active:
            logging_obj.baseline_observation = capture  # rebind-ok: publish only for the current attempt
    except TimeoutError:
        await invalidate_baseline_cache(logging_obj, "baseline_estimation_timeout")
    except Exception:  # noqa: BLE001  # estimation must preserve inference and billing
        await invalidate_baseline_cache(logging_obj, "observation_unavailable")


def _consume_finalization(task: asyncio.Task[CapturedBaselineObservation]) -> None:
    if not task.cancelled():
        task.exception()


async def _capture_native(
    context: BaselineCacheContext, logging_obj: Logging, response_obj: object
) -> CapturedBaselineObservation:
    capture: Final = context.capture
    original: Final = capture.observation
    event: Final = _WireEvent.model_validate(logging_obj.model_call_details)
    wire: Final = event.httpx_response.request
    usage: Final = _ResponseUsage.model_validate(response_obj).usage
    available: Final = event.completion_start_time.timestamp()
    complete: Final = (
        event.custom_llm_provider == "anthropic"
        and event.httpx_response.status_code == 200
        and (not event.stream or event.prompt_cache_response_complete)
    )
    if context.invalidated or not complete or not original.started_at <= available <= context.collector.clock():
        return capture.with_observation(
            original.model_copy(
                update={
                    "available_at": max(original.started_at, context.collector.clock()),
                    "reason": context.invalidated or "incomplete_response",
                }
            )
        )
    target: Final = context.target
    if isinstance(target, UnsupportedPredictionTarget) or not supported_baseline_recipient(target, wire):
        return capture.with_observation(
            original.model_copy(
                update={
                    "available_at": available,
                    "reason": target.reason
                    if isinstance(target, UnsupportedPredictionTarget)
                    else "unsupported_baseline_recipient",
                }
            )
        )
    body: Final = _JSON_BODY.validate_json(wire.content)
    projected: Final = context.baseline_body
    if projected is None or context.selected_body_digest != _native_body_digest(body):
        return capture.with_observation(
            original.model_copy(
                update={
                    "available_at": available,
                    "usage": usage,
                    "reason": "unsupported_baseline_settings"
                    if projected is None
                    else "unsupported_request_transformation",
                }
            )
        )
    same: Final = logging_obj.get_router_model_id() == context.baseline_deployment_id and _native_body_digest(
        projected
    ) == _native_body_digest(body)
    plan, reason = await context.collector.plan(target, wire, projected, usage)
    return capture.with_observation(
        BaselineObservation(
            request_id=original.request_id,
            started_at=original.started_at,
            available_at=available,
            outcome="complete",
            baseline_equivalent=same,
            usage=usage.model_copy(update={key: projected.get(key) for key in ("speed", "inference_geo")})
            if usage is not None and not same
            else usage,
            plan=plan,
            reason=reason,
            minimum_cache_tokens=get_prompt_cache_min_tokens(target.model),
        )
    )


async def _capture(
    context: BaselineCacheContext, logging_obj: Logging, response_obj: object
) -> CapturedBaselineObservation:
    if _METADATA.validate_python(logging_obj.model_call_details).get("cache_hit") is True:
        return context.capture.model_copy(
            update={
                "observation": context.capture.observation.model_copy(
                    update={
                        "outcome": "response_cache",
                        "reason": "response_cache_hit",
                    }
                ),
            }
        )
    return await (
        _capture_estimated(context, logging_obj, response_obj)
        if context.estimated
        else _capture_native(context, logging_obj, response_obj)
    )


async def _capture_estimated(
    context: BaselineCacheContext, logging_obj: Logging, response_obj: object
) -> CapturedBaselineObservation:
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    original: Final = context.capture.observation
    available: Final = max(
        original.started_at,
        logging_obj.completion_start_time.timestamp()
        if logging_obj.completion_start_time is not None
        else context.collector.clock(),
    )
    raw_usage: Final = _UsageContainer.model_validate(response_obj).usage
    serialized: Final = raw_usage.model_dump() if isinstance(raw_usage, pydantic.BaseModel) else raw_usage
    usage: Final = (
        normalize_cache_usage(
            StandardLoggingPayloadSetup.get_usage_from_response_obj(
                {"usage": serialized}, combined_usage_object=raw_usage if isinstance(raw_usage, Usage) else None
            )
        )
        if raw_usage is not None
        else None
    )
    estimated: Final = (
        await context.collector.estimate(context, usage)
        if context.estimated_request is not None and usage is not None and not context.invalidated
        else None
    )
    plan: Final = estimated.plan if estimated is not None else None
    provider: Final = _METADATA.validate_python(logging_obj.model_call_details).get("custom_llm_provider")
    selected: Final = _resolve_model(logging_obj.model, provider if isinstance(provider, str) else None)
    prices: Final = _PRICES.validate_python(logging_obj.get_router_deployment_model_info()) or (
        _model_info(selected) if selected else None
    )
    token_cost: Final = (
        _cost_of_usage(selected, usage, prices, _pricing_basis(logging_obj.cost_breakdown))
        if selected and usage and prices
        else None
    )
    return context.capture.model_copy(
        update={
            "actual_token_cost": token_cost,
            "observation": original.model_copy(
                update={
                    "available_at": available,
                    "outcome": "uncertain" if context.invalidated or usage is None else "complete",
                    "usage": prepare_baseline_usage(usage, context.capture.provider, context.estimated_request),
                    "plan": plan,
                    "minimum_cache_tokens": get_prompt_cache_min_tokens(
                        f"{context.capture.provider}/{context.capture.model}"
                    ),
                    "assumptions": estimated.assumptions if estimated else (),
                    "reason": context.invalidated
                    or ("missing_usage" if usage is None else "unsupported_cache_request" if plan is None else None),
                }
            ),
        }
    )
