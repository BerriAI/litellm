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
from litellm.proxy.spend_tracking.baseline_accounting import BaselineObservation
from litellm.proxy.spend_tracking.savings import (
    _effective_model_info,  # pyright: ignore[reportPrivateUsage]  # existing deployment-price owner
    _proxy_llm_router,  # pyright: ignore[reportPrivateUsage]  # existing optional proxy-router owner
)
from litellm.router_strategy.complexity_router.context_compaction import compaction_applied
from litellm.router_utils.baseline_request import baseline_request
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.router import BaselineRouteStamp
from litellm.types.utils import CallTypes, ModelInfo, Usage
from litellm.utils import get_prompt_cache_min_tokens

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
    prices: ModelInfo | None
    observation: BaselineObservation

    def with_observation(self, observation: BaselineObservation) -> CapturedBaselineObservation:
        return self.model_copy(update={"observation": observation})


@dataclass(frozen=True, slots=True)
class BaselineCacheContext:
    collector: AutoRouterBaselineCache
    capture: CapturedBaselineObservation
    target: NativePredictionTarget | UnsupportedPredictionTarget
    baseline_deployment_id: str
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


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _native_body_digest(body: Mapping[str, JsonValue]) -> str:
    return _digest({key: value for key, value in body.items() if key not in ("metadata", "stream")})


class AutoRouterBaselineCache(CustomLogger):
    def __init__(
        self,
        prisma_client: PrismaClient | None,
        router: Callable[[], Router | None] = _proxy_llm_router,
        token_counter: TokenCounter | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # legacy callback constructor
        self.router: Final = router
        self.token_counter: Final = token_counter
        self.clock: Final = clock
        self.count_slots: Final = asyncio.Semaphore(8)
        self.counts: Mapping[str, tuple[int, float]] = MappingProxyType({})

    async def async_pre_call_deployment_hook(self, kwargs: Mapping[str, object], call_type: CallTypes | None) -> None:
        from litellm.litellm_core_utils.litellm_logging import Logging

        logging_obj: Final = kwargs.get("litellm_logging_obj")
        if not isinstance(logging_obj, Logging) or call_type != CallTypes.anthropic_messages:
            return
        try:
            raw_metadata: Final = kwargs.get(get_metadata_variable_name_from_kwargs(kwargs))
            metadata: Final = _METADATA.validate_python(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
            if metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY):
                return
            if logging_obj.baseline_cache_context is not None:
                await invalidate_baseline_cache(logging_obj, "retried_request")
                return
            if not isinstance(metadata.get("_autorouter_baseline_route"), BaselineRouteStamp):
                return
            request: Final = _Metadata.model_validate(metadata)
            session: Final = kwargs.get("litellm_session_id") or request.session_id or logging_obj.litellm_session_id
            if not isinstance(session, str) or not session or len(session) > 256:
                return
            router: Final = self.router()
            deployment: Final = router.get_deployment(request.route.baseline_deployment_id) if router else None
            if deployment is None:
                return
            target: Final = resolve_baseline_prediction_target(deployment.litellm_params)
            prices: Final = _PRICES.validate_python(
                _effective_model_info(router, request.route.baseline_deployment_id, request.route.baseline_model)
            )
            params: Final = (
                _METADATA.validate_python(deployment.litellm_params.model_dump(mode="json")) if deployment else {}
            )
            projected: Final = (
                baseline_request(
                    kwargs,
                    request.route.request_parameters,
                    params,
                    include_extra_body=False,
                )
                if request.route.request_parameters is not None
                else None
            )
            scope: Final = "autorouter-baseline:v3:" + _digest(
                (
                    "baseline_request_v4",
                    request.user_api_key_hash,
                    session,
                    request.route.router_name,
                    request.route.baseline_deployment_id,
                    params,
                    prices,
                )
            )
            started: Final = logging_obj.start_time.timestamp()
            capture: Final = CapturedBaselineObservation(
                scope=scope,
                api_key=request.user_api_key_hash,
                session_id=session,
                router_name=request.route.router_name,
                baseline_model=request.route.baseline_model,
                model=target.model if isinstance(target, NativePredictionTarget) else request.route.baseline_model,
                prices=prices,
                observation=BaselineObservation(
                    request_id=logging_obj.litellm_call_id,
                    started_at=started,
                    available_at=started,
                    outcome="uncertain",
                    baseline_equivalent=False,
                    reason="incomplete_response",
                ),
            )
            selected_model: Final = kwargs.get("model")
            selected_body: Final = prepare_native_baseline_body(
                kwargs, selected_model if isinstance(selected_model, str) else logging_obj.model
            )
            logging_obj.baseline_cache_context = BaselineCacheContext(
                self,
                capture,
                target,
                request.route.baseline_deployment_id,
                prepare_native_baseline_body(projected, target.model)
                if projected is not None and isinstance(target, NativePredictionTarget)
                else None,
                _native_body_digest(selected_body)
                if selected_body is not None and not compaction_applied(kwargs)
                else None,
            )
        except Exception:  # noqa: BLE001  # optional observation cannot fail inference
            verbose_proxy_logger.warning("Auto-router baseline observation could not be initialized")

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
    task: Final = context.finalization or asyncio.create_task(_capture(context, logging_obj, response_obj))
    active: Final = context if context.finalization is not None else replace(context, finalization=task)
    if context.finalization is None:
        task.add_done_callback(_consume_finalization)
    logging_obj.baseline_cache_context = active
    try:
        capture: Final = await asyncio.shield(task)
        if logging_obj.baseline_cache_context is active:
            logging_obj.baseline_observation = capture  # rebind-ok: publish only for the current attempt
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
    return await _capture_native(context, logging_obj, response_obj)
