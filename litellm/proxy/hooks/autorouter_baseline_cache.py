from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import (
    get_litellm_metadata_from_kwargs,  # pyright: ignore[reportUnknownVariableType]  # legacy metadata boundary validated below
)
from litellm.llms.anthropic.prompt_cache_prediction import (
    CountedPromptCachePlan,
    NativePredictionTarget,
    TokenCounter,
    UnsupportedCachePlan,
    UnsupportedPredictionTarget,
    count_cache_plan,
    count_prompt_tokens,
    parse_cache_plan,
    resolve_baseline_prediction_target,
    supported_baseline_recipient,
    supported_prediction_headers,
)
from litellm.proxy.spend_tracking.baseline_accounting import BaselineObservation
from litellm.proxy.spend_tracking.savings import (
    _effective_model_info,  # pyright: ignore[reportPrivateUsage]  # existing deployment-price owner
    _proxy_llm_router,  # pyright: ignore[reportPrivateUsage]  # existing optional proxy-router owner
)
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


class CapturedBaselineObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    scope: str
    api_key: str
    session_id: str
    router_name: str
    baseline_model: str
    model: str
    prices: ModelInfo | None
    observation: BaselineObservation


@dataclass(frozen=True, slots=True)
class BaselineCacheContext:
    collector: AutoRouterBaselineCache
    capture: CapturedBaselineObservation
    target: NativePredictionTarget | UnsupportedPredictionTarget
    baseline_deployment_id: str
    invalidated: str | None = None


class _Metadata(BaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)
    route: BaselineRouteStamp = Field(alias="_autorouter_baseline_route")
    user_api_key_hash: str = Field(min_length=1)
    session_id: str | None = None


class _WireEvent(BaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)
    httpx_response: httpx.Response
    api_call_start_time: datetime
    completion_start_time: datetime
    custom_llm_provider: str
    stream: bool = False
    prompt_cache_response_complete: bool = False


class _ResponseUsage(BaseModel):
    model_config = ConfigDict(strict=True, from_attributes=True)
    usage: Usage | None = None


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
            metadata: Final = _METADATA.validate_python(
                get_litellm_metadata_from_kwargs(
                    {"litellm_params": kwargs}  # mutable-ok: legacy metadata owner requires a dictionary
                )
            )
            if metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY):
                return
            if logging_obj.baseline_cache_context is not None:
                await invalidate_baseline_cache(logging_obj, "retried_request")
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
            scope: Final = "autorouter-baseline:v3:" + _digest(
                (
                    request.user_api_key_hash,
                    session,
                    request.route.router_name,
                    request.route.baseline_deployment_id,
                    deployment.litellm_params.model_dump(mode="json"),
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
            logging_obj.baseline_cache_context = BaselineCacheContext(
                self, capture, target, request.route.baseline_deployment_id
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
        if not supported_prediction_headers(wire.headers):
            return None, "unsupported_request_headers"
        plan: Final = parse_cache_plan(body)
        if isinstance(plan, UnsupportedCachePlan):
            return None, plan.reason
        details: Final = usage.prompt_tokens_details if usage is not None else None
        if (
            not plan.breakpoints
            and details is not None
            and ((details.cached_tokens or 0) + (details.cache_creation_tokens or 0))
        ):
            return None, "implicit_cache_without_breakpoints"

        async def count(model: str, api_key: str, body: Mapping[str, JsonValue]) -> int | None:
            return await self._count(target, body)

        try:
            counted: Final = await asyncio.wait_for(
                count_cache_plan(target.model, target.api_key, plan, token_counter=count), timeout=_COUNT_TIMEOUT
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
            update=MappingProxyType(
                {
                    "observation": context.capture.observation.model_copy(
                        update=MappingProxyType(
                            {
                                "available_at": max(context.capture.observation.started_at, context.collector.clock()),
                                "reason": reason,
                            }
                        )
                    ),
                }
            )
        )


async def finalize_baseline_cache(logging_obj: Logging, response_obj: object) -> None:
    context: Final = logging_obj.baseline_cache_context
    if context is None:
        return
    try:
        capture: Final = await _capture(context, logging_obj, response_obj)
        if logging_obj.baseline_cache_context is context:
            logging_obj.baseline_observation = capture  # rebind-ok: attach only to the captured request owner
    except Exception:  # noqa: BLE001  # observation failures must preserve inference and billing
        await invalidate_baseline_cache(logging_obj, "observation_unavailable")


async def _capture(
    context: BaselineCacheContext, logging_obj: Logging, response_obj: object
) -> CapturedBaselineObservation:
    original: Final = context.capture.observation
    details: Final = _METADATA.validate_python(logging_obj.model_call_details)
    if details.get("cache_hit") is True:
        return context.capture.model_copy(
            update=MappingProxyType(
                {
                    "observation": original.model_copy(
                        update=MappingProxyType({"outcome": "response_cache", "reason": "response_cache_hit"})
                    )
                }
            )
        )
    event: Final = _WireEvent.model_validate(details)
    wire: Final = event.httpx_response.request
    usage: Final = _ResponseUsage.model_validate(response_obj).usage
    complete: Final = (
        event.custom_llm_provider == "anthropic"
        and event.httpx_response.status_code == 200
        and (not event.stream or event.prompt_cache_response_complete)
    )
    started: Final = original.started_at
    available: Final = event.completion_start_time.timestamp()
    if context.invalidated or not complete or not started <= available <= context.collector.clock():
        return context.capture.model_copy(
            update=MappingProxyType(
                {
                    "observation": original.model_copy(
                        update=MappingProxyType(
                            {
                                "available_at": max(started, context.collector.clock()),
                                "reason": context.invalidated or "incomplete_response",
                            }
                        )
                    )
                }
            )
        )
    target: Final = context.target
    if isinstance(target, UnsupportedPredictionTarget) or not supported_baseline_recipient(target, wire):
        return context.capture.model_copy(
            update=MappingProxyType(
                {
                    "observation": original.model_copy(
                        update=MappingProxyType(
                            {
                                "available_at": available,
                                "reason": target.reason
                                if isinstance(target, UnsupportedPredictionTarget)
                                else "unsupported_baseline_recipient",
                            }
                        )
                    )
                }
            )
        )
    body: Final = _JSON_BODY.validate_json(wire.content)
    same: Final = (
        logging_obj.get_router_model_id() == context.baseline_deployment_id and body.get("model") == target.model
    )
    plan, reason = await context.collector.plan(target, wire, body, usage)
    minimum: Final = get_prompt_cache_min_tokens(target.model)
    return context.capture.model_copy(
        update=MappingProxyType(
            {
                "observation": BaselineObservation(
                    request_id=original.request_id,
                    started_at=started,
                    available_at=available,
                    outcome="complete",
                    baseline_equivalent=same,
                    usage=usage,
                    plan=plan,
                    minimum_cache_tokens=minimum,
                    reason=reason,
                )
            }
        )
    )
