from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache, RedisCircuitBreakerOpenError
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY, LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.proxy.health_check import resolve_health_check_mode
from litellm.router_strategy.auto_router.auto_router_model_naming import classify_strategy_router_model

if TYPE_CHECKING:
    from litellm.router import Router

# These are completed inference operations. Submitting a video or batch job does not establish that it succeeded.
_PROBE_MODES: Final = MappingProxyType(
    {
        "completion": "chat",
        "text_completion": "completion",
        "embedding": "embedding",
        "responses": "responses",
        "anthropic_messages": "anthropic_messages",
        "speech": "audio_speech",
        "transcription": "audio_transcription",
        "image_generation": "image_generation",
        "image_edit": "image_edit",
        "rerank": "rerank",
        "ocr": "ocr",
        "extraction": "extraction",
    }
)
_SOURCE: Final = "recent_traffic"


def is_traffic_observation(endpoint: object) -> bool:
    fields: Final = as_str_mapping(endpoint)
    return fields is not None and fields.get("health_check_source") == _SOURCE


@dataclass(frozen=True, slots=True)
class TrafficSnapshot:
    successes: Mapping[str, float]
    latest_failure: float
    cutoff: float

    def accepts(self, endpoints: Sequence[Mapping[str, object]], checked_at: float) -> bool:
        return checked_at >= self.latest_failure and all(
            not is_traffic_observation(endpoint) or self._current(endpoint) for endpoint in endpoints
        )

    def _current(self, endpoint: Mapping[str, object]) -> bool:
        ident: Final = endpoint.get("model_id")
        seen: Final = endpoint.get("last_success_at")
        return (
            isinstance(ident, str)
            and isinstance(seen, (int, float))
            and self.cutoff < seen <= self.successes.get(ident, 0.0)
        )


class _LocalTimes(Protocol):
    def get_cache(self, key: str) -> object: ...
    def set_cache(self, key: str, value: int, *, ttl: int) -> None: ...


def _probe_identity(deployment: Mapping[str, object]) -> tuple[str, str] | None:
    info: Final = as_str_mapping(deployment.get("model_info")) or {}
    params: Final = as_str_mapping(deployment.get("litellm_params")) or {}
    ident: Final = info.get("id")
    model: Final = params.get("model")
    if not isinstance(ident, str) or not ident:
        return None
    if not isinstance(model, str) or "*" in model:
        return None
    if classify_strategy_router_model(model) is not None:
        return None
    if info.get("health_check_model") or info.get("health_check_params") or params.get("mock_response") is not None:
        return None
    mode: Final = resolve_health_check_mode(info, params)
    return (ident, mode) if isinstance(mode, str) and mode in _PROBE_MODES.values() else None


def _event_identity(kwargs: Mapping[str, object], response: object, success: bool) -> tuple[str, str] | None:
    event: Final = as_str_mapping(kwargs.get("standard_logging_object")) or {}
    ident: Final = event.get("model_id")
    call_type: Final = event.get("call_type")
    if not isinstance(ident, str) or not ident or not isinstance(call_type, str):
        return None
    mode: Final = _PROBE_MODES.get(call_type) or _PROBE_MODES.get(call_type.removeprefix("a"))
    if mode is None or event.get("status") != ("success" if success else "failure"):
        return None
    params: Final = as_str_mapping(kwargs.get("litellm_params")) or {}
    optional: Final = as_str_mapping(kwargs.get("optional_params")) or {}
    metadata: Final = as_str_mapping(event.get("metadata")) or {}
    tags: Final = event.get("request_tags")
    if (
        not isinstance(kwargs.get("current_attempt_api_call_start_time"), datetime)
        or kwargs.get("cache_hit") is True
        or event.get("cache_hit") is True
    ):
        return None
    if params.get("mock_response") is not None or optional.get("mock_response") is not None:
        return None
    probe_tag: Final = isinstance(tags, (list, tuple)) and LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME in tags
    if (
        metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY)
        or metadata.get("user_api_key_team_id") == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
        or probe_tag
    ):
        return None
    if success:
        hidden: Final = as_str_mapping(getattr(response, "_hidden_params", None)) or {}
        wire: Final = kwargs.get("httpx_response")
        if isinstance(wire, httpx.Response):
            if not wire.is_success:
                return None
        elif not (as_str_mapping(kwargs.get("response_headers")) or as_str_mapping(hidden.get("headers"))):
            return None
        if kwargs.get("stream") is True and kwargs.get("complete_streaming_response") is None:
            return None
        response_fields: Final = as_str_mapping(response)
        status: Final = (
            response_fields.get("status") if response_fields is not None else getattr(response, "status", None)
        )
        if isinstance(status, str) and status not in ("completed", "succeeded", "success"):
            return None
    return ident, mode


def _key(identity: tuple[str, str], success: bool) -> str:
    ident, mode = identity
    outcome: Final = "success" if success else "failure"
    return f"health_traffic:{outcome}:{mode}:{ident}"


class RecentTrafficHealth(CustomLogger):
    def __init__(
        self, window: int, redis: Callable[[], RedisCache | None], clock: Callable[[], float] = time.time
    ) -> None:
        super().__init__()
        self.window = window
        self.redis = redis
        self.clock = clock
        self.local: _LocalTimes = InMemoryCache(max_size_in_memory=10000, default_ttl=window, clock=clock)

    def _local_time(self, key: str) -> int:
        value: Final[object] = self.local.get_cache(key)
        return value if isinstance(value, int) else 0

    async def _record(self, kwargs: Mapping[str, object], response: object, end_time: datetime, success: bool) -> None:
        identity: Final = _event_identity(kwargs, response, success)
        if identity is None:
            return
        finished: Final = end_time.timestamp()
        now: Final = self.clock()
        if not now - self.window < finished <= now:
            return
        stamp: Final = int(finished * 1000)
        key: Final = _key(identity, success)
        previous: Final = self._local_time(key)
        if stamp <= previous:
            return
        redis: Final = self.redis()
        recently_published: Final = (
            stamp - previous < min(1000, self.window * 250) and self._local_time(_key(identity, False)) < previous
        )
        if redis is not None and success and recently_published:
            return
        self.local.set_cache(key, stamp, ttl=self.window)
        if redis is not None:
            from redis.exceptions import RedisError

            try:
                await redis.async_set_max(key, stamp, ttl=self.window)
            except (RedisError, RedisCircuitBreakerOpenError, OSError, RuntimeError, ValueError):
                verbose_proxy_logger.debug("Could not publish recent health traffic", exc_info=True)

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await self._record(kwargs, response_obj, end_time, True)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await self._record(kwargs, response_obj, end_time, False)

    async def _read_times(self, keys: Sequence[str]) -> tuple[int, ...]:
        redis: Final = self.redis()
        if redis is None or not keys:
            return tuple(self._local_time(key) for key in keys)
        from redis.exceptions import RedisError

        try:
            shared: Final = await redis.async_batch_get_counts(list(keys))
            # A worker that could not publish a failure still probes locally.
            return tuple(
                max(stamp or 0, self._local_time(key) if key.startswith("health_traffic:failure:") else 0)
                for key, stamp in zip(keys, shared)
            )
        except (RedisError, RedisCircuitBreakerOpenError, OSError, RuntimeError, ValueError):
            verbose_proxy_logger.debug("Could not read recent health traffic", exc_info=True)
            return ()

    async def snapshot(
        self, deployments: Sequence[Mapping[str, object]], router: Router | None = None
    ) -> TrafficSnapshot:
        identities: Final = tuple(identity for d in deployments if (identity := _probe_identity(d)) is not None)
        keys: Final = list(chain.from_iterable((_key(i, True), _key(i, False)) for i in identities))
        values: Final = await self._read_times(keys)
        now: Final = self.clock()
        cutoff: Final = now - self.window
        successes: Final = {
            identity[0]: success / 1000
            for identity, success, failure in zip(identities, values[::2], values[1::2])
            if success > max(cutoff * 1000, failure) and success <= now * 1000
        }
        failed_probes: Final = (
            await router.health().health_checks.async_get_unhealthy_deployment_ids()
            if successes and router is not None
            else frozenset[str]()
        )
        eligible: Final = (
            {ident: seen for ident, seen in successes.items() if ident not in failed_probes}
            if failed_probes
            else successes
        )
        return TrafficSnapshot(eligible, max(values[1::2], default=0) / 1000, cutoff)
