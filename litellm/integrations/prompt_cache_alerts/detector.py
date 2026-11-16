"""Opens one prompt cache alert per deployment and configuration when a prefix the provider proved cacheable gets
three consecutive requests that neither read nor wrote the cache.

Streaks live in Redis and every update is one Lua script. Events are ordered by when their request started and
ended, which a replayed callback repeats exactly: a proof, hit or unknown moves the prefix's epoch to the end of
the request that caused it, and only a miss that started after the epoch counts. The proof expires with the cache
entry it proved, measured from the request's start.
"""

import hashlib
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from redis.exceptions import RedisError

from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.token_counter import offload_token_count
from litellm.models.alert import AlertDetection, AlertStore, PromptCacheEvidence

from .evidence import (
    CacheablePrefix,
    NotCaching,
    Unsupported,
    cache_counters,
    cacheable_prefix,
    fingerprint_key,
    keyed_digest,
)

if TYPE_CHECKING:
    from litellm.caching.redis_cache import RedisCache

PROMPT_CACHE_ALERTS_SETTING: Final = "prompt_cache_alerts"
MISSES_TO_ALERT: Final = 3
TTL_MS: Final = 300_000
_BREAK_TTL_MS: Final = 2 * TTL_MS
_CALL_TYPES: Final = frozenset(("completion", "acompletion"))
_STORE_FAILURES: Final = (RedisError, RedisCircuitBreakerOpenError, OSError, TimeoutError)

STREAK_SCRIPT: Final = """
local event = ARGV[1]
local started = tonumber(ARGV[2])
local ended = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])
local function later(a, b)
  if a > b then return a end
  return b
end
local broken = tonumber(redis.call('GET', KEYS[1]) or '0')
if event == 'break' then
  redis.call('SET', KEYS[1], later(broken, ended), 'PX', ARGV[9])
  return {'unproven'}
end
local proven_until = tonumber(redis.call('HGET', KEYS[2], 'until') or '0')
local epoch = later(tonumber(redis.call('HGET', KEYS[2], 'epoch') or '0'), broken)
if event == 'proof' then
  if started + ttl <= now then return {'unproven'} end
  local live = later(proven_until, started + ttl)
  local fresh = later(epoch, ended)
  redis.call('HSET', KEYS[2], 'tokens', ARGV[7], 'until', live, 'epoch', fresh)
  redis.call('PEXPIRE', KEYS[2], live - now)
  redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', fresh)
  return {'unproven'}
end
if proven_until <= now then
  redis.call('DEL', KEYS[2], KEYS[3])
  return {'unproven'}
end
if event ~= 'miss' then
  local fresh = later(epoch, ended)
  if event == 'hit' then
    proven_until = later(proven_until, started + ttl)
    redis.call('HSET', KEYS[2], 'until', proven_until)
    redis.call('PEXPIRE', KEYS[2], proven_until - now)
  end
  redis.call('HSET', KEYS[2], 'epoch', fresh)
  redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', fresh)
  return {'unproven'}
end
if started <= epoch then return {'unproven'} end
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', epoch)
redis.call('ZADD', KEYS[3], 'NX', started, ARGV[6])
redis.call('PEXPIRE', KEYS[3], proven_until - now)
local misses = redis.call('ZRANGE', KEYS[3], 0, -1, 'WITHSCORES')
local count = #misses / 2
if count < tonumber(ARGV[8]) then return {'counting', count} end
redis.call('HSET', KEYS[2], 'epoch', misses[#misses])
redis.call('DEL', KEYS[3])
return {'triggered', count, redis.call('HGET', KEYS[2], 'tokens'), misses[2]}
"""


class StreakEvent(StrEnum):
    PROOF = "proof"
    HIT = "hit"
    MISS = "miss"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Scope:
    deployment_id: str
    config_fingerprint: str

    def slot(self) -> str:
        digest: Final = hashlib.sha256(f"{self.deployment_id}\0{self.config_fingerprint}".encode()).hexdigest()
        return f"prompt_cache_alert:{{{digest}}}"


@dataclass(frozen=True, slots=True)
class PrefixEvent:
    scope: Scope
    prefix_fingerprint: str
    event: StreakEvent
    call_id: str
    started_ms: int
    ended_ms: int
    write_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ScopeBreak:
    """A caching request whose prefix could not be compared may have used any of the scope's prefixes."""

    scope: Scope
    ended_ms: int


StreakUpdate: TypeAlias = PrefixEvent | ScopeBreak


@dataclass(frozen=True, slots=True)
class Triggered:
    misses: int
    proven_write_tokens: int
    first_miss_ms: int


class StreakStore(Protocol):
    async def apply(self, update: StreakUpdate, now_ms: int) -> Triggered | None: ...


RunScript: TypeAlias = Callable[[Sequence[str], Sequence[str | int]], Awaitable[object]]
_REPLY: Final = TypeAdapter(tuple[str | bytes | int, ...])


def _triggered(reply: object) -> Triggered | None:
    try:
        parts: Final = tuple(
            part.decode() if isinstance(part, bytes) else str(part) for part in _REPLY.validate_python(reply)
        )
        match parts:
            case ("triggered", misses, tokens, first):
                return Triggered(misses=int(misses), proven_write_tokens=int(tokens), first_miss_ms=int(float(first)))
            case _:
                return None
    except (ValidationError, ValueError):
        return None


class RedisStreakStore:
    def __init__(self, run_script: RunScript) -> None:
        self._run_script: Final = run_script

    async def apply(self, update: StreakUpdate, now_ms: int) -> Triggered | None:
        slot: Final = update.scope.slot()
        prefix: Final = update.prefix_fingerprint if isinstance(update, PrefixEvent) else ""
        keys: Final = (f"{slot}:break", f"{slot}:{prefix}:state", f"{slot}:{prefix}:misses")
        args: Final = (
            (
                update.event.value,
                update.started_ms,
                update.ended_ms,
                now_ms,
                TTL_MS,
                update.call_id,
                update.write_tokens,
            )
            if isinstance(update, PrefixEvent)
            else ("break", update.ended_ms, update.ended_ms, now_ms, 0, "", 0)
        )
        return _triggered(await self._run_script(keys, (*args, MISSES_TO_ALERT, _BREAK_TTL_MS)))


def redis_streaks(redis: "RedisCache | None") -> StreakStore | None:
    return RedisStreakStore(redis.async_register_script(STREAK_SCRIPT)) if redis is not None else None


class _Payload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    litellm_call_id: str | None = None
    call_type: str
    stream: bool | None = None
    model: str = ""
    custom_llm_provider: str | None = None
    api_base: str | None = None
    model_id: str | None = None
    start_time: float = Field(alias="startTime")
    end_time: float = Field(alias="endTime")


class _Request(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    complete_input_dict: object = None


class _Call(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", arbitrary_types_allowed=True)

    standard_logging_object: _Payload
    additional_args: _Request = _Request()
    original_response: object = None
    api_key: object = None
    stream: bool | None = None
    cache_hit: bool | None = None


@dataclass(frozen=True, slots=True)
class _Observed:
    update: StreakUpdate
    model: str


def _event(succeeded: bool, streamed: bool, raw_response: object) -> tuple[StreakEvent, int]:
    counters: Final = cache_counters(raw_response) if succeeded and not streamed else None
    if counters is None:
        return StreakEvent.UNKNOWN, 0
    if counters.write_tokens > 0:
        return StreakEvent.PROOF, counters.write_tokens
    return (StreakEvent.HIT if counters.read_tokens > 0 else StreakEvent.MISS), 0


def observe(kwargs: Mapping[str, object], key: bytes, succeeded: bool) -> _Observed | None:
    """Streaming requests are never assessed: their logged usage is normalized, so it breaks the streak."""
    try:
        call: Final = _Call.model_validate(kwargs)
    except ValidationError:
        return None
    payload: Final = call.standard_logging_object
    if payload.custom_llm_provider != "anthropic" or payload.call_type not in _CALL_TYPES or not payload.model_id:
        return None
    if call.cache_hit is True:
        return None
    api_key: Final = call.api_key if isinstance(call.api_key, str) and call.api_key else None
    scope: Final = Scope(payload.model_id, keyed_digest(key, payload.model, payload.api_base, api_key))
    ended: Final = int(payload.end_time * 1000)
    match cacheable_prefix(call.additional_args.complete_input_dict, key):
        case NotCaching():
            return None
        case Unsupported():
            return _Observed(ScopeBreak(scope, ended), payload.model)
        case CacheablePrefix(fingerprint=fingerprint):
            event, tokens = _event(succeeded, payload.stream is True or call.stream is True, call.original_response)
            return _Observed(
                PrefixEvent(
                    scope,
                    fingerprint,
                    event,
                    payload.litellm_call_id or payload.id,
                    int(payload.start_time * 1000),
                    ended,
                    tokens,
                ),
                payload.model,
            )


def _moment(epoch_ms: int) -> datetime:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)


def detection(observed: PrefixEvent, model: str, triggered: Triggered) -> AlertDetection:
    start: Final = _moment(triggered.first_miss_ms)
    return AlertDetection(
        type="prompt_cache",
        resource_type="deployment",
        resource_id=observed.scope.deployment_id,
        window_start=start,
        window_end=max(_moment(observed.ended_ms), start) + timedelta(milliseconds=1),
        evidence=PromptCacheEvidence(
            config_fingerprint=observed.scope.config_fingerprint,
            prefix_fingerprint=observed.prefix_fingerprint,
            provider="anthropic",
            provider_model=model or observed.scope.deployment_id,
            consecutive_misses=triggered.misses,
            proven_write_tokens=triggered.proven_write_tokens,
            ttl_seconds=300,
        ),
    )


class PromptCacheAlertLogger(CustomLogger):
    """Redis or database trouble skips the evidence and never fails the request. Evidence lost to a Redis error may
    have been what ended a streak, so before anything else reaches Redis the lost evidence's scope is broken as of
    that request's end."""

    def __init__(
        self,
        key: bytes,
        streaks: Callable[[], StreakStore | None],
        alerts: AlertStore,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__()
        self._key: Final = key
        self._streaks: Final = streaks
        self._alerts: Final = alerts
        self._clock: Final = clock
        self._lost: Mapping[Scope, int] = MappingProxyType({})

    def exempt_from_message_redaction(self) -> bool:
        return True

    def _remember(self, update: StreakUpdate) -> None:
        ended: Final = max(self._lost.get(update.scope, update.ended_ms), update.ended_ms)
        self._lost = MappingProxyType({**self._lost, update.scope: ended})

    async def _break_lost_scopes(self, store: StreakStore, now_ms: int) -> None:
        """A scope is forgotten only if no newer evidence was lost for it while its break was in flight."""
        pending: Final = next(iter(self._lost.items()), None)
        if pending is None:
            return
        scope, ended = pending
        await store.apply(ScopeBreak(scope, ended), now_ms)
        if self._lost.get(scope) == ended:
            self._lost = MappingProxyType({key: value for key, value in self._lost.items() if key != scope})
        await self._break_lost_scopes(store, now_ms)

    async def _apply(self, store: StreakStore, update: StreakUpdate) -> Triggered | None:
        now_ms: Final = int(self._clock() * 1000)
        try:
            await self._break_lost_scopes(store, now_ms)
            return await store.apply(update, now_ms)
        except _STORE_FAILURES as error:
            verbose_proxy_logger.warning("prompt cache alert streak unavailable (%s)", type(error).__name__)
            self._remember(update)
            return None

    async def _observe(self, kwargs: Mapping[str, object], succeeded: bool) -> None:
        from prisma.errors import PrismaError

        store: Final = self._streaks()
        observed: Final = await offload_token_count(observe)(kwargs, self._key, succeeded) if store else None
        if store is None or observed is None:
            return
        triggered: Final = await self._apply(store, observed.update)
        if triggered is None or not isinstance(observed.update, PrefixEvent):
            return
        try:
            await self._alerts.record(detection(observed.update, observed.model, triggered))
        except (PrismaError, OSError, TimeoutError) as error:
            verbose_proxy_logger.warning("prompt cache alert not saved (%s)", type(error).__name__)

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._observe(kwargs, succeeded=True)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._observe(kwargs, succeeded=False)


def prompt_cache_alert_logger(
    general_settings: Mapping[str, object],
    alerts: AlertStore | None,
    redis: Callable[[], "RedisCache | None"],
    salt: str | None,
) -> PromptCacheAlertLogger | None:
    """Off unless `general_settings.prompt_cache_alerts` is true and a database and a shared salt exist. Without
    Redis nothing is tracked, since one worker's view of a streak could miss another worker's hit."""
    if general_settings.get(PROMPT_CACHE_ALERTS_SETTING) is not True or alerts is None or not salt:
        return None
    return PromptCacheAlertLogger(fingerprint_key(salt), lambda: redis_streaks(redis()), alerts)
