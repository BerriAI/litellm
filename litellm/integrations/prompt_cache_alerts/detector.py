"""Opens one advisory prompt cache alert per deployment and configuration when three observed requests on a prefix
the provider proved cacheable report no cache read or write.

Only requests whose callback reaches this logger and Redis are observed. A delayed, lost or unlogged request (no-log,
disabled callbacks, a Redis error on another worker) can fall between the observed misses, so an alert does not
guarantee the misses were consecutive or that the provider cache was down.

Streaks live in Redis and every update is one Lua script. Events are ordered by when their request started and
ended, which a replayed callback repeats exactly: a proof, hit or deployment reset moves the epoch to the end of the
request that caused it, and only a miss that started after the epoch counts. The proof expires with the cache entry
it proved, measured from the request's start. Once the stored proof has expired, the next hit or miss drops the
streak, even a hit that started before the expiry, so only a new write can prove the prefix again. After an alert is
saved, the configuration stays quiet for a day.
"""

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from redis.exceptions import RedisError

from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.token_counter import offload_token_count
from litellm.models.alert import AlertDetection, AlertStore, PromptCacheEvidence

from .evidence import CacheablePrefix, NotCaching, cache_counters, cacheable_prefix, fingerprint_key, keyed_digest

if TYPE_CHECKING:
    from litellm.caching.redis_cache import RedisCache

MISSES_TO_ALERT: Final = 3
TTL_MS: Final = 300_000
_QUIET_MS: Final = 86_400_000
_CALL_TYPES: Final = frozenset(("completion", "acompletion"))
_STORE_FAILURES: Final = (RedisError, RedisCircuitBreakerOpenError, OSError, TimeoutError, asyncio.TimeoutError)

STREAK_SCRIPT: Final = """
local event = ARGV[1]
local started = tonumber(ARGV[2])
local ended = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
local function later(a, b)
  if a > b then return a end
  return b
end
if event == 'quiet' then
  redis.call('SET', KEYS[4], '1', 'PX', ARGV[5])
  return {}
end
local reset = tonumber(redis.call('GET', KEYS[1]) or '0')
if event == 'reset' then
  redis.call('SET', KEYS[1], later(reset, ended), 'PX', ARGV[5])
  return {}
end
local ttl = tonumber(ARGV[5])
local proven_until = tonumber(redis.call('HGET', KEYS[2], 'until') or '0')
local epoch = later(tonumber(redis.call('HGET', KEYS[2], 'epoch') or '0'), reset)
if event == 'proof' then
  if started + ttl <= now then return {} end
  local live = later(proven_until, started + ttl)
  local fresh = later(epoch, ended)
  redis.call('HSET', KEYS[2], 'tokens', ARGV[7], 'until', live, 'epoch', fresh)
  redis.call('PEXPIRE', KEYS[2], live - now)
  redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', fresh)
  redis.call('PEXPIRE', KEYS[3], live - now)
  return {}
end
if proven_until <= now then
  redis.call('DEL', KEYS[2], KEYS[3])
  return {}
end
if event == 'hit' then
  local fresh = later(epoch, ended)
  proven_until = later(proven_until, started + ttl)
  redis.call('HSET', KEYS[2], 'until', proven_until, 'epoch', fresh)
  redis.call('PEXPIRE', KEYS[2], proven_until - now)
  redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', fresh)
  redis.call('PEXPIRE', KEYS[3], proven_until - now)
  return {}
end
if started <= epoch then return {} end
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', epoch)
redis.call('ZADD', KEYS[3], 'NX', started, ARGV[6])
redis.call('PEXPIRE', KEYS[3], proven_until - now)
local misses = redis.call('ZRANGE', KEYS[3], 0, -1, 'WITHSCORES')
local count = #misses / 2
if count < tonumber(ARGV[8]) then return {} end
redis.call('HSET', KEYS[2], 'epoch', misses[#misses])
redis.call('DEL', KEYS[3])
if redis.call('EXISTS', KEYS[4]) == 1 then return {} end
return {count, redis.call('HGET', KEYS[2], 'tokens'), misses[2]}
"""

RunScript: TypeAlias = Callable[[Sequence[str], Sequence[str | int]], Awaitable[object]]
_REPLY: Final = TypeAdapter(tuple[str | bytes | int, ...])


@dataclass(frozen=True, slots=True)
class Assessed:
    """An attempt whose counters the provider reported, on a prefix with one five-minute breakpoint."""

    deployment_id: str
    config_fingerprint: str
    model: str
    prefix_fingerprint: str
    event: Literal["proof", "hit", "miss"]
    call_id: str
    started_ms: int
    ended_ms: int
    write_tokens: int = 0


@dataclass(frozen=True, slots=True)
class DeploymentReset:
    """A caching attempt that cannot be assessed: streamed, failed, without authoritative usage, or with a prefix
    that cannot be compared. Its identity may not match the assessed one, so every streak on the deployment ends."""

    deployment_id: str
    ended_ms: int


Update: TypeAlias = Assessed | DeploymentReset


@dataclass(frozen=True, slots=True)
class Triggered:
    misses: int
    proven_write_tokens: int
    first_miss_ms: int


def _keys(deployment_id: str, config: str = "", prefix: str = "") -> tuple[str, str, str, str]:
    slot: Final = f"prompt_cache_alert:{{{hashlib.sha256(deployment_id.encode()).hexdigest()}}}"
    return (
        f"{slot}:reset",
        f"{slot}:{config}:{prefix}:state",
        f"{slot}:{config}:{prefix}:misses",
        f"{slot}:{config}:quiet",
    )


async def apply(run: RunScript, update: Update, now_ms: int) -> Triggered | None:
    if isinstance(update, DeploymentReset):
        await run(
            _keys(update.deployment_id), ("reset", 0, update.ended_ms, now_ms, 2 * TTL_MS, "", 0, MISSES_TO_ALERT)
        )
        return None
    keys: Final = _keys(update.deployment_id, update.config_fingerprint, update.prefix_fingerprint)
    args: Final = (
        update.event,
        update.started_ms,
        update.ended_ms,
        now_ms,
        TTL_MS,
        update.call_id,
        update.write_tokens,
    )
    reply: Final = await run(keys, (*args, MISSES_TO_ALERT))
    try:
        match tuple(part.decode() if isinstance(part, bytes) else str(part) for part in _REPLY.validate_python(reply)):
            case (misses, tokens, first):
                return Triggered(int(misses), int(tokens), int(float(first)))
            case _:
                return None
    except (ValidationError, ValueError):
        return None


async def quiet(run: RunScript, update: Assessed, now_ms: int) -> None:
    keys: Final = _keys(update.deployment_id, update.config_fingerprint)
    await run(keys, ("quiet", 0, 0, now_ms, _QUIET_MS, "", 0, MISSES_TO_ALERT))


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


def observe(kwargs: Mapping[str, object], key: bytes, succeeded: bool) -> Update | None:
    try:
        call: Final = _Call.model_validate(kwargs)
    except ValidationError:
        return None
    payload: Final = call.standard_logging_object
    deployment: Final = payload.model_id
    if payload.custom_llm_provider != "anthropic" or payload.call_type not in _CALL_TYPES or not deployment:
        return None
    if call.cache_hit is True:
        return None
    ended: Final = int(payload.end_time * 1000)
    prefix: Final = cacheable_prefix(call.additional_args.complete_input_dict, key)
    if isinstance(prefix, NotCaching):
        return None
    streamed: Final = payload.stream is True or call.stream is True
    counters: Final = cache_counters(call.original_response) if succeeded and not streamed else None
    if not isinstance(prefix, CacheablePrefix) or counters is None:
        return DeploymentReset(deployment, ended)
    api_key: Final = call.api_key if isinstance(call.api_key, str) and call.api_key else None
    return Assessed(
        deployment_id=deployment,
        config_fingerprint=keyed_digest(key, payload.model, payload.api_base, api_key),
        model=payload.model or deployment,
        prefix_fingerprint=prefix.fingerprint,
        event="proof" if counters.write_tokens > 0 else "hit" if counters.read_tokens > 0 else "miss",
        call_id=payload.litellm_call_id or payload.id,
        started_ms=int(payload.start_time * 1000),
        ended_ms=ended,
        write_tokens=counters.write_tokens,
    )


def detection(update: Assessed, triggered: Triggered) -> AlertDetection:
    start: Final = datetime.fromtimestamp(triggered.first_miss_ms / 1000, tz=timezone.utc)
    end: Final = datetime.fromtimestamp(update.ended_ms / 1000, tz=timezone.utc)
    return AlertDetection(
        type="prompt_cache",
        resource_type="deployment",
        resource_id=update.deployment_id,
        window_start=start,
        window_end=max(end, start) + timedelta(milliseconds=1),
        evidence=PromptCacheEvidence(
            config_fingerprint=update.config_fingerprint,
            prefix_fingerprint=update.prefix_fingerprint,
            provider="anthropic",
            provider_model=update.model,
            consecutive_misses=triggered.misses,
            proven_write_tokens=triggered.proven_write_tokens,
            ttl_seconds=300,
        ),
    )


class PromptCacheAlertLogger(CustomLogger):
    """Redis or database trouble never fails the request. Evidence lost to a Redis error, or seen while the shared
    Redis client is unavailable, may have been what ended a streak, so before anything else reaches Redis the lost
    evidence's deployment is reset as of that request's end. A reset older than a proof's lifetime can no longer
    matter and is dropped."""

    def __init__(
        self,
        key: bytes,
        script: Callable[[], RunScript | None],
        alerts: AlertStore,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._key: Final = key
        self._script: Final = script
        self._alerts: Final = alerts
        self._clock: Final = clock
        self._lost: Mapping[str, int] = MappingProxyType({})

    def exempt_from_message_redaction(self) -> bool:
        return True

    async def _apply(self, run: RunScript, update: Update, now_ms: int) -> Triggered | None:
        try:
            while pending := next((item for item in self._lost.items() if item[1] > now_ms - TTL_MS), None):
                deployment, ended = pending
                await apply(run, DeploymentReset(deployment, ended), now_ms)
                if self._lost.get(deployment) == ended:
                    self._lost = MappingProxyType({k: v for k, v in self._lost.items() if k != deployment})
            self._lost = MappingProxyType({k: v for k, v in self._lost.items() if v > now_ms - TTL_MS})
            return await apply(run, update, now_ms)
        except _STORE_FAILURES as error:
            verbose_proxy_logger.warning("prompt cache alert streak unavailable (%s)", type(error).__name__)
            self._remember(update)
            return None

    def _remember(self, update: Update) -> None:
        lost: Final = max(self._lost.get(update.deployment_id, update.ended_ms), update.ended_ms)
        self._lost = MappingProxyType({**self._lost, update.deployment_id: lost})

    async def _observe(self, kwargs: Mapping[str, object], succeeded: bool) -> None:
        from prisma.errors import PrismaError

        update: Final = await offload_token_count(observe)(kwargs, self._key, succeeded)
        run: Final = self._script()
        if update is None:
            return
        if run is None:
            self._remember(update)
            return
        now_ms: Final = int(self._clock() * 1000)
        triggered: Final = await self._apply(run, update, now_ms)
        if triggered is None or not isinstance(update, Assessed):
            return
        try:
            await self._alerts.record(detection(update, triggered))
            await quiet(run, update, now_ms)
        except (PrismaError, *_STORE_FAILURES) as error:
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
    alerts: AlertStore, redis: Callable[[], "RedisCache | None"], salt: str
) -> PromptCacheAlertLogger:
    """Without Redis nothing is tracked, since one worker's view of a streak could miss another worker's hit."""

    def script() -> RunScript | None:
        cache: Final = redis()
        return cache.async_register_script(STREAK_SCRIPT) if cache is not None else None

    return PromptCacheAlertLogger(fingerprint_key(salt), script, alerts)
