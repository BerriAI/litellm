"""Per-prefix miss streaks, shared across workers through Redis.

A streak belongs to one prefix on one deployment and cache configuration. It only exists once the provider's
own cache write proved that prefix cacheable, and only while that cache entry lives. Each event carries its
request's own start and end, which a replayed callback repeats exactly. A proof, hit or unknown moves the
prefix's epoch to the end of the request that caused it, and only a miss that started after the epoch can
count, so a request already in flight while the proving write was made never counts as caching having stopped.
An incident moves the epoch to its newest miss. So a replay of any earlier event, a stale proof, or a miss from
before a hit, an unknown or an incident never counts again, with no history kept beyond the current streak.
Expiry is measured from a request's start, the earliest its cache entry could have been written or read.
"""

import hashlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol, TypeAlias

from pydantic import TypeAdapter, ValidationError

from ..prompt_caching_cache import PROMPT_CACHE_ONE_HOUR_PIN_TTL_SECONDS

MISSES_TO_OPEN_INCIDENT: Final = 3
SCOPE_BREAK_TTL_MS: Final = 2 * PROMPT_CACHE_ONE_HOUR_PIN_TTL_SECONDS * 1000


class StreakEvent(StrEnum):
    PROOF = "proof"
    HIT = "hit"
    MISS = "miss"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class StreakScope:
    deployment_id: str
    config_fingerprint: str

    def slot(self) -> str:
        digest: Final = hashlib.sha256(f"{self.deployment_id}\0{self.config_fingerprint}".encode()).hexdigest()
        return f"prompt_cache_streak:{{{digest}}}"


@dataclass(frozen=True, slots=True)
class PrefixEvent:
    scope: StreakScope
    prefix_fingerprint: str
    event: StreakEvent
    call_id: str
    started_ms: int
    ended_ms: int
    ttl_ms: int
    write_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ScopeBreak:
    """A caching request on the deployment and configuration whose prefix could not be read. It may have been
    any of the scope's prefixes, so it ends every one of their streaks as of when it ended."""

    scope: StreakScope
    ended_ms: int


StreakUpdate: TypeAlias = PrefixEvent | ScopeBreak


@dataclass(frozen=True, slots=True)
class Unproven:
    pass


@dataclass(frozen=True, slots=True)
class Counting:
    misses: int


@dataclass(frozen=True, slots=True)
class Triggered:
    misses: int
    proven_write_tokens: int
    first_miss_ms: int


StreakOutcome: TypeAlias = Unproven | Counting | Triggered


class StreakStore(Protocol):
    async def apply(self, update: StreakUpdate, now_ms: int) -> StreakOutcome: ...


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

RunScript: TypeAlias = Callable[[Sequence[str], Sequence[str | int]], Awaitable[object]]

_REPLY: Final = TypeAdapter(tuple[str | bytes | int, ...])


def _text(value: str | bytes | int) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _outcome(reply: object) -> StreakOutcome:
    try:
        parts: Final = tuple(_text(part) for part in _REPLY.validate_python(reply))
        match parts:
            case ("counting", misses):
                return Counting(int(misses))
            case ("triggered", misses, tokens, first):
                return Triggered(misses=int(misses), proven_write_tokens=int(tokens), first_miss_ms=int(float(first)))
            case _:
                return Unproven()
    except (ValidationError, ValueError):
        return Unproven()


def _keys(update: StreakUpdate) -> tuple[str, str, str]:
    slot: Final = update.scope.slot()
    prefix: Final = update.prefix_fingerprint if isinstance(update, PrefixEvent) else ""
    return f"{slot}:break", f"{slot}:{prefix}:state", f"{slot}:{prefix}:misses"


def _args(update: StreakUpdate, now_ms: int) -> tuple[str | int, ...]:
    match update:
        case PrefixEvent(
            event=event, call_id=call_id, started_ms=started, ended_ms=ended, ttl_ms=ttl, write_tokens=tokens
        ):
            return (
                event.value,
                started,
                ended,
                now_ms,
                ttl,
                call_id,
                tokens,
                MISSES_TO_OPEN_INCIDENT,
                SCOPE_BREAK_TTL_MS,
            )
        case ScopeBreak(ended_ms=ended):
            return "break", ended, ended, now_ms, 0, "", 0, MISSES_TO_OPEN_INCIDENT, SCOPE_BREAK_TTL_MS


class RedisStreakStore:
    """One Lua script per update, so workers that see the same scope at once apply their updates in some order
    and never lose one. All of a scope's keys share one cluster slot, which lets a scope break reach every
    prefix's streak."""

    def __init__(self, run_script: RunScript) -> None:
        self._run_script: Final = run_script

    async def apply(self, update: StreakUpdate, now_ms: int) -> StreakOutcome:
        return _outcome(await self._run_script(_keys(update), _args(update, now_ms)))
