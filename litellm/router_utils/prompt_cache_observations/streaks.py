"""Per-prefix miss streaks, shared across workers.

A streak belongs to one prefix on one deployment and cache configuration. It only exists once the provider's
own cache write proved that prefix cacheable, and only for as long as that cache entry lives. Every other kind of
evidence about the prefix ends the streak, so only consecutive misses count.
"""

import hashlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final, Protocol, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm.caching.in_memory_cache import InMemoryCache

MISSES_TO_OPEN_INCIDENT: Final = 3
_IN_MEMORY_STREAK_LIMIT: Final = 10_000


class StreakEvent(StrEnum):
    PROOF = "proof"
    HIT = "hit"
    MISS = "miss"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class StreakKey:
    deployment_id: str
    config_fingerprint: str
    prefix_fingerprint: str

    def digest(self) -> str:
        return hashlib.sha256(
            f"{self.deployment_id}\0{self.config_fingerprint}\0{self.prefix_fingerprint}".encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class StreakUpdate:
    key: StreakKey
    event: StreakEvent
    call_id: str
    now_ms: int
    ttl_ms: int
    write_tokens: int = 0


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
    async def apply(self, update: StreakUpdate) -> StreakOutcome: ...


@dataclass(frozen=True, slots=True)
class _Streak:
    write_tokens: int
    proven_until_ms: int
    misses: frozenset[str] = frozenset()
    first_miss_ms: int | None = None


def _next(current: _Streak | None, update: StreakUpdate) -> tuple[_Streak | None, StreakOutcome]:
    if update.event is StreakEvent.PROOF:
        return _Streak(write_tokens=update.write_tokens, proven_until_ms=update.now_ms + update.ttl_ms), Unproven()
    proven: Final = current is not None and current.proven_until_ms > update.now_ms
    if current is None or not proven:
        return None, Unproven()
    match update.event:
        case StreakEvent.HIT:
            return _Streak(write_tokens=current.write_tokens, proven_until_ms=update.now_ms + update.ttl_ms), Unproven()
        case StreakEvent.UNKNOWN:
            return replace(current, misses=frozenset(), first_miss_ms=None), Unproven()
        case _:
            misses: Final = current.misses | {update.call_id}
            first: Final = current.first_miss_ms if current.first_miss_ms is not None else update.now_ms
            if len(misses) < MISSES_TO_OPEN_INCIDENT:
                return replace(current, misses=misses, first_miss_ms=first), Counting(len(misses))
            return (
                replace(current, misses=frozenset(), first_miss_ms=None),
                Triggered(misses=len(misses), proven_write_tokens=current.write_tokens, first_miss_ms=first),
            )


class _BoundedCache(Protocol):
    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: object, *, ttl: int) -> None: ...

    def delete_cache(self, key: str) -> None: ...


def _bounded_cache() -> _BoundedCache:
    return InMemoryCache(max_size_in_memory=_IN_MEMORY_STREAK_LIMIT, default_ttl=86_400)


class InMemoryStreakStore:
    """For a gateway without Redis. Each worker keeps its own streaks, so misses spread across workers are not
    added up, which can only delay an incident, never open one early."""

    def __init__(self) -> None:
        self._streaks: Final = _bounded_cache()

    async def apply(self, update: StreakUpdate) -> StreakOutcome:
        key: Final = update.key.digest()
        stored: Final = self._streaks.get_cache(key)
        streak, outcome = _next(stored if isinstance(stored, _Streak) else None, update)
        if streak is None:
            self._streaks.delete_cache(key)
        else:
            self._streaks.set_cache(key, streak, ttl=max(1, (streak.proven_until_ms - update.now_ms) // 1000 + 1))
        return outcome


STREAK_SCRIPT: Final = """
local now = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
if ARGV[1] == 'proof' then
  redis.call('DEL', KEYS[2])
  redis.call('DEL', KEYS[1])
  redis.call('HSET', KEYS[1], 'tokens', ARGV[5], 'until', now + ttl)
  redis.call('PEXPIRE', KEYS[1], ttl)
  return {'unproven'}
end
local proven_until = tonumber(redis.call('HGET', KEYS[1], 'until') or '0')
if proven_until <= now then
  redis.call('DEL', KEYS[1], KEYS[2])
  return {'unproven'}
end
if ARGV[1] == 'hit' then
  redis.call('DEL', KEYS[2])
  redis.call('HDEL', KEYS[1], 'first')
  redis.call('HSET', KEYS[1], 'until', now + ttl)
  redis.call('PEXPIRE', KEYS[1], ttl)
  return {'unproven'}
end
if ARGV[1] == 'unknown' then
  redis.call('DEL', KEYS[2])
  redis.call('HDEL', KEYS[1], 'first')
  return {'unproven'}
end
redis.call('SADD', KEYS[2], ARGV[4])
redis.call('PEXPIRE', KEYS[2], proven_until - now)
redis.call('HSETNX', KEYS[1], 'first', now)
local misses = redis.call('SCARD', KEYS[2])
if misses < tonumber(ARGV[6]) then
  return {'counting', misses}
end
local first = redis.call('HGET', KEYS[1], 'first')
redis.call('DEL', KEYS[2])
redis.call('HDEL', KEYS[1], 'first')
return {'triggered', misses, redis.call('HGET', KEYS[1], 'tokens'), first}
"""

RunScript: TypeAlias = Callable[[Sequence[str], Sequence[str | int]], Awaitable[object]]

_REPLY: Final = TypeAdapter(tuple[str | bytes | int, ...])


def _text(value: str | bytes | int) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _outcome(reply: object) -> StreakOutcome:
    try:
        parts: Final = tuple(_text(part) for part in _REPLY.validate_python(reply))
    except ValidationError:
        return Unproven()
    match parts:
        case ("counting", misses):
            return Counting(int(misses))
        case ("triggered", misses, tokens, first):
            return Triggered(misses=int(misses), proven_write_tokens=int(tokens), first_miss_ms=int(first))
        case _:
            return Unproven()


class RedisStreakStore:
    """One Lua script per update, so workers that see the same prefix at once apply their updates in some order
    and never lose one, and a request logged twice adds one miss."""

    def __init__(self, run_script: RunScript) -> None:
        self._run_script: Final = run_script

    async def apply(self, update: StreakUpdate) -> StreakOutcome:
        slot: Final = f"prompt_cache_streak:{{{update.key.digest()}}}"
        return _outcome(
            await self._run_script(
                (f"{slot}:state", f"{slot}:misses"),
                (
                    update.event.value,
                    update.now_ms,
                    update.ttl_ms,
                    update.call_id,
                    update.write_tokens,
                    MISSES_TO_OPEN_INCIDENT,
                ),
            )
        )
