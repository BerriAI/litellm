"""Splits a priority class's share of a model among the teams active on it.

Enabled by ``priority_reservation_settings.priority_class_share_splits_equally_among_active_teams``.
"""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol

from pydantic import TypeAdapter

from litellm.caching.caching import DualCache
from litellm.caching.in_memory_cache import InMemoryCache

ACTIVE_TEAMS_KEY_PREFIX: Final = "dynamic_rate_limiter:active_teams:"
MEMBER_SEPARATOR: Final = "\x1f"
DEFAULT_CLASS: Final = "default"

_TOUCH_LUA: Final = (
    "redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1]) "
    "redis.call('ZADD', KEYS[1], ARGV[2], ARGV[3]) "
    "redis.call('EXPIRE', KEYS[1], ARGV[4]) "
    "return redis.call('ZRANGE', KEYS[1], 0, -1)"
)
_MEMBERS: Final = TypeAdapter[tuple[str | bytes, ...]](tuple[str | bytes, ...])
_LOCAL_MEMBERS: Final = TypeAdapter[Mapping[str, float] | None](Mapping[str, float] | None)


class _LocalMembers(Protocol):
    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: Mapping[str, float], *, ttl: int) -> None: ...


@dataclass(frozen=True, slots=True)
class TeamShare:
    fraction: float
    active_teams_in_class: int
    active_classes: int


def team_share(
    priority: str, members: Mapping[str, str], class_weights: Mapping[str, float], default_weight: float
) -> TeamShare:
    """``members`` maps every team active on the model, the caller included, to its priority class. Classes with no
    active team give their weight away, and each present class is split equally among its active teams."""
    present: Final = frozenset(members.values())
    weights: Final = {cls: class_weights.get(cls, default_weight) for cls in present}
    total: Final = sum(weights.values())
    class_fraction: Final = weights[priority] / total if total > 0 else 1.0 / len(present)
    teams_in_class: Final = sum(1 for cls in members.values() if cls == priority)
    return TeamShare(
        fraction=class_fraction / teams_in_class, active_teams_in_class=teams_in_class, active_classes=len(present)
    )


def _members(raw: tuple[str | bytes, ...]) -> Mapping[str, str]:
    """Oldest first, so a team seen under two classes in one window keeps its most recent one."""
    decoded: Final = (item.decode() if isinstance(item, bytes) else item for item in raw)
    pairs: Final = (member.split(MEMBER_SEPARATOR, 1) for member in decoded)
    return MappingProxyType({pair[0]: pair[1] for pair in pairs if len(pair) == 2})


class ActiveTeamRegistry:
    """Which teams used a model in the current window, as one sorted set per model scored by last-seen time, in Redis
    when the proxy has one and otherwise in process memory."""

    def __init__(self, dual_cache: DualCache, time_provider: Callable[[], float] = time.time) -> None:
        self._dual_cache = dual_cache
        self._local: _LocalMembers = InMemoryCache()
        self._now = time_provider

    async def touch(self, model: str, team_id: str, priority: str, window: int) -> Mapping[str, str]:
        """Record ``team_id`` as active on ``model`` and return every team active on it, the caller included."""
        key: Final = f"{ACTIVE_TEAMS_KEY_PREFIX}{model}"
        member: Final = f"{team_id}{MEMBER_SEPARATOR}{priority}"
        now: Final = self._now()
        redis_cache: Final = self._dual_cache.redis_cache
        if redis_cache is None:
            return self._touch_in_memory(key, member, now, window)
        touch: Final = redis_cache.async_register_script(_TOUCH_LUA)
        return _members(_MEMBERS.validate_python(await touch(keys=(key,), args=(now - window, now, member, window))))

    def _touch_in_memory(self, key: str, member: str, now: float, window: int) -> Mapping[str, str]:
        previous: Final = _LOCAL_MEMBERS.validate_python(self._local.get_cache(key)) or {}
        alive: Final = {seen_member: seen for seen_member, seen in previous.items() if seen > now - window}
        current: Final = MappingProxyType({**alive, member: now})
        self._local.set_cache(key, dict(current), ttl=window)
        return _members(tuple(sorted(current, key=current.__getitem__)))
