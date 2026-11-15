"""Splits a priority class's share of a model among the teams active on it.

Enabled by ``priority_reservation_settings.priority_class_share_splits_equally_among_active_teams``.
"""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm._internal_context import with_service_target
from litellm.caching.caching import DualCache
from litellm.redis.stores.rate_limits import RATE_LIMITS_TARGET

ACTIVE_TEAMS_KEY_PREFIX: Final = "dynamic_rate_limiter:active_teams:"
MEMBER_SEPARATOR: Final = "\x1f"
DEFAULT_CLASS: Final = "\x1edefault"
_POOL_PREFIX: Final = f"{MEMBER_SEPARATOR}pool{MEMBER_SEPARATOR}"

_TOUCH_LUA: Final = (
    "redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1]) "
    "redis.call('ZADD', KEYS[1], ARGV[2], ARGV[3]) "
    "redis.call('EXPIRE', KEYS[1], ARGV[4]) "
    "return redis.call('ZRANGE', KEYS[1], 0, -1)"
)
_MEMBERS: Final = TypeAdapter[tuple[str | bytes, ...]](tuple[str | bytes, ...])


def pool_member(priority: str) -> str:
    """The registry id that stands for every key without a team in ``priority``. It keeps the class present without
    being a team, and starts with the separator so no team id can spell it."""
    return f"{_POOL_PREFIX}{priority}"


def team_member(team_id: str) -> str:
    """The registry id of a real team. It never starts with the separator, so it cannot spell a pool marker."""
    return f"team{MEMBER_SEPARATOR}{team_id}"


def is_pool_member(member: str) -> bool:
    return member.startswith(_POOL_PREFIX)


def counter_value(model: str, team_id: str) -> str:
    """One counter per model and team. The length prefix keeps ``chat`` + ``prod:victim`` apart from
    ``chat:prod`` + ``victim``."""
    return f"{len(model)}:{model}:{team_id}"


@dataclass(frozen=True, slots=True)
class TeamShare:
    fraction: float
    active_teams_in_class: int
    active_classes: int


def team_share(
    priority: str, members: Mapping[str, str], class_weights: Mapping[str, float], default_weight: float
) -> TeamShare:
    """``members`` maps every team active on the model, the caller included, to its priority class. Classes with no
    active member give their weight away, and each present class is split equally among its active teams. A pool
    member keeps its class present but takes no slice, since keys without a team are held to the class pool."""
    present: Final = frozenset(members.values())
    weights: Final = {cls: class_weights.get(cls, default_weight) for cls in present}
    total: Final = sum(weights.values())
    class_fraction: Final = weights[priority] / total if total > 0 else 1.0 / len(present)
    teams_in_class: Final = sum(1 for team, cls in members.items() if cls == priority and not is_pool_member(team))
    return TeamShare(
        fraction=class_fraction / max(1, teams_in_class),
        active_teams_in_class=teams_in_class,
        active_classes=len(present),
    )


def _members(raw: tuple[str | bytes, ...]) -> Mapping[str, str]:
    """Oldest first, so a team seen under two classes in one window keeps its most recent one. The class is split off
    from the right because a team id may itself contain the separator."""
    decoded: Final = (item.decode() if isinstance(item, bytes) else item for item in raw)
    pairs: Final = (member.rsplit(MEMBER_SEPARATOR, 1) for member in decoded)
    return MappingProxyType({pair[0]: pair[1] for pair in pairs if len(pair) == 2})


class ActiveTeamRegistry:
    """Which teams used a model in the current window, as one sorted set per model scored by last-seen time, in Redis
    when the proxy has one and otherwise in process memory."""

    def __init__(self, dual_cache: DualCache, time_provider: Callable[[], float] = time.time) -> None:
        self._dual_cache = dual_cache
        self._local: Mapping[str, Mapping[str, float]] = MappingProxyType({})
        self._now = time_provider

    @with_service_target(RATE_LIMITS_TARGET)
    async def touch(self, model: str, team_id: str, priority: str, window: int) -> Mapping[str, str]:
        """Record ``team_id`` as active on ``model`` and return every team active on it, the caller included. The
        caller is always listed under ``priority``, whatever class another worker's clock last saw it in."""
        key: Final = f"{ACTIVE_TEAMS_KEY_PREFIX}{model}"
        member: Final = f"{team_id}{MEMBER_SEPARATOR}{priority}"
        now: Final = self._now()
        redis_cache: Final = self._dual_cache.redis_cache
        seen: Final = (
            self._touch_in_memory(key, member, now, window)
            if redis_cache is None
            else _members(
                _MEMBERS.validate_python(
                    await redis_cache.async_register_script(_TOUCH_LUA)(
                        keys=(key,), args=(now - window, now, member, window)
                    )
                )
            )
        )
        return MappingProxyType({**seen, team_id: priority})

    def _touch_in_memory(self, key: str, member: str, now: float, window: int) -> Mapping[str, str]:
        """Prunes by each member's own last-seen time, so the map lives exactly as long as its newest member."""
        cutoff: Final = now - window
        alive: Final = {seen_member: seen for seen_member, seen in self._local.get(key, {}).items() if seen > cutoff}
        current: Final = MappingProxyType({**alive, member: now})
        others: Final = {
            other_key: others_members
            for other_key, others_members in self._local.items()
            if other_key != key and any(seen > cutoff for seen in others_members.values())
        }
        self._local = MappingProxyType({**others, key: current})
        return _members(tuple(sorted(current, key=current.__getitem__)))
