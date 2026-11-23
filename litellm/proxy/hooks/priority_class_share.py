"""Splits a priority class's share of a model among the callers active on it.

Enabled by ``priority_reservation_settings.priority_class_share_config``.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from litellm.caching.caching import DualCache
from litellm.redis.stores.active_teams import ActiveTeamStore
from litellm.redis.stores.connection import connect

MEMBER_SEPARATOR: Final = "\x1f"
DEFAULT_CLASS: Final = "\x1edefault"
_POOL_PREFIX: Final = f"{MEMBER_SEPARATOR}pool{MEMBER_SEPARATOR}"


def pool_member(priority: str) -> str:
    """The registry id that stands for every key without a team in ``priority``. It keeps the class present without
    being a team, and starts with the separator so no team id can spell it."""
    return f"{_POOL_PREFIX}{priority}"


def team_member(team_id: str) -> str:
    """The registry id of a real team. It never starts with the separator, so it cannot spell a pool marker."""
    return f"team{MEMBER_SEPARATOR}{team_id}"


def key_member(key_hash: str) -> str:
    """The registry id and counter owner of a key that takes its own slice of its class, apart from every team id."""
    return f"key{MEMBER_SEPARATOR}{key_hash}"


def customer_member(kind: str, entity_id: str) -> str:
    """The registry id and counter owner of a customer organization or customer member taking its own slice."""
    return f"{kind}{MEMBER_SEPARATOR}{entity_id}"


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


class ActiveTeamRegistry:
    """Which teams used a model in the current window, as one sorted set per model scored by last-seen time, in Redis
    when the proxy has one and otherwise in process memory."""

    def __init__(self, dual_cache: DualCache, time_provider: Callable[[], float] | None = None) -> None:
        self._store: Final = ActiveTeamStore(connect(dual_cache.redis_cache).active_teams(time_provider))

    async def touch(self, model: str, team_id: str, priority: str, window: int) -> Mapping[str, str]:
        """Record ``team_id`` as active on ``model`` and return every team active on it, the caller included. The
        caller is always listed under ``priority``, whatever class another worker's clock last saw it in."""
        seen: Final = await self._store.touch(model, team_id, str(priority), window)
        return MappingProxyType({**seen, team_id: priority})
