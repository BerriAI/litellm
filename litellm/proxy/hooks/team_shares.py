"""Guaranteed per-team shares for the dynamic rate limiter.

Each team that sent a request on a model inside the current window is a member of that model's share table. A
priority class gets its configured weight, normalized over the classes that have at least one active team, and
the class share is split equally among its active teams. A team may borrow above its share while the model is
under its saturation threshold, bounded by an optional burst multiplier, and is held to its share once the model
is saturated.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Protocol, TypeAlias

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth

ThrottleReason: TypeAlias = Literal["over_share_saturated", "burst_cap", "model_ceiling"]
DEFAULT_CLASS: Final = "default"
_MEMBER_TABLE: Final[TypeAdapter[dict[str, tuple[str, float]]]] = TypeAdapter(dict[str, tuple[str, float]])
_STORED_TABLE: Final[TypeAdapter[str | bytes | dict[str, object] | None]] = TypeAdapter(
    str | bytes | dict[str, object] | None
)


class _TextCache(Protocol):
    """The two cache tiers, which store the encoded member table as text."""

    async def async_get_cache(self, key: str) -> object: ...

    async def async_set_cache(self, key: str, value: object, *, ttl: float | None) -> object: ...


@dataclass(frozen=True, slots=True)
class TeamShare:
    """One team's slice of a model, as a fraction of the model's capacity."""

    team_id: str
    priority: str
    fraction: float
    active_teams_in_class: int
    active_classes: int

    def tokens(self, model_tpm: int | None) -> int | None:
        return None if model_tpm is None else int(model_tpm * self.fraction)

    def requests(self, model_rpm: int | None) -> int | None:
        return None if model_rpm is None else int(model_rpm * self.fraction)


def compute_team_share(
    team_id: str,
    members: Mapping[str, str],
    class_weights: Mapping[str, float],
    default_weight: float,
) -> TeamShare:
    """``members`` maps each active team to its priority class. Weights of absent classes are given away and each
    present class is split equally among its teams, so the fractions of all active teams sum to 1."""
    priority: Final = members[team_id]
    present_classes: Final = frozenset(members.values())
    weights: Final = {cls: class_weights.get(cls, default_weight) for cls in present_classes}
    total_weight: Final = sum(weights.values())
    class_fraction: Final = weights[priority] / total_weight if total_weight > 0 else 1.0 / len(present_classes)
    teams_in_class: Final = sum(1 for cls in members.values() if cls == priority)
    return TeamShare(
        team_id=team_id,
        priority=priority,
        fraction=class_fraction / teams_in_class,
        active_teams_in_class=teams_in_class,
        active_classes=len(present_classes),
    )


def team_burst_multiplier(user_api_key_dict: UserAPIKeyAuth, default: float | None) -> float | None:
    """The team's ``max_borrow_multiplier`` metadata, else the configured default. ``None`` means unbounded
    borrowing below the saturation threshold."""
    raw: Final = (user_api_key_dict.team_metadata or {}).get("max_borrow_multiplier")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return max(1.0, float(raw))
    return default


class TeamShareRegistry:
    """Which teams used a model recently, keyed by model, shared across workers through the dual cache.

    The table lives in Redis for one rate-limit window and in memory for ``local_ttl`` seconds, the same horizon
    the limiter caches saturation on, so every worker converges on the same member set within that lag. A team is
    rewritten only when its entry is older than a quarter window, which keeps the write rate bounded."""

    def __init__(self, dual_cache: DualCache, window_size: int, local_ttl: int) -> None:
        self._memory: Final[_TextCache] = dual_cache.in_memory_cache
        self._redis: Final[_TextCache | None] = dual_cache.redis_cache
        self._window_size: Final = window_size
        self._local_ttl: Final = local_ttl

    @staticmethod
    def key(model: str) -> str:
        return f"dynamic_rate_limiter:team_shares:{model}"

    async def members(self, model: str, team_id: str, priority: str, now: float | None = None) -> Mapping[str, str]:
        """Record ``team_id`` as active on ``model`` and return every team active inside the window with its class."""
        current: Final = time.time() if now is None else now
        stored: Final = await self._read(model)
        fresh_cutoff: Final = current - self._window_size
        live: Final = {team: entry for team, entry in stored.items() if entry[1] > fresh_cutoff}
        own: Final = live.get(team_id)
        stale_own: Final = own is None or own[0] != priority or own[1] < current - self._window_size / 4
        table: Final = {**live, team_id: (priority, current)} if stale_own else live
        if stale_own or len(table) != len(stored):
            await self._write(model, table)
        return MappingProxyType({team: entry[0] for team, entry in table.items()})

    async def _read(self, model: str) -> dict[str, tuple[str, float]]:
        key: Final = self.key(model)
        cached: Final = _STORED_TABLE.validate_python(await self._memory.async_get_cache(key))
        raw: Final = (
            cached
            if cached is not None or self._redis is None
            else _STORED_TABLE.validate_python(await self._redis.async_get_cache(key))
        )
        if raw is None:
            return {}
        try:
            return _MEMBER_TABLE.validate_python(json.loads(raw) if isinstance(raw, (str, bytes)) else raw)
        except (ValueError, TypeError) as error:
            verbose_proxy_logger.warning(
                "Dynamic rate limiter: discarding unreadable team share table %s: %s", key, error
            )
            return {}

    async def _write(self, model: str, table: Mapping[str, tuple[str, float]]) -> None:
        key: Final = self.key(model)
        encoded: Final = json.dumps({team: list(entry) for team, entry in table.items()})
        await self._memory.async_set_cache(key, encoded, ttl=self._local_ttl)
        if self._redis is not None:
            await self._redis.async_set_cache(key, encoded, ttl=self._window_size)
