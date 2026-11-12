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
from typing import Annotated, Final, Literal, Protocol, TypeAlias, cast

from pydantic import Field, TypeAdapter, ValidationError

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.redis.stores.rate_limits import RATE_LIMITS_TARGET

ThrottleReason: TypeAlias = Literal["over_share_saturated", "burst_cap", "model_ceiling"]
DEFAULT_CLASS: Final = "default"
_MEMBER_TABLE: Final[TypeAdapter[dict[str, tuple[str, float]]]] = TypeAdapter(dict[str, tuple[str, float]])
_STORED_TABLE: Final[TypeAdapter[str | bytes | dict[str, object] | None]] = TypeAdapter(
    str | bytes | dict[str, object] | None
)
_ACTIVE_ENTRIES: Final[TypeAdapter[list[tuple[str | bytes, float]]]] = TypeAdapter(list[tuple[str | bytes, float]])
_BURST_MULTIPLIER: Final[TypeAdapter[float]] = TypeAdapter(Annotated[float, Field(ge=1.0, allow_inf_nan=False)])
_MEMBER_SEPARATOR: Final = "\x1f"


class _SortedSetCommands(Protocol):
    """The sorted-set commands the registry uses, present on both the Redis and RedisCluster async clients."""

    async def zremrangebyscore(self, name: str, min: float | str, max: float | str) -> object: ...

    async def zadd(self, name: str, mapping: Mapping[str, float]) -> object: ...

    async def expire(self, name: str, time: int) -> object: ...

    async def zrangebyscore(self, name: str, min: float | str, max: float | str, withscores: bool) -> object: ...


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
    borrowing below the saturation threshold. A multiplier below 1 or not finite is ignored with a warning."""
    raw: Final[object] = (user_api_key_dict.team_metadata or {}).get("max_borrow_multiplier")
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        return default
    try:
        return _BURST_MULTIPLIER.validate_python(raw)
    except ValidationError:
        verbose_proxy_logger.warning(
            "Dynamic rate limiter: ignoring max_borrow_multiplier %r on team %s, it must be a finite number >= 1",
            raw,
            user_api_key_dict.team_id,
        )
        return default


class TeamShareRegistry:
    """Which teams used a model recently, keyed by model, shared across workers through Redis.

    Redis holds one sorted set per model, scored by the time each team last registered, so workers add and prune
    entries atomically instead of rewriting a table they read earlier. Each worker keeps a snapshot in memory for
    ``local_ttl`` seconds, the horizon the limiter caches saturation on, and re-registers a team once its entry is
    older than a quarter window, which bounds the Redis write rate. Without Redis the in-memory table is the only
    store and lives for the full window."""

    def __init__(self, dual_cache: DualCache, window_size: int, local_ttl: int) -> None:
        self._memory: Final[_TextCache] = dual_cache.in_memory_cache
        self._redis: Final[RedisCache | None] = dual_cache.redis_cache
        self._window_size: Final = window_size
        self._local_ttl: Final = local_ttl if dual_cache.redis_cache is not None else window_size

    @staticmethod
    def key(model: str) -> str:
        return f"dynamic_rate_limiter:team_shares:{model}"

    async def members(self, model: str, team_id: str, priority: str, now: float | None = None) -> Mapping[str, str]:
        """Record ``team_id`` as active on ``model`` and return every team active inside the window with its class."""
        current: Final = time.time() if now is None else now
        cutoff: Final = current - self._window_size
        snapshot: Final = {team: entry for team, entry in (await self._read_local(model)).items() if entry[1] > cutoff}
        own: Final = snapshot.get(team_id)
        fresh_own: Final = own is not None and own[0] == priority and own[1] >= current - self._window_size / 4
        table: Final = snapshot if fresh_own else await self._register(model, team_id, priority, current, snapshot)
        return MappingProxyType({team: entry[0] for team, entry in table.items()})

    async def _register(
        self, model: str, team_id: str, priority: str, now: float, snapshot: Mapping[str, tuple[str, float]]
    ) -> dict[str, tuple[str, float]]:
        table: Final = (
            {**snapshot, team_id: (priority, now)}
            if self._redis is None
            else await self._register_in_redis(self._redis, model, team_id, priority, now, snapshot)
        )
        await self._write_local(model, table)
        return table

    @with_service_target(RATE_LIMITS_TARGET)
    async def _register_in_redis(
        self,
        redis: RedisCache,
        model: str,
        team_id: str,
        priority: str,
        now: float,
        snapshot: Mapping[str, tuple[str, float]],
    ) -> dict[str, tuple[str, float]]:
        key: Final = self.key(model)
        cutoff: Final = now - self._window_size
        try:
            # the cluster client stubs omit the sorted-set mixin both clients implement
            client: Final = cast(_SortedSetCommands, redis.init_async_client())
            await client.zremrangebyscore(key, "-inf", cutoff)
            await client.zadd(key, {f"{team_id}{_MEMBER_SEPARATOR}{priority}": now})
            await client.expire(key, self._window_size)
            entries = _ACTIVE_ENTRIES.validate_python(await client.zrangebyscore(key, cutoff, "+inf", withscores=True))
        except Exception as error:  # noqa: BLE001 - any Redis failure falls back to this worker's own snapshot
            verbose_proxy_logger.warning(
                "Dynamic rate limiter: could not register team %s on %s: %s", team_id, key, error
            )
            return {**snapshot, team_id: (priority, now)}
        table: dict[str, tuple[str, float]] = {}
        for member, score in entries:
            text = member.decode() if isinstance(member, bytes) else member
            team, _, team_class = text.rpartition(_MEMBER_SEPARATOR)
            previous = table.get(team)
            if previous is None or score >= previous[1]:
                table[team] = (team_class, score)
        return table

    async def _read_local(self, model: str) -> dict[str, tuple[str, float]]:
        key: Final = self.key(model)
        raw: Final = _STORED_TABLE.validate_python(await self._memory.async_get_cache(key))
        if raw is None:
            return {}
        try:
            return _MEMBER_TABLE.validate_python(json.loads(raw) if isinstance(raw, (str, bytes)) else raw)
        except (ValueError, TypeError) as error:
            verbose_proxy_logger.warning(
                "Dynamic rate limiter: discarding unreadable team share table %s: %s", key, error
            )
            return {}

    async def _write_local(self, model: str, table: Mapping[str, tuple[str, float]]) -> None:
        encoded: Final = json.dumps({team: list(entry) for team, entry in table.items()})
        await self._memory.async_set_cache(self.key(model), encoded, ttl=self._local_ttl)
