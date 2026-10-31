from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Literal

from litellm.redis.failure import RedisFailure, from_native
from litellm.rust_bridge._native import NativeRedisFailure, NativeSpendCounters

Scope = Literal["key", "team", "team_member", "user", "end_user", "org", "tag", "project"]


@dataclass(frozen=True, slots=True)
class SpendCounter:
    scope: Scope
    id: str
    team_id: str | None = None
    window: str | None = None


@dataclass(frozen=True, slots=True)
class Seeded:
    pass


@dataclass(frozen=True, slots=True)
class Existing:
    spend: float | None


def _seconds(ttl: timedelta) -> int:
    return int(ttl.total_seconds())


@dataclass(frozen=True, slots=True)
class NativeSpendCounterStore:
    handle: NativeSpendCounters

    async def read(self, counters: Sequence[SpendCounter]) -> tuple[float | None, ...] | RedisFailure:
        reply: Final = await self.handle.read(counters)
        return from_native(reply) if isinstance(reply, NativeRedisFailure) else tuple(reply)

    async def add(
        self, deltas: Sequence[tuple[SpendCounter, float]], *, ttl: timedelta
    ) -> tuple[float, ...] | RedisFailure:
        reply: Final = await self.handle.add(deltas, ttl_seconds=_seconds(ttl))
        return from_native(reply) if isinstance(reply, NativeRedisFailure) else tuple(reply)

    async def seed(self, counter: SpendCounter, spend: float, *, ttl: timedelta) -> Seeded | Existing | RedisFailure:
        reply: Final = await self.handle.seed(counter, spend, ttl_seconds=_seconds(ttl))
        match reply:
            case NativeRedisFailure():
                return from_native(reply)
            case ("existing", stored):
                return Existing(stored)
            case _:
                return Seeded()

    async def raise_to(self, counter: SpendCounter, spend: float, *, ttl: timedelta) -> float | RedisFailure:
        reply: Final = await self.handle.raise_to(counter, spend, ttl_seconds=_seconds(ttl))
        return from_native(reply) if isinstance(reply, NativeRedisFailure) else reply
