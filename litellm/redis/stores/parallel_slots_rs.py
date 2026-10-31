from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Final

from litellm.redis.failure import RedisFailure, from_native
from litellm.rust_bridge._native import NativeParallelSlots, NativeRedisFailure

SLOT_TTL: Final = timedelta(seconds=3600)


@dataclass(frozen=True, slots=True)
class Descriptor:
    key: str
    value: str


@dataclass(frozen=True, slots=True)
class Gauge:
    descriptor: Descriptor
    limit: int


@dataclass(frozen=True, slots=True)
class Admitted:
    in_flight: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Rejected:
    gauge: Gauge
    in_flight: int


def _pair(descriptor: Descriptor) -> tuple[str, str]:
    return (descriptor.key, descriptor.value)


@dataclass(frozen=True, slots=True)
class NativeParallelSlotStore:
    handle: NativeParallelSlots
    slot_ttl: timedelta = SLOT_TTL

    async def acquire(self, gauges: Sequence[Gauge], slot_id: str) -> Admitted | Rejected | RedisFailure:
        reply: Final = await self.handle.acquire(
            [(*_pair(gauge.descriptor), gauge.limit) for gauge in gauges],
            slot_id,
            slot_ttl_seconds=int(self.slot_ttl.total_seconds()),
        )
        match reply:
            case NativeRedisFailure():
                return from_native(reply)
            case ("rejected", int() as index, int() as in_flight, _):
                return Rejected(gauges[index], in_flight)
            case (_, list() as in_flight):
                return Admitted(tuple(in_flight))

    async def release(self, descriptors: Sequence[Descriptor], slot_id: str) -> tuple[int, ...] | RedisFailure:
        reply: Final = await self.handle.release([_pair(d) for d in descriptors], slot_id)
        return from_native(reply) if isinstance(reply, NativeRedisFailure) else tuple(reply)

    async def count(self, descriptors: Sequence[Descriptor]) -> tuple[int, ...] | RedisFailure:
        reply: Final = await self.handle.count(
            [_pair(d) for d in descriptors], slot_ttl_seconds=int(self.slot_ttl.total_seconds())
        )
        return from_native(reply) if isinstance(reply, NativeRedisFailure) else tuple(reply)
