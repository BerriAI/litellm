from dataclasses import dataclass
from datetime import timedelta
from typing import Final

from litellm.redis.failure import RedisFailure, from_native
from litellm.rust_bridge._native import NativePodLock, NativeRedisFailure


@dataclass(frozen=True, slots=True)
class Acquired:
    pass


@dataclass(frozen=True, slots=True)
class HeldBy:
    owner: str


@dataclass(frozen=True, slots=True)
class Vacant:
    pass


@dataclass(frozen=True, slots=True)
class NativePodLockStore:
    handle: NativePodLock

    async def acquire(self, cronjob_id: str, owner: str, *, ttl: timedelta) -> Acquired | HeldBy | Vacant | RedisFailure:
        reply: Final = await self.handle.acquire(cronjob_id, owner, ttl_seconds=int(ttl.total_seconds()))
        match reply:
            case NativeRedisFailure():
                return from_native(reply)
            case ("held_by", str() as holder):
                return HeldBy(holder)
            case ("acquired", _):
                return Acquired()
            case _:
                return Vacant()

    async def holder(self, cronjob_id: str) -> HeldBy | None | RedisFailure:
        owner: Final = await self.handle.holder(cronjob_id)
        if isinstance(owner, NativeRedisFailure):
            return from_native(owner)
        return None if owner is None else HeldBy(owner)

    async def release(self, cronjob_id: str, owner: str) -> bool | RedisFailure:
        released: Final = await self.handle.release(cronjob_id, owner)
        return from_native(released) if isinstance(released, NativeRedisFailure) else released

    async def remove(self, cronjob_id: str) -> bool | RedisFailure:
        removed: Final = await self.handle.remove(cronjob_id)
        return from_native(removed) if isinstance(removed, NativeRedisFailure) else removed
