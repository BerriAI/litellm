import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisTransport

_RELEASE: Final = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""

_OWNER: Final = TypeAdapter(str)


def lock_key(cronjob_id: str) -> str:
    return f"cronjob_lock:{cronjob_id}"


@dataclass(frozen=True, slots=True)
class Acquired:
    pass


@dataclass(frozen=True, slots=True)
class HeldBy:
    owner: str


@dataclass(frozen=True, slots=True)
class Vacant:
    pass


class PodLockStore(Protocol):
    async def acquire(
        self, cronjob_id: str, owner: str, *, ttl: timedelta
    ) -> Acquired | HeldBy | Vacant | RedisFailure: ...

    async def holder(self, cronjob_id: str) -> HeldBy | None | RedisFailure: ...

    async def release(self, cronjob_id: str, owner: str) -> bool | RedisFailure: ...

    async def remove(self, cronjob_id: str) -> bool | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportPodLockStore:
    transport: RedisTransport

    async def acquire(
        self, cronjob_id: str, owner: str, *, ttl: timedelta
    ) -> Acquired | HeldBy | Vacant | RedisFailure:
        set_reply: Final = await self.transport.set(lock_key(cronjob_id), _encode(owner), ttl=ttl, nx=True)
        if isinstance(set_reply, RedisFailure):
            return set_reply
        if set_reply:
            return Acquired()
        match await self.holder(cronjob_id):
            case None:
                return Vacant()
            case held:
                return held

    async def holder(self, cronjob_id: str) -> HeldBy | None | RedisFailure:
        held: Final = await self.transport.get(lock_key(cronjob_id))
        if held is None or isinstance(held, RedisFailure):
            return held
        return _decode(held)

    async def release(self, cronjob_id: str, owner: str) -> bool | RedisFailure:
        reply: Final = await self.transport.eval(_RELEASE, (lock_key(cronjob_id),), (_encode(owner),))
        return reply if isinstance(reply, RedisFailure) else reply == 1

    async def remove(self, cronjob_id: str) -> bool | RedisFailure:
        deleted: Final = await self.transport.delete(lock_key(cronjob_id))
        return deleted if isinstance(deleted, RedisFailure) else deleted == 1


def _encode(owner: str) -> str:
    return json.dumps(owner)


def _decode(raw: bytes) -> HeldBy | RedisFailure:
    try:
        return HeldBy(_OWNER.validate_json(raw))
    except ValidationError as e:
        return RedisFailure(kind="decode", message=f"pod lock owner is not a JSON string: {e}")
