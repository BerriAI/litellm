from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from litellm.redis.transport.contract import RedisFailure, RedisTransport, ttl_ms

_RELEASE_IF_OWNER: Final = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"
)
_EXTEND_IF_OWNER: Final = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('pexpire', KEYS[1], ARGV[2]) else return 0 end"
)


@dataclass(frozen=True, slots=True)
class RefreshLockTarget:
    user_id: str
    server_id: str


def _lock_key(target: RefreshLockTarget) -> str:
    return f"mcp:refresh_lock:{target.user_id}:{target.server_id}"


class McpRefreshLockStore(Protocol):
    async def acquire(self, target: RefreshLockTarget, owner: str, *, ttl: timedelta) -> bool | RedisFailure: ...

    async def extend(self, target: RefreshLockTarget, owner: str, *, ttl: timedelta) -> bool | RedisFailure: ...

    async def release(self, target: RefreshLockTarget, owner: str) -> bool | RedisFailure: ...

    async def is_held(self, target: RefreshLockTarget) -> bool | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportMcpRefreshLockStore:
    transport: RedisTransport

    async def acquire(self, target: RefreshLockTarget, owner: str, *, ttl: timedelta) -> bool | RedisFailure:
        return await self.transport.set(_lock_key(target), owner, ttl=ttl, nx=True)

    async def extend(self, target: RefreshLockTarget, owner: str, *, ttl: timedelta) -> bool | RedisFailure:
        reply: Final = await self.transport.eval_direct(_EXTEND_IF_OWNER, (_lock_key(target),), (owner, ttl_ms(ttl)))
        return reply if isinstance(reply, RedisFailure) else reply == 1

    async def release(self, target: RefreshLockTarget, owner: str) -> bool | RedisFailure:
        reply: Final = await self.transport.eval_direct(_RELEASE_IF_OWNER, (_lock_key(target),), (owner,))
        return reply if isinstance(reply, RedisFailure) else reply == 1

    async def is_held(self, target: RefreshLockTarget) -> bool | RedisFailure:
        reply: Final = await self.transport.get(_lock_key(target))
        return reply if isinstance(reply, RedisFailure) else reply is not None
