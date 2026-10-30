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


def refresh_lock_key(user_id: str, server_id: str) -> str:
    return f"mcp:refresh_lock:{user_id}:{server_id}"


@dataclass(frozen=True, slots=True)
class McpRefreshLockIdentity:
    user_id: str
    server_id: str


class McpRefreshLockStore(Protocol):
    async def acquire(self, identity: McpRefreshLockIdentity, owner: str, *, ttl: timedelta) -> bool | RedisFailure: ...

    async def extend(self, identity: McpRefreshLockIdentity, owner: str, *, ttl: timedelta) -> bool | RedisFailure: ...

    async def release(self, identity: McpRefreshLockIdentity, owner: str) -> bool | RedisFailure: ...

    async def is_held(self, identity: McpRefreshLockIdentity) -> bool | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportMcpRefreshLockStore:
    transport: RedisTransport

    async def acquire(self, identity: McpRefreshLockIdentity, owner: str, *, ttl: timedelta) -> bool | RedisFailure:
        return await self.transport.set(refresh_lock_key(identity.user_id, identity.server_id), owner, ttl=ttl, nx=True)

    async def extend(self, identity: McpRefreshLockIdentity, owner: str, *, ttl: timedelta) -> bool | RedisFailure:
        key: Final = refresh_lock_key(identity.user_id, identity.server_id)
        reply: Final = await self.transport.eval(_EXTEND_IF_OWNER, (key,), (owner, ttl_ms(ttl)))
        return reply if isinstance(reply, RedisFailure) else reply == 1

    async def release(self, identity: McpRefreshLockIdentity, owner: str) -> bool | RedisFailure:
        key: Final = refresh_lock_key(identity.user_id, identity.server_id)
        reply: Final = await self.transport.eval(_RELEASE_IF_OWNER, (key,), (owner,))
        return reply if isinstance(reply, RedisFailure) else reply == 1

    async def is_held(self, identity: McpRefreshLockIdentity) -> bool | RedisFailure:
        reply: Final = await self.transport.get(refresh_lock_key(identity.user_id, identity.server_id))
        return reply if isinstance(reply, RedisFailure) else reply is not None
