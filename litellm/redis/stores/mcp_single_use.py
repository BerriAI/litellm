from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from litellm.redis.transport.contract import EvalReply, RedisFailure, RedisTransport

_CLAIM: Final = """
local count = redis.call("incrbyfloat", KEYS[1], ARGV[1])
if redis.call("ttl", KEYS[1]) < 0 then
    redis.call("expire", KEYS[1], ARGV[2])
end
return count
"""


class McpSingleUseStore(Protocol):
    async def claim(self, key: str, *, ttl: timedelta) -> int | RedisFailure: ...

    async def peek(self, key: str) -> bool | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportMcpSingleUseStore:
    transport: RedisTransport

    async def claim(self, key: str, *, ttl: timedelta) -> int | RedisFailure:
        reply: Final = await self.transport.eval(_CLAIM, (key,), (1, ttl // timedelta(seconds=1)))
        return reply if isinstance(reply, RedisFailure) else _count(reply)

    async def peek(self, key: str) -> bool | RedisFailure:
        reply: Final = await self.transport.get(key)
        return reply if isinstance(reply, RedisFailure) else reply is not None


def _count(reply: EvalReply) -> int | RedisFailure:
    if not isinstance(reply, bytes):
        return RedisFailure(kind="decode", message=f"single-use count is not a bulk string: {reply!r}")
    try:
        return int(float(reply))
    except ValueError:
        return RedisFailure(kind="decode", message=f"single-use count is not a number: {reply!r}")
