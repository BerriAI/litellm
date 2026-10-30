from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from litellm.redis.transport.contract import RedisFailure, RedisTransport

_INCREMENT: Final = """
local key = KEYS[1]
local ttl = tonumber(ARGV[1])

local current = redis.call('INCR', key)
if current == 1 then
    redis.call('EXPIRE', key, ttl)
end

return current
"""


def count_key(session_id: str) -> str:
    return f"{{session_iterations:{session_id}}}:count"


class IterationCountStore(Protocol):
    async def increment(self, session_id: str, *, ttl: timedelta) -> int | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportIterationCountStore:
    transport: RedisTransport

    async def increment(self, session_id: str, *, ttl: timedelta) -> int | RedisFailure:
        reply: Final = await self.transport.eval(_INCREMENT, (count_key(session_id),), (ttl // timedelta(seconds=1),))
        if isinstance(reply, (RedisFailure, int)):
            return reply
        return RedisFailure(kind="decode", message=f"session iteration count is not an integer: {reply!r}")
