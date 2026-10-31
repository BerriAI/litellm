from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from litellm.redis.transport.contract import EvalReply, RedisFailure, RedisTransport

_ADD_SPEND: Final = """
local key = KEYS[1]
local amount = ARGV[1]
local ttl = tonumber(ARGV[2])

local existed = redis.call('EXISTS', key)
local new_val = redis.call('INCRBYFLOAT', key, amount)
if existed == 0 then
    redis.call('EXPIRE', key, ttl)
end

return new_val
"""


def spend_key(session_id: str) -> str:
    return f"{{session_budget:{session_id}}}:spend"


class SessionBudgetStore(Protocol):
    async def add_spend(self, session_id: str, amount: float, *, ttl: timedelta) -> float | RedisFailure: ...

    async def spend(self, session_id: str) -> float | None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportSessionBudgetStore:
    transport: RedisTransport

    async def add_spend(self, session_id: str, amount: float, *, ttl: timedelta) -> float | RedisFailure:
        reply: Final = await self.transport.eval(
            _ADD_SPEND, (spend_key(session_id),), (str(amount), ttl // timedelta(seconds=1))
        )
        return reply if isinstance(reply, RedisFailure) else _amount(reply)

    async def spend(self, session_id: str) -> float | None | RedisFailure:
        raw: Final = await self.transport.get(spend_key(session_id))
        if raw is None or isinstance(raw, RedisFailure):
            return raw
        return _amount(raw)


def _amount(reply: EvalReply) -> float | RedisFailure:
    if not isinstance(reply, (bytes, int)):
        return RedisFailure(kind="decode", message=f"session spend is not a number: {reply!r}")
    try:
        return float(reply)
    except ValueError:
        return RedisFailure(kind="decode", message=f"session spend is not a number: {reply!r}")
