import hashlib
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol, TypeAlias

from typing_extensions import assert_never

from litellm.redis.transport.contract import EvalReply, RedisFailure, RedisTransport

_CLAIM: Final = """
local count = redis.call("incrbyfloat", KEYS[1], ARGV[1])
if redis.call("ttl", KEYS[1]) < 0 then
    redis.call("expire", KEYS[1], ARGV[2])
end
return count
"""


@dataclass(frozen=True, slots=True)
class CodeMarker:
    jti: str


@dataclass(frozen=True, slots=True)
class FlowMarker:
    jti: str


@dataclass(frozen=True, slots=True)
class RefreshMarker:
    jti: str


@dataclass(frozen=True, slots=True)
class RevokedRefreshFamilyMarker:
    family: str


@dataclass(frozen=True, slots=True)
class IdJagMarker:
    issuer: str
    jti: str


McpSingleUseMarker: TypeAlias = CodeMarker | FlowMarker | RefreshMarker | RevokedRefreshFamilyMarker | IdJagMarker


def marker_key(marker: McpSingleUseMarker) -> str:
    match marker:
        case CodeMarker(jti):
            return f"mcp_gateway_dcr_code_used:{jti}"
        case FlowMarker(jti):
            return f"mcp_gateway_dcr_flow_used:{jti}"
        case RefreshMarker(jti):
            return f"mcp_gateway_dcr_refresh_used:{jti}"
        case RevokedRefreshFamilyMarker(family):
            return f"mcp_gateway_dcr_refresh_family_revoked:{family}"
        case IdJagMarker(issuer, jti):
            return "mcp_gateway_dcr:id_jag:" + hashlib.sha256(f"{issuer}\n{jti}".encode()).hexdigest()
        case _:
            assert_never(marker)


class McpSingleUseStore(Protocol):
    async def claim(self, marker: McpSingleUseMarker, *, ttl: timedelta) -> int | RedisFailure: ...

    async def peek(self, marker: McpSingleUseMarker) -> bool | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportMcpSingleUseStore:
    transport: RedisTransport

    async def claim(self, marker: McpSingleUseMarker, *, ttl: timedelta) -> int | RedisFailure:
        reply: Final = await self.transport.eval(_CLAIM, (marker_key(marker),), (1, ttl // timedelta(seconds=1)))
        return reply if isinstance(reply, RedisFailure) else _count(reply)

    async def peek(self, marker: McpSingleUseMarker) -> bool | RedisFailure:
        reply: Final = await self.transport.get(marker_key(marker))
        return reply if isinstance(reply, RedisFailure) else reply is not None


def _count(reply: EvalReply) -> int | RedisFailure:
    if not isinstance(reply, bytes):
        return RedisFailure(kind="decode", message=f"single-use count is not a bulk string: {reply!r}")
    try:
        return int(float(reply))
    except (ValueError, OverflowError):
        return RedisFailure(kind="decode", message=f"single-use count is not a number: {reply!r}")
