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
class ConnectFlowUsed:
    jti: str


@dataclass(frozen=True, slots=True)
class AuthCodeUsed:
    jti: str


@dataclass(frozen=True, slots=True)
class RefreshTokenUsed:
    jti: str


@dataclass(frozen=True, slots=True)
class RefreshFamilyRevoked:
    family: str


@dataclass(frozen=True, slots=True)
class AssertionRedeemed:
    issuer: str
    jti: str


SingleUseMarker: TypeAlias = (
    ConnectFlowUsed | AuthCodeUsed | RefreshTokenUsed | RefreshFamilyRevoked | AssertionRedeemed
)


def marker_key(marker: SingleUseMarker) -> str:
    match marker:
        case ConnectFlowUsed(jti=jti):
            return f"mcp_gateway_dcr_flow_used:{jti}"
        case AuthCodeUsed(jti=jti):
            return f"mcp_gateway_dcr_code_used:{jti}"
        case RefreshTokenUsed(jti=jti):
            return f"mcp_gateway_dcr_refresh_used:{jti}"
        case RefreshFamilyRevoked(family=family):
            return f"mcp_gateway_dcr_refresh_family_revoked:{family}"
        case AssertionRedeemed(issuer=issuer, jti=jti):
            return "mcp_gateway_dcr:id_jag:" + hashlib.sha256(f"{issuer}\n{jti}".encode()).hexdigest()
        case _:
            assert_never(marker)


class McpSingleUseStore(Protocol):
    async def claim(self, marker: SingleUseMarker, *, ttl: timedelta) -> int | RedisFailure: ...

    async def peek(self, marker: SingleUseMarker) -> bool | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportMcpSingleUseStore:
    transport: RedisTransport

    async def claim(self, marker: SingleUseMarker, *, ttl: timedelta) -> int | RedisFailure:
        reply: Final = await self.transport.eval_direct(_CLAIM, (marker_key(marker),), (1, ttl // timedelta(seconds=1)))
        return reply if isinstance(reply, RedisFailure) else _count(reply)

    async def peek(self, marker: SingleUseMarker) -> bool | RedisFailure:
        reply: Final = await self.transport.get(marker_key(marker))
        return reply if isinstance(reply, RedisFailure) else reply is not None


def _count(reply: EvalReply) -> int | RedisFailure:
    if not isinstance(reply, bytes):
        return RedisFailure(kind="decode", message=f"single-use count is not a bulk string: {reply!r}")
    try:
        return int(float(reply))
    except ValueError:
        return RedisFailure(kind="decode", message=f"single-use count is not a number: {reply!r}")
