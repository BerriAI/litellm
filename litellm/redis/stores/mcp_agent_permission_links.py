import ast
import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisTransport

AGENT_NO_MCP_PERMISSION_SENTINEL: Final = "__agent_no_mcp_permission__"
_VALUE: Final = TypeAdapter(object)


def agent_permission_link_key(agent_id: str) -> str:
    return f"agent_object_permission_id:{agent_id}"


class McpAgentPermissionLinkStore(Protocol):
    async def get(self, agent_id: str) -> str | None | RedisFailure: ...

    async def fill(self, agent_id: str, permission_id: str | None, *, ttl: timedelta) -> bool | None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportMcpAgentPermissionLinkStore:
    transport: RedisTransport

    async def get(self, agent_id: str) -> str | None | RedisFailure:
        reply: Final = await self.transport.get(agent_permission_link_key(agent_id))
        if reply is None or isinstance(reply, RedisFailure):
            return reply
        try:
            decoded: Final = reply.decode("utf-8")
            try:
                value: Final = _VALUE.validate_json(decoded)
            except ValidationError:
                literal: Final = _VALUE.validate_python(ast.literal_eval(decoded))
                return _permission_id(literal)
            return _permission_id(value)
        except (UnicodeDecodeError, ValueError, SyntaxError):
            return RedisFailure(kind="decode", message="Invalid agent permission link")

    async def fill(self, agent_id: str, permission_id: str | None, *, ttl: timedelta) -> bool | None | RedisFailure:
        seconds: Final = int(ttl.total_seconds())
        key: Final = agent_permission_link_key(agent_id)
        value: Final = json.dumps(permission_id or AGENT_NO_MCP_PERMISSION_SENTINEL)
        expiry: Final = timedelta(seconds=seconds)
        batch: Final = self.transport.request_batch()
        if batch is not None:
            _ = batch.set_seconds(key, value, ttl=expiry)
            return None
        return await self.transport.set_seconds(key, value, ttl=expiry)


def _permission_id(value: object) -> str | RedisFailure:
    if isinstance(value, str):
        return value
    return RedisFailure(kind="decode", message="Agent permission link is not a string")
