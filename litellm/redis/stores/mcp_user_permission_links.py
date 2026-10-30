import ast
import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisTransport

USER_NO_MCP_PERMISSION_SENTINEL: Final = "__user_no_mcp_permission__"
_VALUE: Final = TypeAdapter(object)


def user_permission_link_key(user_id: str) -> str:
    return f"user_object_permission_id:{user_id}"


class McpUserPermissionLinkStore(Protocol):
    async def get(self, user_id: str) -> str | None | RedisFailure: ...

    async def fill(self, user_id: str, permission_id: str | None, *, ttl: timedelta) -> bool | None | RedisFailure: ...

    async def invalidate(self, user_id: str) -> int | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportMcpUserPermissionLinkStore:
    transport: RedisTransport

    async def get(self, user_id: str) -> str | None | RedisFailure:
        reply: Final = await self.transport.get(user_permission_link_key(user_id))
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
            return RedisFailure(kind="decode", message="Invalid user permission link")

    async def fill(self, user_id: str, permission_id: str | None, *, ttl: timedelta) -> bool | None | RedisFailure:
        seconds: Final = int(ttl.total_seconds())
        if seconds <= 0:
            return RedisFailure(kind="server", message="invalid expire time in 'set' command")
        key: Final = user_permission_link_key(user_id)
        value: Final = json.dumps(permission_id or USER_NO_MCP_PERMISSION_SENTINEL)
        expiry: Final = timedelta(seconds=seconds)
        batch: Final = self.transport.request_batch()
        if batch is not None:
            _ = batch.set(key, value, ttl=expiry)
            return None
        return await self.transport.set(key, value, ttl=expiry)

    async def invalidate(self, user_id: str) -> int | RedisFailure:
        return await self.transport.delete(user_permission_link_key(user_id))


def _permission_id(value: object) -> str | RedisFailure:
    if isinstance(value, str):
        return value
    return RedisFailure(kind="decode", message="User permission link is not a string")
