import ast
import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import TypeAdapter

from litellm.constants import MCP_PER_USER_TOKEN_REDIS_KEY_PREFIX
from litellm.redis.transport.contract import RedisFailure, RedisTransport

_DECODED: Final = TypeAdapter(object)


def per_user_token_key(user_id: str, server_id: str) -> str:
    return f"{MCP_PER_USER_TOKEN_REDIS_KEY_PREFIX}:{user_id}:{server_id}"


def _decode(raw: bytes) -> str | None | RedisFailure:
    text: Final = raw.decode("utf-8", errors="replace")
    try:
        value: Final = _DECODED.validate_python(json.loads(text))
    except ValueError:
        try:
            value_literal: Final = _DECODED.validate_python(ast.literal_eval(text))
        except (ValueError, SyntaxError) as error:
            return RedisFailure(kind="decode", message=str(error))
        return value_literal if isinstance(value_literal, str) else None
    return value if isinstance(value, str) else None


class McpPerUserTokenStore(Protocol):
    async def get(self, user_id: str, server_id: str) -> str | None | RedisFailure: ...

    async def put(self, user_id: str, server_id: str, blob: str, *, ttl: timedelta) -> bool | RedisFailure: ...

    async def delete(self, user_id: str, server_id: str) -> int | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportMcpPerUserTokenStore:
    transport: RedisTransport

    async def get(self, user_id: str, server_id: str) -> str | None | RedisFailure:
        raw: Final = await self.transport.get(per_user_token_key(user_id, server_id))
        if raw is None or isinstance(raw, RedisFailure):
            return raw
        return _decode(raw)

    async def put(self, user_id: str, server_id: str, blob: str, *, ttl: timedelta) -> bool | RedisFailure:
        return await self.transport.set_seconds(per_user_token_key(user_id, server_id), json.dumps(blob), ttl=ttl)

    async def delete(self, user_id: str, server_id: str) -> int | RedisFailure:
        return await self.transport.delete(per_user_token_key(user_id, server_id))
