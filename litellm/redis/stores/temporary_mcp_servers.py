import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.redis.transport.contract import RedisFailure, RedisTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

KEY_PREFIX: Final = "litellm:mcp:temporary_server"
_STRING: Final = TypeAdapter(str)


def temporary_server_key(server_id: str) -> str:
    return f"{KEY_PREFIX}:{server_id}"


def encode_server(server: MCPServer) -> str | RedisFailure:
    try:
        return _STRING.validate_python(encrypt_value_helper(json.dumps(server.model_dump(mode="json"))), strict=True)
    except Exception:
        return RedisFailure(kind="decode", message="Temporary MCP server encryption failed")


def decode_server(payload: object) -> MCPServer | RedisFailure:
    if not isinstance(payload, str):
        return RedisFailure(kind="decode", message="Temporary MCP server envelope is not a string")
    decrypted: Final = decrypt_value_helper(value=payload, key="temporary_mcp_server", exception_type="debug")
    if decrypted is None:
        return RedisFailure(kind="decode", message="Temporary MCP server decryption failed")
    try:
        return MCPServer.model_validate_json(decrypted)
    except ValidationError:
        return RedisFailure(kind="decode", message="Invalid temporary MCP server document")


class TemporaryMCPServersStore(Protocol):
    async def put(self, server: MCPServer, *, ttl: timedelta) -> None | RedisFailure: ...

    async def get(self, server_id: str) -> MCPServer | None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportTemporaryMCPServersStore:
    transport: RedisTransport

    async def put(self, server: MCPServer, *, ttl: timedelta) -> None | RedisFailure:
        payload: Final = encode_server(server)
        if isinstance(payload, RedisFailure):
            return payload
        result: Final = await self.transport.set(temporary_server_key(server.server_id), json.dumps(payload), ttl=ttl)
        return result if isinstance(result, RedisFailure) else None

    async def get(self, server_id: str) -> MCPServer | None | RedisFailure:
        raw: Final = await self.transport.get(temporary_server_key(server_id))
        if raw is None or isinstance(raw, RedisFailure):
            return raw
        try:
            return decode_server(_STRING.validate_json(raw, strict=True))
        except ValidationError:
            return RedisFailure(kind="decode", message="Invalid temporary MCP server envelope")
