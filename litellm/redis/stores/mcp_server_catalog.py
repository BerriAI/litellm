import ast
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Literal, Protocol, TypeVar

from pydantic import TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisTransport
from litellm.types.mcp_server.mcp_toolset import MCPToolset

TOOLSET_NOT_FOUND: Final = "__not_found__"

ToolsetNotFound = Literal["__not_found__"]
CachedToolset = MCPToolset | ToolsetNotFound

SUBMITTED_SERVER_IDS: Final = TypeAdapter(tuple[str, ...])
TOOLSET_PERMISSIONS: Final = TypeAdapter(dict[str, tuple[str, ...]])
CACHED_TOOLSET: Final[TypeAdapter[CachedToolset]] = TypeAdapter(CachedToolset)
_VALUE: Final = TypeAdapter(object)

_T = TypeVar("_T")


def submitted_servers_key(user_id: str) -> str:
    return f"byom_submitted_servers:{user_id}"


def toolset_permissions_key(toolset_ids: Sequence[str]) -> str:
    return "toolset_perms:" + ",".join(sorted(toolset_ids))


def toolset_name_key(toolset_name: str) -> str:
    return f"toolset_name:{toolset_name}"


class McpServerCatalogStore(Protocol):
    async def get_submitted_server_ids(self, user_id: str) -> tuple[str, ...] | None | RedisFailure: ...

    async def put_submitted_server_ids(
        self, user_id: str, server_ids: Sequence[str], *, ttl: timedelta
    ) -> bool | None | RedisFailure: ...

    async def invalidate_submitted_server_ids(self, user_id: str) -> int | RedisFailure: ...

    async def get_toolset_permissions(
        self, toolset_ids: Sequence[str]
    ) -> dict[str, tuple[str, ...]] | None | RedisFailure: ...

    async def put_toolset_permissions(
        self, toolset_ids: Sequence[str], permissions: Mapping[str, Sequence[str]], *, ttl: timedelta
    ) -> bool | None | RedisFailure: ...

    async def get_toolset_by_name(self, toolset_name: str) -> CachedToolset | None | RedisFailure: ...

    async def put_toolset_by_name(
        self, toolset_name: str, toolset: MCPToolset | None, *, ttl: timedelta
    ) -> bool | None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportMcpServerCatalogStore:
    transport: RedisTransport

    async def get_submitted_server_ids(self, user_id: str) -> tuple[str, ...] | None | RedisFailure:
        return await self._get(submitted_servers_key(user_id), SUBMITTED_SERVER_IDS)

    async def put_submitted_server_ids(
        self, user_id: str, server_ids: Sequence[str], *, ttl: timedelta
    ) -> bool | None | RedisFailure:
        return await self._put(submitted_servers_key(user_id), json.dumps(list(server_ids)), ttl)

    async def invalidate_submitted_server_ids(self, user_id: str) -> int | RedisFailure:
        return await self.transport.delete(submitted_servers_key(user_id))

    async def get_toolset_permissions(
        self, toolset_ids: Sequence[str]
    ) -> dict[str, tuple[str, ...]] | None | RedisFailure:
        return await self._get(toolset_permissions_key(toolset_ids), TOOLSET_PERMISSIONS)

    async def put_toolset_permissions(
        self, toolset_ids: Sequence[str], permissions: Mapping[str, Sequence[str]], *, ttl: timedelta
    ) -> bool | None | RedisFailure:
        payload: Final = json.dumps({server_id: list(tools) for server_id, tools in permissions.items()})
        return await self._put(toolset_permissions_key(toolset_ids), payload, ttl)

    async def get_toolset_by_name(self, toolset_name: str) -> CachedToolset | None | RedisFailure:
        return await self._get(toolset_name_key(toolset_name), CACHED_TOOLSET)

    async def put_toolset_by_name(
        self, toolset_name: str, toolset: MCPToolset | None, *, ttl: timedelta
    ) -> bool | None | RedisFailure:
        payload: Final = json.dumps(toolset.model_dump(mode="json") if toolset is not None else TOOLSET_NOT_FOUND)
        return await self._put(toolset_name_key(toolset_name), payload, ttl)

    async def _get(self, key: str, adapter: TypeAdapter[_T]) -> _T | None | RedisFailure:
        reply: Final = await self.transport.get(key)
        if reply is None or isinstance(reply, RedisFailure):
            return reply
        return _decode(reply, adapter)

    async def _put(self, key: str, payload: str, ttl: timedelta) -> bool | None | RedisFailure:
        expiry: Final = timedelta(seconds=int(ttl.total_seconds()))
        batch: Final = self.transport.request_batch()
        if batch is not None:
            _ = batch.set_seconds(key, payload, ttl=expiry)
            return None
        return await self.transport.set_seconds(key, payload, ttl=expiry)


def _decode(reply: bytes, adapter: TypeAdapter[_T]) -> _T | RedisFailure:
    try:
        text: Final = reply.decode("utf-8")
        try:
            value: Final = _VALUE.validate_python(json.loads(text))
        except ValueError:
            return adapter.validate_python(ast.literal_eval(text))
        return adapter.validate_python(value)
    except (UnicodeDecodeError, ValidationError, ValueError, SyntaxError) as error:
        return RedisFailure(kind="decode", message=f"Invalid MCP server catalog cache entry: {type(error).__name__}")
