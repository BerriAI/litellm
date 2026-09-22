from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, ParamSpec, TypeVar

if TYPE_CHECKING:
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

_P: Final = ParamSpec("_P")
_R: Final = TypeVar("_R")


@dataclass(slots=True)
class _CatalogScope:
    servers: Mapping[str, MCPServer]
    owner_task_id: int
    active: bool = True


class TargetCatalog:
    def __init__(self, manager: MCPServerManager) -> None:
        self._manager = manager
        self._reload_lock = asyncio.Lock()
        self._scope: ContextVar[_CatalogScope | None] = ContextVar("mcp_catalog_scope", default=None)

    @property
    def current(self) -> Mapping[str, MCPServer] | None:
        scope: Final = self._scope.get()
        return scope.servers if scope is not None and scope.active else None

    async def refresh(self) -> None:
        async with self._reload_lock:
            await self._manager._reload_servers_from_database()  # pyright: ignore[reportPrivateUsage]  # existing staged loader

    async def list(self) -> Mapping[str, MCPServer]:
        from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # runtime proxy dependency

        scope: Final = self._scope.get()
        if scope is not None and scope.active and scope.owner_task_id == id(asyncio.current_task()):
            return scope.servers
        async with self._reload_lock:
            if prisma_client is not None:
                try:
                    await self._manager._reload_servers_from_database(reuse_unchanged=True)  # pyright: ignore[reportPrivateUsage]  # existing staged loader
                except Exception as exc:  # noqa: BLE001  # never serve an unverified database snapshot
                    from fastapi import HTTPException  # noqa: PLC0415  # optional proxy dependency

                    raise HTTPException(
                        status_code=503, detail="MCP server configuration could not be refreshed"
                    ) from exc
            return MappingProxyType(self._manager.config_mcp_servers | self._manager.registry)

    @asynccontextmanager
    async def operation(self) -> AsyncIterator[Mapping[str, MCPServer]]:
        existing: Final = self._scope.get()
        if existing is not None and existing.active and existing.owner_task_id == id(asyncio.current_task()):
            yield existing.servers
            return
        scope: Final = _CatalogScope(await self.list(), id(asyncio.current_task()))
        token: Final = self._scope.set(scope)
        try:
            yield scope.servers
        finally:
            scope.active = False
            self._scope.reset(token)

    async def resolve(self, lookup: str, client_ip: str | None = None) -> MCPServer | None:
        async with self.operation():
            return self._manager.get_mcp_server_by_name(
                lookup, client_ip=client_ip
            ) or self._manager.get_mcp_server_by_id(lookup, client_ip=client_ip)


def with_mcp_catalog(function: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R]]:
    @wraps(function)
    async def wrapped(
        *args: _P.args,
        **kwargs: _P.kwargs,  # kwargs-ok: ParamSpec preserves each wrapped endpoint keyword contract
    ) -> _R:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (  # noqa: PLC0415  # manager imports catalog
            global_mcp_server_manager,
        )

        async with global_mcp_server_manager.catalog.operation():
            return await function(*args, **kwargs)

    return wrapped
