"""Authoritative MCP catalog snapshots shared by legacy lookup adapters."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import wraps
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, ParamSpec, TypeVar

from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from pydantic import BaseModel

    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
    from litellm.types.mcp_server.mcp_server_manager import MCPServer
    from litellm.types.mcp_server.tool_registry import MCPTool

_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    servers: Mapping[str, MCPServer]
    identity: str
    tools: Mapping[str, MCPTool]
    routing: dict[str, str]


def _configuration_identity(server: MCPServer) -> str:
    return json.dumps(
        server.model_dump(
            mode="json",
            exclude=frozenset(("short_prefix", "scopes", "authorization_url", "token_url", "registration_url"))
            | (frozenset() if server.issuer_is_anchored else frozenset(("issuer",))),
        ),
        sort_keys=True,
    )


def _check_oauth_revision(selected: MCPServer, candidate: MCPServer | None) -> None:
    from fastapi import HTTPException

    if candidate is None or _configuration_identity(selected) != _configuration_identity(candidate):
        raise HTTPException(status_code=503, detail="OAuth metadata discovery changed repeatedly; retry shortly")


def _snapshot(manager: MCPServerManager, database_identity: str) -> CatalogSnapshot:
    from litellm.proxy._experimental.mcp_server.tool_registry import global_mcp_tool_registry

    servers: Final = manager.config_mcp_servers | manager.registry
    detached: Final = MappingProxyType({key: value.model_copy(deep=True) for key, value in servers.items()})
    serialized: Final = json.dumps(
        (
            database_identity,
            tuple(sorted(manager.registry)),
            tuple((key, _configuration_identity(value)) for key, value in sorted(manager.config_mcp_servers.items())),
        ),
        sort_keys=True,
    )
    return CatalogSnapshot(
        detached,
        hashlib.sha256(serialized.encode()).hexdigest(),
        MappingProxyType(dict(global_mcp_tool_registry.published_tools)),
        dict(manager.published_tool_routes),
    )


class TargetCatalog:
    def __init__(self, manager: MCPServerManager) -> None:
        self.manager = manager
        self._refresh_lock = asyncio.Lock()
        self._database_identity = ""
        self._arrival_ticket = 0
        self._completed_ticket = 0
        self._shared_snapshot: CatalogSnapshot | None = None
        self._warned_shadowed_config_server_ids: frozenset[str] = frozenset()
        self._warned_capturing_config_server_ids: frozenset[str] = frozenset()
        self._operation: ContextVar[tuple[CatalogSnapshot, asyncio.Event, int] | None] = ContextVar(
            "mcp_catalog_snapshot", default=None
        )
        self._staged_routing: ContextVar[tuple[dict[str, str], asyncio.Event] | None] = ContextVar(
            "mcp_catalog_routing", default=None
        )

    def current(self) -> CatalogSnapshot | None:
        scoped: Final = self._operation.get()
        return scoped[0] if scoped is not None and not scoped[1].is_set() else None

    def routing(self) -> dict[str, str]:
        staged: Final = self._staged_routing.get()
        if staged is not None and not staged[1].is_set():
            return staged[0]
        snapshot: Final = self.current()
        return snapshot.routing if snapshot is not None else self.manager.published_tool_routes

    def registry(self) -> Mapping[str, MCPServer]:
        snapshot: Final = self.current()
        return snapshot.servers if snapshot is not None else self.manager.config_mcp_servers | self.manager.registry

    async def _fresh_snapshot(self) -> CatalogSnapshot:
        from fastapi import HTTPException

        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            return _snapshot(self.manager, self._database_identity)
        from litellm.proxy.proxy_server import should_load_db_object

        if not should_load_db_object("mcp"):
            return _snapshot(self.manager, self._database_identity)
        self._arrival_ticket += 1
        arrival: Final = self._arrival_ticket
        async with self._refresh_lock:
            if arrival > self._completed_ticket:
                try:
                    await self._publish_refresh(reuse_unchanged=True)
                except Exception as exc:
                    raise HTTPException(
                        status_code=503, detail="MCP server configuration could not be refreshed"
                    ) from exc
            if self._shared_snapshot is None:
                raise HTTPException(status_code=503, detail="MCP server configuration could not be refreshed")
            return self._shared_snapshot

    async def list(self) -> Mapping[str, MCPServer]:
        async with self.operation() as snapshot:
            return snapshot.servers

    def assert_current(self, server: MCPServer) -> None:
        from fastapi import HTTPException

        snapshot: Final = self.current()
        if snapshot is None or server.server_id not in snapshot.servers:
            return
        expected: Final = snapshot.servers[server.server_id]
        registered: Final = self.manager.registry.get(server.server_id) or self.manager.config_mcp_servers.get(
            server.server_id
        )
        if (
            registered is None
            or registered.updated_at != expected.updated_at
            or server.updated_at != expected.updated_at
        ):
            raise HTTPException(status_code=503, detail="MCP server configuration changed; retry the operation")

    @asynccontextmanager
    async def operation(self) -> AsyncIterator[CatalogSnapshot]:
        current: Final = self.current()
        scoped: Final = self._operation.get()
        if current is not None and scoped is not None and scoped[2] == id(asyncio.current_task()):
            yield current
            return
        from litellm.proxy._experimental.mcp_server.tool_registry import global_mcp_tool_registry

        shared: Final = await self._fresh_snapshot()
        snapshot: Final = replace(
            shared,
            servers=MappingProxyType({key: value.model_copy(deep=True) for key, value in shared.servers.items()}),
            routing=dict(shared.routing),
        )
        closed: Final = asyncio.Event()
        token: Final = self._operation.set((snapshot, closed, id(asyncio.current_task())))
        try:
            with global_mcp_tool_registry.catalog_scope(snapshot.tools):
                yield snapshot
        finally:
            closed.set()
            self._operation.reset(token)
            self._retain_discovered_routing(snapshot)

    def _retain_discovered_routing(self, snapshot: CatalogSnapshot) -> None:
        self.manager.published_tool_routes = self.manager.published_tool_routes | self._unchanged_routing(
            snapshot.servers, snapshot.routing
        )

    def _unchanged_routing(
        self, servers: Mapping[str, MCPServer], routing: Mapping[str, str]
    ) -> MappingProxyType[str, str]:
        from litellm.proxy._experimental.mcp_server.utils import normalize_server_name

        current: Final = self.manager.config_mcp_servers | self.manager.registry
        unchanged_owners: Final = frozenset(
            owner
            for key, server in servers.items()
            if (candidate := current.get(key)) is not None
            and _configuration_identity(candidate) == _configuration_identity(server)
            for owner in self.manager.owned_mapping_values(server)
        )
        return MappingProxyType(
            {name: owner for name, owner in routing.items() if normalize_server_name(owner) in unchanged_owners}
        )

    async def resolve(self, identifier: str, client_ip: str | None = None) -> MCPServer | None:
        async with self.operation():
            return self.manager.get_mcp_server_by_name(
                identifier, client_ip=client_ip
            ) or self.manager.get_mcp_server_by_id(identifier, client_ip=client_ip)

    async def resolve_oauth_metadata(
        self,
        server: MCPServer,
        resolve: Callable[[MCPServer], Awaitable[MCPServer]],
    ) -> MCPServer:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import oauth_endpoints_unresolved

        snapshot: Final = self.current()
        selected: Final = snapshot.servers.get(server.server_id) if snapshot is not None else None
        if selected is None:
            return await resolve(server)
        if not oauth_endpoints_unresolved(selected):
            return selected
        registered: Final = self.manager.registry.get(server.server_id) or self.manager.config_mcp_servers.get(
            server.server_id
        )
        self.assert_current(server)
        _check_oauth_revision(selected, registered)
        resolved: Final = await resolve(selected)
        _check_oauth_revision(selected, resolved)
        return resolved

    async def reload(self) -> None:
        async with self._refresh_lock:
            await self._publish_refresh()

    async def _publish_refresh(self, *, reuse_unchanged: bool = False) -> None:
        covered: Final = self._arrival_ticket
        token: Final = self._operation.set(None)
        try:
            await self._reload_and_publish(reuse_unchanged=reuse_unchanged)
        except Exception:
            self._shared_snapshot = None
            self._completed_ticket = covered
            raise
        else:
            self._shared_snapshot = _snapshot(self.manager, self._database_identity)
            self._completed_ticket = covered
        finally:
            self._operation.reset(token)

    async def _reload_and_publish(self, *, reuse_unchanged: bool) -> None:
        from litellm.proxy._experimental.mcp_server.tool_registry import global_mcp_tool_registry
        from litellm.proxy._experimental.mcp_server.utils import normalize_server_name

        previous_config: Final = MappingProxyType(
            {key: value.model_copy(deep=True) for key, value in self.manager.config_mcp_servers.items()}
        )
        live_registry: Final = self.manager.registry
        previous_servers: Final = previous_config | MappingProxyType(
            {key: value.model_copy(deep=True) for key, value in live_registry.items()}
        )
        config_identities: Final = MappingProxyType(
            {key: _configuration_identity(value) for key, value in previous_config.items()}
        )
        staged_config: Final = MappingProxyType(
            {key: value.model_copy(deep=True) for key, value in previous_config.items()}
        )
        await self.manager.hydrate_config_servers_dcr_clients(tuple(staged_config.values()))
        initial_routing: Final = MappingProxyType(dict(self.manager.published_tool_routes))
        staged_routing: Final = dict(initial_routing)
        closed: Final = asyncio.Event()
        routing_token: Final = self._staged_routing.set((staged_routing, closed))
        initial_tools: Final = MappingProxyType(dict(global_mcp_tool_registry.published_tools))
        try:
            with global_mcp_tool_registry.catalog_scope(initial_tools) as staged_tools:
                await self._reload(reuse_unchanged=reuse_unchanged)
                refreshed_openapi_owners: Final = frozenset(
                    owner
                    for server in self.manager.registry.values()
                    if server.spec_path and server is not live_registry.get(server.server_id)
                    for owner in self.manager.owned_mapping_values(server)
                )
                live_routes: Final = self._unchanged_routing(
                    previous_servers,
                    MappingProxyType(
                        {
                            name: owner
                            for name, owner in self.manager.published_tool_routes.items()
                            if normalize_server_name(owner) not in refreshed_openapi_owners
                        }
                    ),
                )
                concurrent_routes: Final = self._unchanged_routing(
                    self.manager.config_mcp_servers | live_registry,
                    MappingProxyType(
                        {
                            name: owner
                            for name, owner in self.manager.published_tool_routes.items()
                            if initial_routing.get(name) != owner
                            and (name not in staged_routing or staged_routing.get(name) == initial_routing.get(name))
                        }
                    ),
                )
                removed_routes: Final = initial_routing.keys() - self.manager.published_tool_routes.keys()
                retained_staged_routes: Final = self._unchanged_routing(
                    previous_config | self.manager.registry,
                    MappingProxyType(
                        {
                            name: owner
                            for name, owner in staged_routing.items()
                            if name not in removed_routes or owner != initial_routing[name]
                        }
                    ),
                )
                concurrent_tools: Final = MappingProxyType(
                    {
                        name: tool
                        for name, tool in global_mcp_tool_registry.published_tools.items()
                        if (
                            name in live_routes
                            or name in concurrent_routes
                            or (name not in initial_routing and name not in self.manager.published_tool_routes)
                        )
                        and initial_tools.get(name) is staged_tools.get(name)
                    }
                )
                self.manager.config_mcp_servers = {
                    key: value.model_copy(
                        update=staged_config[key].model_dump(
                            include=frozenset(("client_id", "client_secret", "token_endpoint_auth_method"))
                        )
                    )
                    if config_identities.get(key) == _configuration_identity(value)
                    else value
                    for key, value in self.manager.config_mcp_servers.items()
                }
                global_mcp_tool_registry.tools = (
                    MappingProxyType(
                        {
                            name: tool
                            for name, tool in staged_tools.items()
                            if name in global_mcp_tool_registry.published_tools or tool is not initial_tools.get(name)
                        }
                    )
                    | concurrent_tools
                )
                self.manager.published_tool_routes = live_routes | retained_staged_routes | concurrent_routes
        finally:
            closed.set()
            self._staged_routing.reset(routing_token)

    async def _reload(self, *, reuse_unchanged: bool) -> None:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            carry_forward_resolved_oauth_endpoints,
            config_ids_capturing_db_identifiers,
            oauth_endpoints_unresolved,
            warn_on_server_name_fields,
        )
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            get_prisma_client_or_throw,
        )

        verbose_logger.debug("Loading MCP servers from database into registry...")

        prisma_client: Final = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        # Load only "active", legacy "approved", and NULL (no approval workflow) rows.
        # Pending/rejected servers are excluded at the DB level so we never load them.
        from litellm.proxy._experimental.mcp_server.db import LiteLLM_MCPServerTable, get_runtime_mcp_server_rows

        raw_rows: Final[Sequence[BaseModel]] = await get_runtime_mcp_server_rows(prisma_client)
        database_identity: Final = hashlib.sha256(
            json.dumps(
                tuple(sorted(json.dumps(row.model_dump(mode="json"), sort_keys=True, default=str) for row in raw_rows))
            ).encode()
        ).hexdigest()
        verbose_logger.info("Found %s MCP servers in database", len(raw_rows))

        previous_registry: Final = self.manager.registry
        new_registry: Final[dict[str, MCPServer]] = {}

        # Stage one: build every server.  Stage two assigns short prefixes
        # against the *full* set so dedup is deterministic regardless of
        # iteration order.
        for row in raw_rows:
            try:
                server = LiteLLM_MCPServerTable.model_validate(row.model_dump())
                existing_server = previous_registry.get(server.server_id)

                if (
                    existing_server is not None
                    and (reuse_unchanged or not existing_server.spec_path)
                    and existing_server.updated_at is not None
                    and server.updated_at is not None
                    and existing_server.updated_at == server.updated_at
                    and (
                        self.manager.oauth_discovery_slot(server.server_id) is not None
                        or not oauth_endpoints_unresolved(existing_server)
                    )
                ):
                    # Re-use existing server instance to avoid re-running build_mcp_server_from_table()
                    # which can perform network discovery for OAuth2 servers.
                    new_registry[server.server_id] = existing_server
                    continue

                warn_on_server_name_fields(
                    server_id=server.server_id,
                    alias=getattr(server, "alias", None),
                    server_name=getattr(server, "server_name", None),
                )
                verbose_logger.debug("Building server from DB: %s (%s)", server.server_id, server.server_name)
                # raw_rows come straight from the DB, so their global env var
                # values (like credentials) are still encrypted here, unlike the
                # already-decrypted records add_server/update_server are handed.
                # Decrypt them while building the registry entry.
                new_server = await self.manager.build_mcp_server_from_table(
                    server, env_vars_are_encrypted=True, register_oauth_discovery=False
                )
                # Carry the cached short_prefix from the previous registry entry
                # (if any) so the prefix is stable across reloads.
                if existing_server is not None and existing_server.short_prefix:
                    new_server.short_prefix = existing_server.short_prefix
                carry_forward_resolved_oauth_endpoints(new_server=new_server, previous_server=existing_server)
                new_registry[server.server_id] = new_server
            except Exception as e:
                verbose_logger.exception(
                    "Skipping MCP server %s (%s) during DB reload: %s",
                    getattr(row, "server_id", None),
                    getattr(row, "alias", None),
                    e,
                )

        # Assign short prefixes against the full candidate set without
        # publishing the staged registry to concurrent callers.
        registered_registry: Final[dict[str, MCPServer]] = {}
        for server_id, new_server in new_registry.items():
            try:
                if new_server is not previous_registry.get(server_id):
                    self.manager.assign_unique_short_prefix(new_server, registry=new_registry)
                # Register OpenAPI tools *after* the final short prefix is assigned
                # so the tools are stored in the global registry under the same
                # prefix that lookups will use.
                if new_server is not previous_registry.get(server_id):
                    if previous_server := previous_registry.get(server_id):
                        self.manager.remove_server_tool_routing(previous_server)
                    await self.manager.maybe_register_openapi_tools(new_server, initialize_mapping=False)
                registered_registry[server_id] = new_server
            except Exception as e:
                self.manager.remove_server_tool_routing(new_server)
                verbose_logger.exception(
                    "Skipping MCP server %s (%s) during DB reload: %s",
                    new_server.server_id,
                    getattr(new_server, "alias", None),
                    e,
                )

        dropped_registry_keys: Final = previous_registry.keys() - registered_registry.keys()
        for registry_key in dropped_registry_keys:
            self.manager.remove_server_tool_routing(previous_registry[registry_key])
            self.manager.invalidate_oauth_discovery_state(previous_registry[registry_key].server_id)

        for server_id in previous_registry.keys() | registered_registry.keys():
            if previous_registry.get(server_id) != registered_registry.get(server_id):
                self.manager.invalidate_discovery_lists(server_id)
                self.manager.invalidate_oauth_discovery_state(server_id)
        self._database_identity = database_identity
        self.manager.registry = registered_registry
        # A discovery task may have published into ``previous_registry`` while
        # this replacement was being staged. Reconcile every published entry
        # synchronously after the swap so a lost publication cannot also leave
        # the replacement unresolved with no retry slot.
        registered_servers: Final = tuple(registered_registry.values())
        self.manager.reconcile_oauth_discovery_slots_for_servers(registered_servers)
        self.manager.prime_oauth_metadata_discovery_for_servers(registered_servers)

        verbose_logger.debug("MCP registry refreshed (%s servers in registry)", len(registered_registry))

        # get_registry() is ``config_mcp_servers | registry``, so a database row sharing an id with a
        # config.yaml server hides that server everywhere. Only reachable once an operator pins
        # ``server_id`` in config.yaml; say so rather than letting the server disappear silently.
        shadowed_config_server_ids: Final = frozenset(
            self.manager.config_mcp_servers.keys() & registered_registry.keys()
        )
        if shadowed_config_server_ids and shadowed_config_server_ids != self._warned_shadowed_config_server_ids:
            verbose_logger.warning(
                "config.yaml MCP server_id(s) %s are also database-backed MCP servers. The database "
                "entry takes precedence, so the config.yaml server is unreachable. Give the config "
                "entry a different server_id.",
                ", ".join(sorted(shadowed_config_server_ids)),
            )
        self._warned_shadowed_config_server_ids = shadowed_config_server_ids

        # The mirror image of the block above: a config server_id that is a database server's name
        # answers that server's grants instead, because ids are matched before names.
        capturing_config_server_ids: Final = config_ids_capturing_db_identifiers(
            self.manager.config_mcp_servers.keys(), registered_registry.values()
        )
        if capturing_config_server_ids and capturing_config_server_ids != self._warned_capturing_config_server_ids:
            verbose_logger.warning(
                "config.yaml MCP server_id(s) %s are the name or alias of a database-backed MCP "
                "server. Permission entries naming them resolve to the config.yaml server, not the "
                "database one. Give the config entry a different server_id.",
                ", ".join(sorted(capturing_config_server_ids)),
            )
        self._warned_capturing_config_server_ids = capturing_config_server_ids


def catalog_operation(
    manager: Callable[[], MCPServerManager],
) -> Callable[[Callable[_P, Awaitable[_R]]], Callable[_P, Awaitable[_R]]]:
    def decorate(function: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R]]:
        @wraps(function)
        async def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:  # kwargs-ok: preserves ParamSpec

            async with manager().catalog.operation():
                return await function(*args, **kwargs)

        return wrapped

    return decorate


def global_manager() -> MCPServerManager:
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import global_mcp_server_manager

    return global_mcp_server_manager


public_catalog_operation: Final[Callable[[Callable[_P, Awaitable[_R]]], Callable[_P, Awaitable[_R]]]] = (
    catalog_operation(global_manager)
)
