"""Authoritative MCP catalog snapshots shared by legacy lookup adapters."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import time
import uuid
from collections import UserDict
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, MutableMapping, Sequence
from contextlib import ExitStack, asynccontextmanager
from contextvars import Context, ContextVar, copy_context
from dataclasses import dataclass, replace
from functools import partial, wraps
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Generic, Literal, ParamSpec, Protocol, TypeAlias, TypeVar, cast

import anyio
from mcp.shared.exceptions import MCPError
from mcp.types import CacheableResult, ListToolsResult, Tool
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter
from typing_extensions import Self

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy.mcp_server.result_conversion import age_freshness, aggregate_freshness

if TYPE_CHECKING:
    from mcp.types import PaginatedRequestParams, PaginatedResult

    from litellm.caching.redis_cache import RedisCache
    from litellm.proxy.mcp_server.contracts import CatalogListRequest, CatalogListResult, OperationContext
    from litellm.proxy.mcp_server.db import OAuthCredentialPayload
    from litellm.proxy.mcp_server.faults.list_outcomes import AggregateToolListing, ServerOutcome
    from litellm.proxy.mcp_server.mcp_server_manager import MCPServerManager
    from litellm.rust_bridge._native import NativeAuthCache
    from litellm.types.mcp_server.mcp_server_manager import MCPServer
    from litellm.types.mcp_server.tool_registry import MCPTool

_P = ParamSpec("_P")
_R = TypeVar("_R")
_Page = TypeVar("_Page", bound="PaginatedResult")
_PREPARED_TOOL_HEADERS: TypeAlias = tuple[
    str | Mapping[str, str] | None,
    Mapping[str, str] | None,
    str | Mapping[str, str] | None,
]
_TOOL_LISTING_CURSOR_TTL: Final = 600
_TOOL_PAGE_PRODUCER_MAX: Final = 64
_TOOL_PAGE_WRITE_TIMEOUT: Final = 1.0
_TOOL_PAGE_POLL_INTERVAL: Final = 0.05
_JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_OUTCOME_VALUES: Final[TypeAdapter[dict[str, JsonValue]]] = TypeAdapter(dict[str, JsonValue])


class _OperationRoutes(UserDict[str, str]):
    def __init__(self, initial: Mapping[str, str]) -> None:
        super().__init__()
        self.data = dict(initial)
        self.written_names: set[str] = set()  # mutable-ok: record route writes without copying the journal per tool

    def __setitem__(self, name: str, owner: str) -> None:
        self.data[name] = owner
        self.written_names.add(name)


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    servers: Mapping[str, MCPServer]
    identity: str
    tools: Mapping[str, MCPTool]
    routing: MutableMapping[str, str]


def _configuration_identity(server: MCPServer) -> str:
    return json.dumps(
        server.model_dump(
            mode="json",
            exclude=frozenset(
                (
                    "short_prefix",
                    "scopes",
                    "authorization_url",
                    "token_url",
                    "registration_url",
                    "authorization_response_iss_parameter_supported",
                    "client_id_metadata_document_supported",
                )
            )
            | (frozenset() if server.issuer_is_anchored else frozenset(("issuer",))),
        ),
        sort_keys=True,
    )


def _check_oauth_revision(selected: MCPServer, candidate: MCPServer | None) -> None:
    from fastapi import HTTPException

    if candidate is None or _configuration_identity(selected) != _configuration_identity(candidate):
        raise HTTPException(status_code=503, detail="OAuth metadata discovery changed repeatedly; retry shortly")


def _snapshot(manager: MCPServerManager, database_identity: str) -> CatalogSnapshot:
    from litellm.proxy.mcp_server.tool_registry import global_mcp_tool_registry

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


class CatalogSnapshots:
    def __init__(self, manager: MCPServerManager) -> None:
        self.manager = manager
        self._refresh_lock = asyncio.Lock()
        self._database_identity = ""
        self._arrival_ticket = 0
        self._completed_ticket = 0
        self._shared_snapshot: CatalogSnapshot | None = None
        self._applied_revision: int | None = None
        self._warned_shadowed_config_server_ids: frozenset[str] = frozenset()
        self._warned_capturing_config_server_ids: frozenset[str] = frozenset()
        self._operation: ContextVar[tuple[CatalogSnapshot, asyncio.Event] | None] = ContextVar(
            "mcp_catalog_snapshot", default=None
        )
        self._staged_routing: ContextVar[tuple[dict[str, str], asyncio.Event] | None] = ContextVar(
            "mcp_catalog_routing", default=None
        )

    def current(self) -> CatalogSnapshot | None:
        scoped: Final = self._operation.get()
        return scoped[0] if scoped is not None and not scoped[1].is_set() else None

    def detached_context(self) -> Context:
        context: Final = copy_context()
        context.run(self._operation.set, None)
        context.run(self._staged_routing.set, None)
        from litellm.caching.redis_batch import (
            _active_request_batches,  # pyright: ignore[reportPrivateUsage]  # clear inherited producer batching
        )

        context.run(_active_request_batches.set, None)
        return context

    def routing(self) -> MutableMapping[str, str]:
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
        from litellm.proxy.mcp_server.db import get_mcp_catalog_revision

        try:
            revision: Final = await get_mcp_catalog_revision(prisma_client)
            if revision is not None and revision == self._applied_revision and self._shared_snapshot is not None:
                return self._shared_snapshot
            self._arrival_ticket += 1
            arrival: Final = self._arrival_ticket
            async with self._refresh_lock:
                if arrival > self._completed_ticket or revision != self._applied_revision:
                    await self._publish_refresh(revision, reuse_unchanged=True)
                if self._shared_snapshot is None:
                    raise HTTPException(status_code=503, detail="MCP server configuration could not be refreshed")
                return self._shared_snapshot
        except Exception as exc:
            raise HTTPException(status_code=503, detail="MCP server configuration could not be refreshed") from exc

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
    async def operation(self) -> AsyncGenerator[CatalogSnapshot]:
        current: Final = self.current()
        if current is not None:
            yield current
            return
        from litellm.proxy.mcp_server.tool_registry import global_mcp_tool_registry

        shared: Final = await self._fresh_snapshot()
        initial_routing: Final = self._unchanged_routing(shared.servers, self.manager.published_tool_routes)
        routing: Final = _OperationRoutes(initial_routing)
        snapshot: Final = replace(
            shared,
            servers=MappingProxyType({key: value.model_copy(deep=True) for key, value in shared.servers.items()}),
            routing=routing,
        )
        closed: Final = asyncio.Event()
        token: Final = self._operation.set((snapshot, closed))
        try:
            with global_mcp_tool_registry.catalog_scope(snapshot.tools):
                yield snapshot
        finally:
            closed.set()
            self._operation.reset(token)
            self._retain_discovered_routing(snapshot, frozenset(routing.written_names))

    def _retain_discovered_routing(self, snapshot: CatalogSnapshot, written_names: frozenset[str]) -> None:
        self.manager.published_tool_routes = self.manager.published_tool_routes | self._unchanged_routing(
            snapshot.servers,
            {name: owner for name, owner in snapshot.routing.items() if name in written_names},
        )

    def _unchanged_routing(
        self, servers: Mapping[str, MCPServer], routing: Mapping[str, str]
    ) -> MappingProxyType[str, str]:
        from litellm.proxy.mcp_server.utils import normalize_server_name

        current: Final = self.manager.config_mcp_servers | self.manager.registry
        unchanged: Final = (
            server
            for key, server in servers.items()
            if (candidate := current.get(key)) is not None
            and _configuration_identity(candidate) == _configuration_identity(server)
        )
        unchanged_owners: Final = frozenset(chain.from_iterable(map(self.manager.owned_mapping_values, unchanged)))
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
        from litellm.proxy.mcp_server.mcp_server_manager import oauth_endpoints_unresolved

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
        from litellm.proxy.mcp_server.db import get_mcp_catalog_revision
        from litellm.proxy.proxy_server import prisma_client

        revision: Final = await get_mcp_catalog_revision(prisma_client) if prisma_client is not None else None
        async with self._refresh_lock:
            await self._publish_refresh(revision)

    async def _publish_refresh(self, revision: int | None, *, reuse_unchanged: bool = False) -> None:
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
            self._applied_revision = revision
            self._completed_ticket = covered
        finally:
            self._operation.reset(token)

    async def _reload_and_publish(self, *, reuse_unchanged: bool) -> None:
        from litellm.proxy.mcp_server.tool_registry import global_mcp_tool_registry
        from litellm.proxy.mcp_server.utils import normalize_server_name

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
                refreshed_openapi: Final = (
                    server
                    for server in self.manager.registry.values()
                    if server.spec_path and server is not live_registry.get(server.server_id)
                )
                refreshed_openapi_owners: Final = frozenset(
                    chain.from_iterable(map(self.manager.owned_mapping_values, refreshed_openapi))
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

    async def _stage_servers(
        self, rows: Sequence[BaseModel], *, reuse_unchanged: bool
    ) -> dict[str, MCPServer]:  # mutable-ok: assign_unique_short_prefix requires a dict registry
        from litellm.proxy.mcp_server.db import LiteLLM_MCPServerTable
        from litellm.proxy.mcp_server.mcp_server_manager import (
            carry_forward_resolved_oauth_endpoints,
            oauth_endpoints_unresolved,
            warn_on_server_name_fields,
        )

        previous_registry: Final = self.manager.registry
        new_registry: Final[dict[str, MCPServer]] = {}

        # Stage one: build every server.  Stage two assigns short prefixes
        # against the *full* set so dedup is deterministic regardless of
        # iteration order.
        for row in rows:
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
                self.manager.warn_if_newly_blocked_stdio(server, existing_server)
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

        return new_registry

    async def _reload(self, *, reuse_unchanged: bool) -> None:
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            get_prisma_client_or_throw,
        )
        from litellm.proxy.mcp_server.mcp_server_manager import (
            config_ids_capturing_db_identifiers,
            warn_on_shared_identifier_prefixes,
        )

        verbose_logger.debug("Loading MCP servers from database into registry...")

        prisma_client: Final = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        # Load only "active", legacy "approved", and NULL (no approval workflow) rows.
        # Pending/rejected servers are excluded at the DB level so we never load them.
        from litellm.proxy.mcp_server.db import get_runtime_mcp_server_rows

        raw_rows: Final[Sequence[BaseModel]] = await get_runtime_mcp_server_rows(prisma_client)
        database_identity: Final = hashlib.sha256(
            json.dumps(
                tuple(sorted(json.dumps(row.model_dump(mode="json"), sort_keys=True, default=str) for row in raw_rows))
            ).encode()
        ).hexdigest()
        verbose_logger.info("Found %s MCP servers in database", len(raw_rows))

        previous_registry: Final = self.manager.registry
        new_registry: Final = await self._stage_servers(raw_rows, reuse_unchanged=reuse_unchanged)

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
                self.manager.invalidate_server_definition_caches(server_id)
                self.manager.invalidate_oauth_discovery_state(server_id)
        self._database_identity = database_identity
        self.manager.registry = registered_registry
        if not reuse_unchanged:
            self.manager.clear_initialize_instructions()
        warn_on_shared_identifier_prefixes(registered_registry.values())
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
    from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager

    return global_mcp_server_manager


def public_catalog_operation(function: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R]]:
    return catalog_operation(global_manager)(function)


async def paginate_catalog(
    *,
    method: str,
    cursor: str | None,
    caller_scope: str,
    snapshot: str,
    server_ids: tuple[str, ...],
    fetch: Callable[[str, str | None], Awaitable[_Page]],
    now: int,
) -> tuple[tuple[_Page, ...], str | None, Mapping[str, JsonValue]]:
    from mcp.shared.exceptions import MCPError
    from mcp.types import INVALID_PARAMS
    from pydantic import ValidationError

    from litellm.constants import MCP_TOOL_LISTING_MAX_PAGES
    from litellm.proxy.mcp_server.faults.list_outcomes import SERVER_OUTCOMES_META_KEY, ServerOutcome
    from litellm.proxy.mcp_server.outbound_credentials.result import Error
    from litellm.proxy.mcp_server.state_tokens import open_state, seal_state

    ordered: Final = tuple(sorted(server_ids))
    purpose: Final = "mcp.catalog.list.v1:" + method
    if cursor is None:
        state = _ListingState(
            caller_scope=caller_scope,
            snapshot=snapshot,
            expires_at=now + _TOOL_LISTING_CURSOR_TTL,
            positions=tuple(_UpstreamPosition(server_id=server_id) for server_id in ordered),
        )
    else:
        opened: Final = open_state(cursor, purpose=purpose, now=now)
        if isinstance(opened, Error):
            raise MCPError(code=INVALID_PARAMS, message=opened.error.value)
        try:
            state = _ListingState.model_validate_json(json.dumps(opened.ok))
        except ValidationError as error:
            raise MCPError(code=INVALID_PARAMS, message="Invalid pagination state; start a fresh listing") from error
        if (
            state.caller_scope != caller_scope
            or state.snapshot != snapshot
            or tuple(position.server_id for position in state.positions) != ordered
            or state.expires_at <= now
        ):
            raise MCPError(code=INVALID_PARAMS, message="Pagination scope or snapshot changed; start a fresh listing")

    async def advance(position: _UpstreamPosition) -> tuple[_UpstreamPosition, _Page | None]:
        if position.complete:
            return position, None
        result: Final = await fetch(position.server_id, position.cursor)
        if method == "tools/list" and result.next_cursor is None:
            page_outcomes: Final = TypeAdapter(dict[str, ServerOutcome]).validate_python(
                (result.meta or {}).get(SERVER_OUTCOMES_META_KEY, {})
            )
            if any(outcome.tag != "ok" for outcome in page_outcomes.values()):
                return (
                    _UpstreamPosition(
                        server_id=position.server_id,
                        cursor=None,
                        complete=True,
                        revision=position.revision,
                        seen=position.seen,
                    ),
                    result,
                )
        revision: Final = (result.meta or {}).get("revision")
        available_revision: Final = revision if isinstance(revision, (str, int)) else None
        if position.revision is not None and position.revision != available_revision:
            raise MCPError(code=INVALID_PARAMS, message="Upstream snapshot changed; start a fresh listing")
        following: Final = result.next_cursor or None
        fingerprint: Final = (
            base64.urlsafe_b64encode(hashlib.sha256(following.encode()).digest()).decode("ascii").rstrip("=")
            if following is not None
            else None
        )
        if fingerprint in position.seen:
            raise MCPError(code=INVALID_PARAMS, message="Upstream repeated a pagination cursor; start a fresh listing")
        if following is not None and len(position.seen) + 1 >= MCP_TOOL_LISTING_MAX_PAGES:
            raise MCPError(code=INVALID_PARAMS, message="Upstream pagination limit reached; start a fresh listing")
        return (
            _UpstreamPosition(
                server_id=position.server_id,
                cursor=following,
                complete=following is None,
                revision=available_revision,
                seen=position.seen + ((fingerprint,) if fingerprint is not None else ()),
            ),
            result,
        )

    started: Final = time.monotonic()
    tasks: Final = tuple(asyncio.create_task(advance(position)) for position in state.positions)
    try:
        results: Final = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    elapsed: Final = time.monotonic() - started
    pages: Final = tuple(
        age_freshness(result, elapsed) if isinstance(result, CacheableResult) else result
        for _, result in results
        if result is not None
    )
    page_outcomes: Final = (
        _OUTCOME_VALUES.validate_python((result.meta or {}).get(SERVER_OUTCOMES_META_KEY, {})) for result in pages
    )
    outcomes: Final = dict(state.failures) | dict(chain.from_iterable(outcome.items() for outcome in page_outcomes))
    failures: Final = {
        key: value for key, value in outcomes.items() if isinstance(value, dict) and value.get("tag") != "ok"
    }
    following_state: Final = state.model_copy(
        update={
            "positions": tuple(position for position, _ in results),
            "failures": failures,
        }
    )
    next_cursor: str | None = None
    if any(not position.complete for position in following_state.positions):
        sealed: Final = seal_state(
            _JSON_VALUE.validate_json(following_state.model_dump_json()),
            purpose=purpose,
            expires_at=following_state.expires_at,
            now=now,
        )
        if isinstance(sealed, Error):
            raise MCPError(code=INVALID_PARAMS, message=sealed.error.value)
        next_cursor = sealed.ok
    return pages, next_cursor, MappingProxyType(outcomes)


async def list_tools_page(
    *,
    cursor: str | None,
    caller_scope: str,
    snapshot: str,
    server_ids: tuple[str, ...],
    fetch: Callable[[str, str | None], Awaitable[ListToolsResult]],
    now: int,
) -> ListToolsResult:
    from mcp.types import ListToolsResult

    from litellm.proxy.mcp_server.faults.list_outcomes import SERVER_OUTCOMES_META_KEY

    pages, next_cursor, outcomes = await paginate_catalog(
        method="tools/list",
        cursor=cursor,
        caller_scope=caller_scope,
        snapshot=snapshot,
        server_ids=server_ids,
        fetch=fetch,
        now=now,
    )
    return ListToolsResult(
        tools=list(chain.from_iterable(page.tools for page in pages)),
        ttl_ms=0
        if any(isinstance(value, dict) and value.get("tag") != "ok" for value in outcomes.values())
        else aggregate_freshness(pages).ttl_ms,
        cache_scope="private",
        next_cursor=next_cursor,
        _meta={SERVER_OUTCOMES_META_KEY: dict(outcomes)} if outcomes else None,
    )


class _UpstreamPosition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    server_id: str
    cursor: str | None = None
    complete: bool = False
    revision: str | int | None = None
    seen: tuple[str, ...] = ()


class _ListingState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    caller_scope: str
    snapshot: str
    expires_at: int
    positions: tuple[_UpstreamPosition, ...]
    failures: Mapping[str, JsonValue] = {}


class _CachedToolPage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    server_id: str
    snapshot_id: str
    page_index: int
    captured_at: float
    expires_at: float
    result: ListToolsResult


class _ToolProducerStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    state: Literal["pending", "complete", "failed"]
    deadline_at: float
    expires_at: float
    next_page_index: int
    failure: JsonValue | None = None
    meta: Mapping[str, JsonValue] | None = None


class _ToolPageStore(Protocol):
    @property
    def available(self) -> bool: ...
    async def get(self, key: str) -> JsonValue | None: ...
    async def set(self, key: str, value: JsonValue, ttl: float) -> None: ...


class _RedisToolPageStore:
    def __init__(self) -> None:
        self._cache: RedisCache | None = None
        self._store: NativeAuthCache | None = None
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        from litellm.proxy.proxy_server import redis_usage_cache

        return redis_usage_cache is not None

    async def _native_store(self) -> NativeAuthCache:
        from litellm.proxy import proxy_server
        from litellm.redis.stores.auth_cache import AuthCacheSettings
        from litellm.redis.stores.connection import connect

        redis_cache: Final = proxy_server.redis_usage_cache
        if redis_cache is None:
            raise RuntimeError("Redis is unavailable")
        if self._cache is redis_cache and self._store is not None:
            return self._store
        async with self._lock:
            if self._cache is not redis_cache or self._store is None:
                store: Final = connect(redis_cache).auth_cache(AuthCacheSettings(memory_entries=0, memory_ttl=0))
                if not store.has_redis:
                    raise RuntimeError("Redis is unavailable")
                self._store = store
                self._cache = redis_cache
            return self._store

    async def get(self, key: str) -> JsonValue | None:
        from litellm.rust_bridge._native import NativeSend

        store: Final = await self._native_store()
        pending: Final = store.get_many([key], NativeSend.Now())
        if pending is None:
            raise RuntimeError("Redis is unavailable")
        values: Final = await pending.resolve()
        if len(values) != 1:
            raise RuntimeError("Redis returned an unexpected tool page count")
        value: Final = values[0]
        return _JSON_VALUE.validate_python(value) if value is not None else None

    async def set(self, key: str, value: JsonValue, ttl: float) -> None:
        from litellm.rust_bridge._native import NativeSend

        store: Final = await self._native_store()
        await store.declare_set(key, value, ttl, NativeSend.Now()).resolve()


_TOOL_PAGE_STORE: Final[_ToolPageStore] = _RedisToolPageStore()
_TOOL_PRODUCERS: Final[dict[tuple[str, str, str], asyncio.Task[None]]] = {}  # mutable-ok: retain producers for shutdown
_TOOL_PRODUCER_LOCK: Final = asyncio.Lock()


_CACHED_TOOL_PAGE: Final[TypeAdapter[_CachedToolPage]] = TypeAdapter(_CachedToolPage)


def _tool_page_cursor(snapshot_id: str, page_index: int) -> str:
    return f"mcp-tools-page-v2:{snapshot_id}:{page_index}"


def _parse_tool_page_cursor(cursor: str) -> tuple[str, int] | None:
    prefix: Final = "mcp-tools-page-v2:"
    if not cursor.startswith(prefix):
        return None
    snapshot_id, separator, raw_index = cursor.removeprefix(prefix).partition(":")
    valid_snapshot_id: Final = len(snapshot_id) == 32 and all(
        character in "0123456789abcdef" for character in snapshot_id
    )
    valid_page_index: Final = raw_index.isascii() and raw_index.isdecimal() and len(raw_index) <= 8
    if not separator or not valid_snapshot_id or not valid_page_index:
        return None
    page_index: Final = int(raw_index)
    if page_index < 1:
        return None
    return snapshot_id, page_index


def _tool_page_cache_key(caller_scope: str, server_id: str, snapshot_id: str, page_index: int) -> str:
    server_hash: Final = hashlib.sha256(server_id.encode()).hexdigest()
    return f"mcp:tools:list-page:v2:{caller_scope}:{server_hash}:{snapshot_id}:{page_index}"


def _tool_page_status_key(caller_scope: str, server_id: str, snapshot_id: str) -> str:
    server_hash: Final = hashlib.sha256(server_id.encode()).hexdigest()
    return f"mcp:tools:list-status:v2:{caller_scope}:{server_hash}:{snapshot_id}"


class _StreamingToolPagePublisher:
    def __init__(
        self,
        store: _ToolPageStore,
        caller_scope: str,
        server_id: str,
        snapshot_id: str,
        server_key: str,
        expires_at: float,
        first_page: asyncio.Future[ListToolsResult],
    ) -> None:
        self.store: Final = store
        self.caller_scope: Final = caller_scope
        self.server_id: Final = server_id
        self.snapshot_id: Final = snapshot_id
        self.server_key: Final = server_key
        self.expires_at: Final = expires_at
        self.first_page: Final = first_page
        self.next_index = 0
        self.streaming = True
        self.pending_final: ListToolsResult | None = None
        self.last_meta: Mapping[str, JsonValue] | None = None

    def _ttl(self) -> float:
        remaining: Final = self.expires_at - time.time()
        if remaining <= 0:
            raise RuntimeError("Tool listing expired")
        return remaining

    async def start(self, deadline_at: float) -> None:
        status: Final = _ToolProducerStatus(
            state="pending",
            deadline_at=deadline_at,
            expires_at=self.expires_at,
            next_page_index=0,
        )
        try:
            await self._set(
                _tool_page_status_key(self.caller_scope, self.server_id, self.snapshot_id),
                _JSON_VALUE.validate_json(status.model_dump_json()),
            )
        except Exception:  # noqa: BLE001  # storage failures switch to full-response fallback
            self.streaming = False

    async def _get(self, key: str) -> JsonValue | None:
        if self.first_page.done():
            return await self.store.get(key)
        with anyio.fail_after(_TOOL_PAGE_WRITE_TIMEOUT):
            return await self.store.get(key)

    async def _set(self, key: str, value: JsonValue) -> None:
        if self.first_page.done():
            await self.store.set(key, value, self._ttl())
            return
        with anyio.fail_after(_TOOL_PAGE_WRITE_TIMEOUT):
            await self.store.set(key, value, self._ttl())

    def _envelope(self, result: ListToolsResult, page_index: int) -> _CachedToolPage:
        return _CachedToolPage(
            server_id=self.server_id,
            snapshot_id=self.snapshot_id,
            page_index=page_index,
            captured_at=time.time(),
            expires_at=self.expires_at,
            result=result,
        )

    async def _write(self, result: ListToolsResult) -> None:
        page_index: Final = self.next_index
        envelope: Final = self._envelope(result, page_index)
        key: Final = _tool_page_cache_key(self.caller_scope, self.server_id, self.snapshot_id, page_index)
        existing: Final = await self._get(key)
        existing_page: Final = _CACHED_TOOL_PAGE.validate_python(existing) if existing is not None else None
        if existing_page is not None:
            if (
                existing_page.server_id,
                existing_page.snapshot_id,
                existing_page.page_index,
                existing_page.expires_at,
                existing_page.result,
            ) != (
                self.server_id,
                self.snapshot_id,
                page_index,
                self.expires_at,
                result,
            ):
                raise RuntimeError("Tool listing page index was already published")
        else:
            await self._set(key, _JSON_VALUE.validate_json(envelope.model_dump_json()))
        self.next_index += 1
        if page_index == 0 and not self.first_page.done():
            self.first_page.set_result(result)

    async def publish(self, result: ListToolsResult) -> tuple[bool, int]:
        if not self.streaming:
            return False, 0
        try:
            return await self._publish(result)
        except Exception:
            if self.first_page.done():
                raise
            self.streaming = False
            return False, 0

    async def _publish(self, result: ListToolsResult) -> tuple[bool, int]:
        from litellm.proxy.mcp_server.faults.list_outcomes import ServerListOk

        self.last_meta = result.meta
        if not result.next_cursor:
            self.pending_final = result
            return False, 0
        page_with_outcome: Final = self._with_outcome(result, ServerListOk(tool_count=len(result.tools)))
        page: Final = page_with_outcome.model_copy(
            update={"next_cursor": _tool_page_cursor(self.snapshot_id, self.next_index + 1)}
        )
        await self._write(page)
        return True, len(page.tools)

    def _with_outcome(self, page: ListToolsResult, outcome: ServerOutcome) -> ListToolsResult:
        from litellm.proxy.mcp_server.faults.list_outcomes import SERVER_OUTCOMES_META_KEY

        meta: Final = {
            **(page.meta or {}),
            SERVER_OUTCOMES_META_KEY: {self.server_key: outcome.model_dump(mode="json")},
        }
        return page.model_copy(update={"meta": meta, "next_cursor": None})

    async def finish(
        self,
        outcome: ServerOutcome,
        *,
        failed: bool,
        complete_result: ListToolsResult | None = None,
    ) -> None:
        if self.streaming:
            try:
                await self._finish(outcome, failed=failed)
            except Exception:
                if self.first_page.done():
                    raise
                self.streaming = False
            else:
                return
        if not self.first_page.done():
            self.first_page.set_result(self._with_outcome(complete_result or ListToolsResult(tools=[]), outcome))

    async def _finish(self, outcome: ServerOutcome, *, failed: bool) -> None:
        page: Final = (
            self.pending_final
            if self.pending_final is not None
            else ListToolsResult(
                tools=[],
                _meta=dict(self.last_meta) if self.last_meta is not None else None,
            )
        )
        terminal_page: Final = self._with_outcome(page, outcome)
        return_terminal_page_without_storage: Final = (
            not failed and self.pending_final is not None and self.next_index == 0 and not self.first_page.done()
        )
        if return_terminal_page_without_storage:
            self.first_page.set_result(terminal_page)
            return
        status: Final = _ToolProducerStatus(
            state="failed" if failed else "complete",
            deadline_at=time.time(),
            expires_at=self.expires_at,
            next_page_index=self.next_index,
            failure=outcome.model_dump(mode="json") if failed else None,
            meta=self.last_meta,
        )
        status_key: Final = _tool_page_status_key(self.caller_scope, self.server_id, self.snapshot_id)
        status_value: Final = _JSON_VALUE.validate_json(status.model_dump_json())
        if failed and self.first_page.done():
            await self._set(status_key, status_value)
            return
        await self._write(terminal_page)
        self.pending_final = None
        await self._set(status_key, status_value)


def _remove_tool_producer(key: tuple[str, str, str], task: asyncio.Task[None]) -> None:
    _TOOL_PRODUCERS.pop(key, None)
    if not task.cancelled():
        task.exception()


async def _produce_tool_pages(
    *,
    publisher: _StreamingToolPagePublisher,
    server: MCPServer,
    context: OperationContext,
    allowed: Sequence[MCPServer],
    prefetched: Mapping[str, OAuthCredentialPayload],
    prepared_headers: _PREPARED_TOOL_HEADERS | None,
    record_listing: bool,
    use_tool_list_cache: bool,
    snapshot_identity: str,
    deadline_at: float,
) -> None:
    from mcp.types import INTERNAL_ERROR

    from litellm.proxy.mcp_server.faults.list_outcomes import ServerListFault, classify_list_exception
    from litellm.proxy.mcp_server.operations import global_mcp_server_manager

    updates = ExitStack()
    try:
        with anyio.fail_after(max(0.0, deadline_at - time.time())):
            await publisher.start(deadline_at)
            async with global_mcp_server_manager.catalog.operation() as snapshot:
                if snapshot.identity != snapshot_identity:
                    raise MCPError(code=INTERNAL_ERROR, message="MCP server catalog changed; retry the listing")
                result, outcome = await get_filtered_server_tools(
                    server,
                    context=context,
                    allowed_mcp_servers=allowed,
                    prefetched_oauth_creds=prefetched,
                    on_page=publisher.publish,
                    record_listing=record_listing,
                    listing_updates=updates,
                    use_tool_list_cache=use_tool_list_cache,
                    prepared_headers=prepared_headers,
                )
                if outcome.tag == "ok":
                    global_mcp_server_manager.catalog.assert_current(server)
                    updates.close()
                    await publisher.finish(outcome, failed=False, complete_result=result)
                else:
                    updates.pop_all()
                    if not publisher.first_page.done():
                        challenge_rejected_credential(server, outcome, allowed)
                    await publisher.finish(outcome, failed=True, complete_result=result)
    except asyncio.CancelledError:
        updates.pop_all()
        with anyio.move_on_after(1, shield=True):
            await publisher.finish(ServerListFault(tag="internal"), failed=True)
        raise
    except Exception as error:  # noqa: BLE001  # publish a terminal outcome for arbitrary producer failures
        updates.pop_all()
        with anyio.move_on_after(1, shield=True):
            await publisher.finish(classify_list_exception(error), failed=True)
        verbose_logger.warning("MCP tool page producer failed for server %s", server.name)
    finally:
        if not publisher.first_page.done():
            publisher.first_page.set_exception(
                MCPError(code=INTERNAL_ERROR, message="Tool listing producer stopped before its first page")
            )


async def _start_tool_page_producer(
    *,
    publisher: _StreamingToolPagePublisher,
    server: MCPServer,
    context: OperationContext,
    allowed: Sequence[MCPServer],
    prefetched: Mapping[str, OAuthCredentialPayload],
    prepared_headers: _PREPARED_TOOL_HEADERS | None,
    record_listing: bool,
    use_tool_list_cache: bool,
    snapshot_identity: str,
    deadline_at: float,
) -> asyncio.Task[None] | None:
    from litellm.proxy.mcp_server.operations import global_mcp_server_manager

    key: Final = (publisher.caller_scope, server.server_id, publisher.snapshot_id)
    async with _TOOL_PRODUCER_LOCK:
        if len(_TOOL_PRODUCERS) >= _TOOL_PAGE_PRODUCER_MAX:
            return None
        producer: Final = _produce_tool_pages(
            publisher=publisher,
            server=server,
            context=context,
            allowed=allowed,
            prefetched=prefetched,
            prepared_headers=prepared_headers,
            record_listing=record_listing,
            use_tool_list_cache=use_tool_list_cache,
            snapshot_identity=snapshot_identity,
            deadline_at=deadline_at,
        )
        task: Final[asyncio.Task[None]] = global_mcp_server_manager.catalog.detached_context().run(
            asyncio.create_task, producer
        )
        _TOOL_PRODUCERS[key] = task
        task.add_done_callback(partial(_remove_tool_producer, key))
        return task


async def shutdown_tool_page_producers() -> None:
    async with _TOOL_PRODUCER_LOCK:
        tasks: Final = tuple(_TOOL_PRODUCERS.values())
        _TOOL_PRODUCERS.clear()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _failed_tool_page(status: _ToolProducerStatus, server_key: str) -> ListToolsResult:
    from mcp.types import INVALID_PARAMS

    from litellm.proxy.mcp_server.faults.list_outcomes import SERVER_OUTCOMES_META_KEY, ServerOutcome

    if status.failure is None:
        raise MCPError(code=INVALID_PARAMS, message="Tool listing page is unavailable; start a fresh listing")
    outcome: Final[ServerOutcome] = TypeAdapter[ServerOutcome](ServerOutcome).validate_python(status.failure)
    return ListToolsResult(
        tools=[],
        _meta={
            **(status.meta or {}),
            SERVER_OUTCOMES_META_KEY: {server_key: outcome.model_dump(mode="json")},
        },
    )


async def _read_tool_page(
    cursor: str,
    *,
    server_id: str,
    server_key: str,
    caller_scope: str,
    store: _ToolPageStore,
    initial_read_timeout: float,
) -> ListToolsResult:
    from mcp.types import INTERNAL_ERROR, INVALID_PARAMS

    def restart() -> MCPError:
        return MCPError(code=INVALID_PARAMS, message="Tool listing page is unavailable; start a fresh listing")

    parsed: Final = _parse_tool_page_cursor(cursor)
    if parsed is None:
        raise restart()
    snapshot_id, page_index = parsed
    key: Final = _tool_page_cache_key(caller_scope, server_id, snapshot_id, page_index)
    status_key: Final = _tool_page_status_key(caller_scope, server_id, snapshot_id)
    poll_deadline: float = time.monotonic() + max(0.0, initial_read_timeout)
    while (remaining := poll_deadline - time.monotonic()) > 0:
        try:
            with anyio.fail_after(remaining):
                value = await store.get(key)
                if value is None:
                    raw_status = await store.get(status_key)
                    if raw_status is None:
                        raise restart()
                    status = _ToolProducerStatus.model_validate(raw_status)
                    if status.expires_at <= time.time():
                        raise restart()
                    if status.state == "failed":
                        return _failed_tool_page(status, server_key)
                    if status.state == "pending":
                        status_deadline = min(status.deadline_at, status.expires_at)
                        status_remaining = max(0.0, status_deadline - time.time())
                        poll_deadline = min(poll_deadline, time.monotonic() + status_remaining)
                    else:
                        final_read_timeout = min(remaining, max(0.0, status.expires_at - time.time()))
                        with anyio.fail_after(final_read_timeout):
                            value = await store.get(key)
                        if value is None:
                            raise restart()
            if value is None:
                await asyncio.sleep(min(_TOOL_PAGE_POLL_INTERVAL, max(0.0, poll_deadline - time.monotonic())))
                continue
            page = _CACHED_TOOL_PAGE.validate_python(value)
            now = time.time()
            valid = (
                page.server_id == server_id
                and page.snapshot_id == snapshot_id
                and page.page_index == page_index
                and math.isfinite(page.captured_at)
                and math.isfinite(page.expires_at)
                and page.expires_at > now
                and page.result.next_cursor in (None, _tool_page_cursor(snapshot_id, page_index + 1))
            )
            if not valid:
                raise restart()
            elapsed_ms = max(0, int((now - page.captured_at) * 1000))
            return page.result.model_copy(update={"ttl_ms": max(0, int(page.result.ttl_ms or 0) - elapsed_ms)})
        except MCPError:
            raise
        except TimeoutError as error:
            raise restart() from error
        except Exception as error:
            raise MCPError(code=INTERNAL_ERROR, message="Unable to read tool listing continuation") from error
    raise restart()


async def prepare_tool_listing_headers(
    server: MCPServer,
    context: OperationContext,
    allowed_mcp_servers: Sequence[MCPServer],
    prefetched_oauth_creds: Mapping[str, OAuthCredentialPayload] | None,
) -> tuple[str | Mapping[str, str] | None, Mapping[str, str] | None, str | Mapping[str, str] | None]:
    from litellm.proxy.mcp_server.operations import (
        _get_byok_credential,
        _get_user_oauth_extra_headers_from_db,
        _prepare_mcp_server_headers,
    )
    from litellm.types.mcp import MCPAuth

    user_api_key_auth, mcp_auth_header, _, mcp_server_auth_headers, oauth2_headers, raw_headers, _ = (
        context.legacy_auth()
    )
    server_auth_header, extra_headers = _prepare_mcp_server_headers(
        server=server,
        mcp_server_auth_headers=mcp_server_auth_headers,
        mcp_auth_header=mcp_auth_header,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
        user_api_key_auth=user_api_key_auth,
        scope_servers=list(allowed_mcp_servers),
    )

    # Prefer server-stored per-user OAuth when configured, so a stale
    # Authorization header from the MCP client cannot override Redis/DB
    # (same issue as call_tool in mcp_server_manager: VS Code caches tokens).
    from litellm.proxy.mcp_server.outbound_credentials.adapter import (  # noqa: PLC0415
        to_server_spec,
    )

    # A server migrated to the v2 resolver gets its token from the resolver at connect
    # time; building it here would double-resolve and be shadowed by the v2 graft. The
    # preemptive 401 already challenged a missing token, so one exists for the connect.
    migrated_to_v2: Final = to_server_spec(server) is not None
    if (
        not migrated_to_v2
        and server.auth_type == MCPAuth.oauth2
        and getattr(server, "needs_user_oauth_token", False)
        and user_api_key_auth is not None
    ):
        db_headers: Final = await _get_user_oauth_extra_headers_from_db(
            server,
            user_api_key_auth,
            prefetched_creds=prefetched_oauth_creds,
        )
        if db_headers:
            extra_headers = db_headers

    elif not migrated_to_v2 and extra_headers is None and server.auth_type == MCPAuth.oauth2:
        extra_headers = await _get_user_oauth_extra_headers_from_db(
            server,
            user_api_key_auth,
            prefetched_creds=prefetched_oauth_creds,
        )

    catalog_auth_header: Final = server_auth_header
    if server.is_byok and server.auth_type != MCPAuth.oauth2 and server_auth_header is None:
        server_auth_header = await _get_byok_credential(server, user_api_key_auth)

    return server_auth_header, extra_headers, catalog_auth_header


def challenge_rejected_credential(server: MCPServer, outcome: ServerOutcome, listed: Sequence[MCPServer]) -> None:
    """Answer a listing of this one server with the gateway's 401 when the caller can fix the rejection."""
    from litellm.proxy.mcp_server.legacy_callbacks import get_listing_scope_response

    scope_response: Final = get_listing_scope_response()
    if (
        scope_response is not None
        and len(listed) == 1
        and outcome.tag == "auth_required"
        and server.reauthorizes_in_client
    ):
        scope_response.deny_reauth()


async def get_filtered_server_tools(
    server: MCPServer,
    *,
    context: OperationContext,
    allowed_mcp_servers: Sequence[MCPServer],
    prefetched_oauth_creds: Mapping[str, OAuthCredentialPayload],
    params: PaginatedRequestParams | None = None,
    on_page: Callable[[ListToolsResult], Awaitable[tuple[bool, int] | None]] | None = None,
    record_listing: bool = False,
    listing_updates: ExitStack | None = None,
    use_tool_list_cache: bool = True,
    prepared_headers: _PREPARED_TOOL_HEADERS | None = None,
) -> tuple[ListToolsResult, ServerOutcome]:
    from litellm.proxy.mcp_server.exceptions import MCPUpstreamAuthError
    from litellm.proxy.mcp_server.faults.list_outcomes import ServerListOk, classify_list_exception
    from litellm.proxy.mcp_server.operations import (
        apply_display_name_overrides,
        filter_tools_by_allowed_tools,
        filter_tools_by_key_team_permissions,
        global_mcp_server_manager,
    )

    user_api_key_auth, _, _, _, oauth2_headers, raw_headers, client_ip = context.legacy_auth()
    mcp_proxy_mode: Final = context.mcp_proxy_mode
    if server is None:
        return ListToolsResult(tools=[]), ServerListOk(tool_count=0)

    first_page_observed: Final = asyncio.Event()
    try:
        resolved_headers: Final = prepared_headers or await prepare_tool_listing_headers(
            server, context, allowed_mcp_servers, prefetched_oauth_creds
        )
        prepared_auth, prepared_extra_headers, prepared_catalog_auth = resolved_headers
        server_auth_header: Final = dict(prepared_auth) if isinstance(prepared_auth, Mapping) else prepared_auth
        extra_headers: Final = dict(prepared_extra_headers) if prepared_extra_headers is not None else None
        catalog_auth_header: Final = (
            dict(prepared_catalog_auth) if isinstance(prepared_catalog_auth, Mapping) else prepared_catalog_auth
        )

        from litellm.proxy.proxy_server import proxy_logging_obj

        listed_generation: Final = global_mcp_server_manager.listed_tools_generation(server.server_id)
        from litellm.proxy.mcp_server.mcp_server_manager import ListedToolsCaller
        from litellm.proxy.mcp_server.utils import strip_known_server_prefix

        observe_pages: Final = on_page is not None or (params is None and listing_updates is not None)

        async def filter_page(tools: Sequence[Tool]) -> tuple[Tool, ...]:
            allowed_tools: Final = filter_tools_by_allowed_tools(list(tools), server)
            filtered_tools: Final = await filter_tools_by_key_team_permissions(
                tools=allowed_tools,
                server_id=server.server_id,
                user_api_key_auth=user_api_key_auth,
            )
            return tuple(filtered_tools)

        def decorate_tools(tools: Sequence[Tool]) -> tuple[Tool, ...]:
            copied_tools: Final = [tool.model_copy(deep=True) for tool in tools]
            if mcp_proxy_mode:
                from litellm.proxy.mcp_server.tool_search import with_mcp_proxy_identity

                return tuple(with_mcp_proxy_identity(tool, server.server_id) for tool in copied_tools)
            return tuple(apply_display_name_overrides(copied_tools, server))

        async def publish_filtered_page(page: ListToolsResult) -> None:
            filtered: Final = await filter_page(page.tools)
            if observe_pages and listing_updates is not None:
                underlying_tools: Final = tuple(
                    tool.model_copy(update={"name": strip_known_server_prefix(tool.name, server)}) for tool in filtered
                )
                global_mcp_server_manager.register_prefixed_tool_routes(
                    underlying_tools,
                    server,
                    register_bare_names=False,
                )
            if on_page is not None:
                delivery: Final = await on_page(page.model_copy(update={"tools": list(decorate_tools(filtered))}))
                if delivery is None:
                    return
                delivered, delivered_tool_count = delivery
                if delivered:
                    global_mcp_server_manager.record_listed_tools(
                        server,
                        tuple(
                            tool.model_copy(update={"name": strip_known_server_prefix(tool.name, server)})
                            for tool in filtered[:delivered_tool_count]
                        ),
                        ListedToolsCaller(
                            user_api_key_auth=user_api_key_auth,
                            mcp_auth_header=catalog_auth_header,
                            raw_headers=raw_headers,
                            oauth2_headers=oauth2_headers,
                        ),
                        listed_generation,
                        record_listing=record_listing,
                        continuation=first_page_observed.is_set(),
                    )
                    first_page_observed.set()
                return
            return

        page: Final = (
            ListToolsResult(
                tools=await global_mcp_server_manager.get_tools_from_server(
                    server=server,
                    mcp_auth_header=server_auth_header,
                    extra_headers=extra_headers,
                    add_prefix=True,
                    raw_headers=raw_headers,
                    client_ip=client_ip,
                    user_api_key_auth=user_api_key_auth,
                    oauth2_headers=oauth2_headers,
                    proxy_logging_obj=proxy_logging_obj,
                    catalog_auth_header=catalog_auth_header,
                    record_listing=False,
                    use_tool_list_cache=use_tool_list_cache,
                )
            )
            if params is None and not observe_pages
            else await global_mcp_server_manager.get_tools_page(
                server=server,
                mcp_auth_header=server_auth_header,
                extra_headers=extra_headers,
                add_prefix=True,
                raw_headers=raw_headers,
                client_ip=client_ip,
                user_api_key_auth=user_api_key_auth,
                oauth2_headers=oauth2_headers,
                proxy_logging_obj=proxy_logging_obj,
                params=params,
                on_page=publish_filtered_page if observe_pages else None,
                catalog_auth_header=catalog_auth_header,
                record_listing=False,
                listing_updates=listing_updates,
                use_tool_list_cache=use_tool_list_cache,
            )
        )
        authorized_tools: Final = await filter_page(page.tools)
        filtered_tools: Final = decorate_tools(authorized_tools)

        if observe_pages and listing_updates is not None:
            underlying_tools: Final = tuple(
                tool.model_copy(update={"name": strip_known_server_prefix(tool.name, server)})
                for tool in authorized_tools
            )
            global_mcp_server_manager.register_prefixed_tool_routes(
                underlying_tools,
                server,
                register_bare_names=True,
                listing_updates=listing_updates,
            )

        record: Final = partial(
            global_mcp_server_manager.record_listed_tools,
            server,
            [
                tool.model_copy(update={"name": strip_known_server_prefix(tool.name, server)})
                for tool in authorized_tools
            ],
            ListedToolsCaller(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=catalog_auth_header,
                raw_headers=raw_headers,
                oauth2_headers=oauth2_headers,
            ),
            listed_generation,
            record_listing=record_listing,
            continuation=params is not None and params.cursor is not None,
        )
        if listing_updates is None:
            record()
        else:
            listing_updates.callback(record)

        verbose_logger.debug(
            "Successfully fetched %s tools from server %s, %s after filtering",
            len(page.tools),
            server.name,
            len(filtered_tools),
        )
        return page.model_copy(update={"tools": list(filtered_tools)}), ServerListOk(tool_count=len(filtered_tools))
    except MCPUpstreamAuthError as e:
        # Absorbed so peers keep their tools; a raise here would reach the client as a JSON-RPC error, not a 401
        verbose_logger.debug("MCP list_tools: omitting %s; it needs upstream auth", server.name)
        outcome: Final = classify_list_exception(e)
        if on_page is None or not first_page_observed.is_set():
            challenge_rejected_credential(server, outcome, allowed_mcp_servers)
        return ListToolsResult(tools=[]), outcome
    except Exception as e:
        verbose_logger.exception("Error getting tools from server %s: %s", server.name, e)
        return ListToolsResult(tools=[]), classify_list_exception(e)


async def _tool_listing_scope(
    context: OperationContext,
    servers: Sequence[MCPServer],
    prefetched_oauth_creds: Mapping[str, OAuthCredentialPayload],
) -> tuple[str, Mapping[str, _PREPARED_TOOL_HEADERS]]:
    from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler

    async def resolve_permissions(server: MCPServer) -> JsonValue:
        try:
            permissions: Final = await MCPRequestHandler.get_allowed_tools_for_server(
                server.server_id, context.user_api_key_auth
            )
        except Exception:  # noqa: BLE001  # isolate permission resolution failures by server
            return {"resolution": "failed"}
        if permissions is None:
            return {"resolution": "unrestricted"}
        return {"resolution": "resolved", "tools": list(sorted(permissions))}

    async def resolve_prepared_headers(
        server: MCPServer,
    ) -> tuple[_PREPARED_TOOL_HEADERS | None, JsonValue]:
        try:
            prepared: Final = await prepare_tool_listing_headers(server, context, servers, prefetched_oauth_creds)
        except Exception:  # noqa: BLE001  # isolate credential preparation failures by server
            return None, {"resolution": "failed"}
        prepared_binding: Final = _OUTCOME_VALUES.validate_python(
            {
                "resolution": "resolved",
                "headers": [dict(value) if isinstance(value, Mapping) else value for value in prepared],
            }
        )
        return prepared, prepared_binding

    async def binding(server: MCPServer) -> tuple[str, JsonValue, _PREPARED_TOOL_HEADERS | None]:
        permission_binding: Final = await resolve_permissions(server)
        prepared_result: Final = await resolve_prepared_headers(server)
        credential: Final = prefetched_oauth_creds.get(server.server_id)
        material: Final = _OUTCOME_VALUES.validate_python(
            {
                "server_id": server.server_id,
                "permissions": permission_binding,
                "prepared_headers": prepared_result[1],
                "prefetched_oauth": dict(credential) if credential is not None else None,
            }
        )
        return server.server_id, material, prepared_result[0]

    bindings: Final = await asyncio.gather(*(binding(server) for server in servers))
    material: Final = (_caller_scope(context, servers), tuple(sorted((item[0], item[1]) for item in bindings)))
    scope: Final = hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    prepared: Final = MappingProxyType(
        {server_id: headers for server_id, _binding, headers in bindings if headers is not None}
    )
    return scope, prepared


def _caller_scope(context: OperationContext, servers: Sequence[MCPServer]) -> str:
    from litellm.proxy.mcp_server.utils import upstream_credential_headers

    caller: Final = context.user_api_key_auth
    headers: Final = context.raw_headers or {}
    credential_names: Final = (
        upstream_credential_headers(headers)
        | frozenset({"authorization"})
        | frozenset(map(str.lower, chain.from_iterable(server.extra_headers or () for server in servers)))
    )
    material: Final = (
        caller.model_dump(include={"api_key", "user_id", "team_id", "org_id", "end_user_id", "user_role"}, mode="json")
        if caller is not None
        else None,
        tuple(sorted(context.mcp_servers)) if context.mcp_servers is not None else None,
        context.mcp_auth_header,
        {key: dict(value) for key, value in (context.mcp_server_auth_headers or {}).items()},
        dict(context.oauth2_headers or {}),
        {key.lower(): value for key, value in headers.items() if key.lower() in credential_names},
        context.client_ip,
        context.protocol_version,
        context.mcp_proxy_mode,
    )
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


async def _fetch_cached_tool_page(
    cursor: str,
    *,
    server: MCPServer,
    server_id: str,
    server_key: str,
    caller_scope: str,
    effective_timeout: float,
    context: OperationContext,
    prepared_headers: Mapping[str, _PREPARED_TOOL_HEADERS],
    manager: MCPServerManager,
    listing_updates: ExitStack,
    record_listing: bool,
) -> ListToolsResult:
    from mcp.types import ListToolsResult

    from litellm.proxy import proxy_server
    from litellm.proxy.mcp_server.faults.list_outcomes import SERVER_OUTCOMES_META_KEY, classify_list_exception
    from litellm.proxy.mcp_server.mcp_server_manager import ListedToolsCaller
    from litellm.proxy.mcp_server.operations import resolve_native_tool_name_for_server
    from litellm.proxy.mcp_server.utils import strip_known_server_prefix

    try:
        page: Final = await _read_tool_page(
            cursor,
            server_id=server_id,
            server_key=server_key,
            caller_scope=caller_scope,
            store=_TOOL_PAGE_STORE,
            initial_read_timeout=effective_timeout,
        )
    except MCPError as page_error:
        outcome: Final = classify_list_exception(page_error)
        return ListToolsResult(
            tools=[],
            _meta={SERVER_OUTCOMES_META_KEY: {server_key: outcome.model_dump(mode="json")}},
        )

    if not page.tools:
        return page

    def cached_tool_definition(tool: Tool) -> Tool:
        meta = tool.meta
        if context.mcp_proxy_mode and meta is not None:
            meta = {key: value for key, value in meta.items() if key != "litellm.ai/proxy_tool_identity"}
        name: Final = (
            strip_known_server_prefix(tool.name, server)
            if context.mcp_proxy_mode
            else resolve_native_tool_name_for_server(tool.name, server, (server,))
        )
        return tool.model_copy(update={"name": name, "meta": meta})

    underlying_tools: Final = tuple(cached_tool_definition(tool) for tool in page.tools)
    served_tools: Final = await manager.scan_tool_catalog(
        server,
        underlying_tools,
        proxy_server.proxy_logging_obj,
        context.user_api_key_auth,
        context.raw_headers,
    )
    wire_tools_by_native_name: Final = {
        native_tool.name: wire_tool for wire_tool, native_tool in zip(page.tools, underlying_tools)
    }
    served_wire_tools: Final = tuple(
        native_tool.model_copy(
            update={
                "name": wire_tools_by_native_name[native_tool.name].name,
                "meta": wire_tools_by_native_name[native_tool.name].meta,
            }
        )
        for native_tool in served_tools
        if native_tool.name in wire_tools_by_native_name
    )
    server_headers: Final = prepared_headers.get(server_id)
    catalog_auth_header: Final = server_headers[2] if server_headers is not None else None
    listed_caller: Final = ListedToolsCaller(
        user_api_key_auth=context.user_api_key_auth,
        mcp_auth_header=(
            catalog_auth_header
            if catalog_auth_header is None or isinstance(catalog_auth_header, str)
            else dict(catalog_auth_header)
        ),
        raw_headers=context.raw_headers,
        oauth2_headers=context.oauth2_headers,
    )
    listed_generation: Final = manager.listed_tools_generation(server_id)
    listing_updates.callback(
        manager.record_listed_tools,
        server,
        served_tools,
        listed_caller,
        listed_generation,
        record_listing=record_listing,
        continuation=True,
    )
    listing_updates.callback(
        manager.register_prefixed_tool_routes,
        served_tools,
        server,
        register_bare_names=False,
    )
    return page.model_copy(update={"tools": list(served_wire_tools)})


async def aggregate_gateway_tools(
    context: OperationContext,
    params: PaginatedRequestParams,
    allowed: Sequence[MCPServer],
    prefetched: Mapping[str, OAuthCredentialPayload],
    *,
    record_listing: bool = False,
    enforce_rate_limits: bool = True,
    use_tool_list_cache: bool = True,
) -> AggregateToolListing:
    import time

    from mcp.types import INVALID_PARAMS, ListToolsResult
    from pydantic import TypeAdapter

    from litellm.constants import MCP_TOOL_LISTING_TIMEOUT
    from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
    from litellm.proxy.mcp_server.faults.list_outcomes import (
        SERVER_OUTCOMES_META_KEY,
        AggregateToolListing,
        ServerOutcome,
        classify_list_exception,
    )
    from litellm.proxy.mcp_server.operations import (
        _aggregate_server_key,
        _mcp_server_rate_limit_rejection,  # pyright: ignore[reportPrivateUsage]  # reuse request-scoped admission
        global_mcp_server_manager,
    )
    from litellm.proxy.mcp_server.outbound_credentials.result import Error
    from litellm.proxy.mcp_server.state_tokens import seal_state

    async with global_mcp_server_manager.catalog.operation() as snapshot:
        servers: Final = {server.server_id: server for server in allowed}
        listing_updates: Final = ExitStack()
        rejections: Final[list[ProxyRateLimitError]] = []  # mutable-ok: retain per-server admission errors
        listing_now: Final = int(time.time())
        expires_at: Final = float(listing_now + _TOOL_LISTING_CURSOR_TTL)
        caller_scope, prepared_headers = await _tool_listing_scope(context, allowed, prefetched)
        can_seal_cursor: Final = not isinstance(
            seal_state(
                _JSON_VALUE.validate_python({}),
                purpose="mcp.catalog.list.v1:tools/list",
                expires_at=listing_now + 1,
                now=listing_now,
            ),
            Error,
        )
        stream_pages: Final = _TOOL_PAGE_STORE.available and can_seal_cursor
        started_producers: Final[list[asyncio.Task[None]]] = []  # mutable-ok: cancel abandoned producers

        async def fetch(server_id: str, cursor: str | None) -> ListToolsResult:
            server: Final = servers[server_id]
            server_key: Final = _aggregate_server_key(server)
            effective_timeout: Final = (
                MCP_TOOL_LISTING_TIMEOUT if server.timeout is None else max(server.timeout, MCP_TOOL_LISTING_TIMEOUT)
            )
            if enforce_rate_limits:
                rate_limit_error: Final = await _mcp_server_rate_limit_rejection(server, context.user_api_key_auth)
                if rate_limit_error is not None:
                    rejections.append(rate_limit_error)
                    if cursor is not None:
                        raise rate_limit_error
                    return ListToolsResult(
                        tools=[],
                        _meta={
                            SERVER_OUTCOMES_META_KEY: {
                                server_key: classify_list_exception(rate_limit_error).model_dump(mode="json")
                            }
                        },
                    )
            if cursor is not None:
                return await _fetch_cached_tool_page(
                    cursor,
                    server=server,
                    server_id=server_id,
                    server_key=server_key,
                    caller_scope=caller_scope,
                    effective_timeout=effective_timeout,
                    context=context,
                    prepared_headers=prepared_headers,
                    manager=global_mcp_server_manager,
                    listing_updates=listing_updates,
                    record_listing=record_listing,
                )

            if stream_pages:
                snapshot_id: Final = uuid.uuid4().hex
                first_page: Final[asyncio.Future[ListToolsResult]] = asyncio.get_running_loop().create_future()
                publisher: Final = _StreamingToolPagePublisher(
                    _TOOL_PAGE_STORE,
                    caller_scope,
                    server_id,
                    snapshot_id,
                    server_key,
                    expires_at,
                    first_page,
                )
                task: Final = await _start_tool_page_producer(
                    publisher=publisher,
                    server=server,
                    context=context,
                    allowed=allowed,
                    prefetched=prefetched,
                    prepared_headers=prepared_headers.get(server_id),
                    record_listing=record_listing,
                    use_tool_list_cache=use_tool_list_cache,
                    snapshot_identity=snapshot.identity,
                    deadline_at=min(expires_at, time.time() + effective_timeout),
                )
                if task is not None:
                    started_producers.append(task)
                    return await first_page

            server_updates: Final = ExitStack()
            result, outcome = await get_filtered_server_tools(
                server,
                context=context,
                allowed_mcp_servers=allowed,
                prefetched_oauth_creds=prefetched,
                params=None,
                record_listing=record_listing,
                listing_updates=server_updates,
                use_tool_list_cache=use_tool_list_cache,
                prepared_headers=prepared_headers.get(server_id),
            )
            outcome_meta: Final = {
                **(result.meta or {}),
                SERVER_OUTCOMES_META_KEY: {server_key: outcome.model_dump(mode="json")},
            }
            if outcome.tag != "ok":
                server_updates.pop_all()
                return result.model_copy(update={"meta": outcome_meta})
            listing_updates.callback(server_updates.close)
            return result.model_copy(update={"meta": outcome_meta, "next_cursor": None})

        try:
            result: Final = await list_tools_page(
                cursor=params.cursor,
                caller_scope=caller_scope,
                snapshot=snapshot.identity,
                server_ids=tuple(servers),
                fetch=fetch,
                now=listing_now,
            )
        except BaseException:
            for task in started_producers:
                task.cancel()
            await asyncio.gather(*started_producers, return_exceptions=True)
            raise
        if params.cursor is None and servers and len(rejections) == len(servers):
            raise rejections[0]
        if params.cursor is not None and servers:
            outcomes: Final = _OUTCOME_VALUES.validate_python((result.meta or {}).get(SERVER_OUTCOMES_META_KEY, {}))
            failed: Final = tuple(
                value for value in outcomes.values() if isinstance(value, dict) and value.get("tag") != "ok"
            )
            if len(failed) == len(servers):
                listing_updates.close()
                raise MCPError(code=INVALID_PARAMS, message="Tool listing page is unavailable; start a fresh listing")
        listing_updates.close()
        return AggregateToolListing(
            tools=result.tools,
            outcomes=TypeAdapter(dict[str, ServerOutcome]).validate_python(
                (result.meta or {}).get(SERVER_OUTCOMES_META_KEY, {})
            ),
            next_cursor=result.next_cursor,
            ttl_ms=result.ttl_ms,
        )


class ToolListingFailedOutright(MCPError):
    pass


async def list_gateway_tools(
    context: OperationContext, params: PaginatedRequestParams, *, log_list_tools_to_spendlogs: bool = True
) -> ListToolsResult:
    from mcp.types import INTERNAL_ERROR, ListToolsResult

    from litellm.proxy.mcp_server.faults.list_outcomes import (
        SERVER_OUTCOMES_META_KEY,
        listing_failed_outright,
        outcome_wire_value,
    )
    from litellm.proxy.mcp_server.operations import _list_mcp_tools

    caller, auth, servers, server_headers, oauth_headers, headers, client_ip = context.legacy_auth()
    listing: Final = await _list_mcp_tools(
        user_api_key_auth=caller,
        mcp_auth_header=auth,
        mcp_servers=servers,
        mcp_server_auth_headers=server_headers,
        oauth2_headers=oauth_headers,
        raw_headers=headers,
        client_ip=client_ip,
        params=params,
        protocol_version=context.protocol_version,
        log_list_tools_to_spendlogs=log_list_tools_to_spendlogs,
        record_listing=True,
        list_tools_log_source="mcp_protocol",
    )
    wire_outcomes: Final = (
        {key: outcome_wire_value(outcome) for key, outcome in listing.outcomes.items()} if listing.outcomes else None
    )
    if params.cursor is None and listing_failed_outright(listing.outcomes):
        raise ToolListingFailedOutright(
            code=INTERNAL_ERROR,
            message="Every MCP server in scope failed to list tools",
            data={SERVER_OUTCOMES_META_KEY: wire_outcomes},
        )
    return ListToolsResult(
        tools=listing.tools,
        next_cursor=listing.next_cursor,
        ttl_ms=listing.ttl_ms,
        cache_scope="private",
        _meta={SERVER_OUTCOMES_META_KEY: wire_outcomes} if wire_outcomes else None,
    )


async def list_gateway_catalog(
    context: OperationContext, request: CatalogListRequest, *, log_list_tools_to_spendlogs: bool = True
) -> CatalogListResult:
    import time

    from mcp.types import (
        ListToolsRequest,
        PaginatedRequestParams,
    )

    from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
    from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler
    from litellm.proxy.mcp_server.operations import (
        _get_allowed_mcp_servers,
        global_mcp_server_manager,
        raise_denied_scoped_mcp_access,
    )

    context = replace(context, _caller=await MCPRequestHandler.refresh_catalog_authority(context.user_api_key_auth))
    params: Final = request.params or PaginatedRequestParams()
    if isinstance(request, ListToolsRequest):
        return await list_gateway_tools(context, params, log_list_tools_to_spendlogs=log_list_tools_to_spendlogs)
    caller: Final = context.user_api_key_auth
    scope: Final = context.mcp_servers
    client_ip: Final = context.client_ip
    async with global_mcp_server_manager.catalog.operation() as snapshot:
        allowed: Final = await _get_allowed_mcp_servers(
            user_api_key_auth=caller, mcp_servers=scope, client_ip=client_ip
        )
        if scope and not allowed:
            await raise_denied_scoped_mcp_access(
                requested_names=list(scope), user_api_key_auth=caller, client_ip=client_ip
            )
        servers: Final = {server.server_id: server for server in allowed}
        rejections: Final[list[ProxyRateLimitError]] = []  # mutable-ok: concurrent fetches share first-page errors

        async def fetch(server_id: str, cursor: str | None) -> CatalogListResult:
            server: Final = servers[server_id]
            from mcp.shared.exceptions import MCPError
            from mcp.types import INVALID_PARAMS

            from litellm.proxy.mcp_server.faults.list_outcomes import (
                SERVER_OUTCOMES_META_KEY,
                classify_list_exception,
            )
            from litellm.proxy.mcp_server.operations import (
                _aggregate_server_key,
                _mcp_server_rate_limit_rejection,
            )

            rate_limit_error: Final = await _mcp_server_rate_limit_rejection(server, caller)
            if rate_limit_error is not None:
                if cursor is not None:
                    raise rate_limit_error
                rejections.append(rate_limit_error)
                return combine_optional_catalog(
                    request,
                    (),
                    None,
                    {
                        SERVER_OUTCOMES_META_KEY: {
                            _aggregate_server_key(server): classify_list_exception(rate_limit_error).model_dump(
                                mode="json"
                            )
                        }
                    },
                )

            try:
                page: Final = await fetch_optional_catalog_page(context, request, server, allowed, cursor)
                page_meta: Final[Mapping[str, object]] = page.meta or {}
                return page.model_copy(
                    update={"meta": {key: value for key, value in page_meta.items() if key != SERVER_OUTCOMES_META_KEY}}
                )
            except Exception as error:
                outcome: Final = classify_list_exception(error)
                challenge_rejected_credential(server, outcome, allowed)
                if cursor is not None:
                    raise MCPError(
                        code=INVALID_PARAMS, message="Upstream continuation failed; start a fresh listing"
                    ) from error
                return combine_optional_catalog(
                    request,
                    (),
                    None,
                    {"litellm.ai/server_outcomes": {_aggregate_server_key(server): outcome.model_dump(mode="json")}},
                )

        pages, next_cursor, outcomes = await paginate_catalog(
            method=request.method,
            cursor=params.cursor,
            caller_scope=_caller_scope(context, allowed),
            snapshot=snapshot.identity,
            server_ids=tuple(servers),
            fetch=fetch,
            now=int(time.time()),
        )
        if params.cursor is None and servers and len(rejections) == len(servers):
            raise rejections[0]
        from litellm.proxy.mcp_server.faults.list_outcomes import (
            SERVER_OUTCOMES_META_KEY,
            ServerOutcome,
            outcome_wire_value,
        )

        typed_outcomes: Final = TypeAdapter(dict[str, ServerOutcome]).validate_python(outcomes)
        return combine_optional_catalog(
            request,
            pages,
            next_cursor,
            _OUTCOME_VALUES.validate_python(
                {SERVER_OUTCOMES_META_KEY: {key: outcome_wire_value(value) for key, value in typed_outcomes.items()}}
            )
            if outcomes
            else None,
        )


async def fetch_optional_catalog_page(
    context: OperationContext,
    request: CatalogListRequest,
    server: MCPServer,
    allowed: Sequence[MCPServer],
    cursor: str | None,
) -> CatalogListResult:
    from mcp.types import PaginatedRequestParams

    from litellm.proxy.mcp_server.operations import _prepare_mcp_server_headers, global_mcp_server_manager

    caller, auth, _, server_headers, oauth_headers, raw_headers, client_ip = context.legacy_auth()
    auth_header, extra_headers = _prepare_mcp_server_headers(
        server=server,
        mcp_server_auth_headers=server_headers,
        mcp_auth_header=auth,
        oauth2_headers=oauth_headers,
        raw_headers=raw_headers,
        user_api_key_auth=caller,
        scope_servers=list(allowed),
    )
    return await global_mcp_server_manager.get_optional_catalog_page(
        server,
        request.model_copy(update={"params": PaginatedRequestParams(cursor=cursor)}),
        caller,
        mcp_auth_header=auth_header,
        extra_headers=extra_headers,
        raw_headers=raw_headers,
        client_ip=client_ip,
    )


def combine_optional_catalog(
    request: CatalogListRequest,
    pages: Sequence[CatalogListResult],
    next_cursor: str | None,
    meta: Mapping[str, JsonValue] | None,
) -> CatalogListResult:
    from mcp.types import (
        ListPromptsRequest,
        ListPromptsResult,
        ListResourcesRequest,
        ListResourcesResult,
        ListResourceTemplatesResult,
    )

    from litellm.proxy.mcp_server.faults.list_outcomes import SERVER_OUTCOMES_META_KEY

    outcomes: Final = (meta or {}).get(SERVER_OUTCOMES_META_KEY)
    incomplete: Final = isinstance(outcomes, dict) and any(
        isinstance(value, dict) and value.get("tag") != "ok" for value in outcomes.values()
    )
    ttl_ms: Final = 0 if incomplete else aggregate_freshness(pages).ttl_ms
    if isinstance(request, ListPromptsRequest):
        return ListPromptsResult(
            prompts=list(chain.from_iterable(page.prompts for page in pages if isinstance(page, ListPromptsResult))),
            next_cursor=next_cursor,
            ttl_ms=ttl_ms,
            cache_scope="private",
            _meta=dict(meta) if meta is not None else None,
        )
    if isinstance(request, ListResourcesRequest):
        return ListResourcesResult(
            resources=list(
                chain.from_iterable(page.resources for page in pages if isinstance(page, ListResourcesResult))
            ),
            next_cursor=next_cursor,
            ttl_ms=ttl_ms,
            cache_scope="private",
            _meta=dict(meta) if meta is not None else None,
        )
    return ListResourceTemplatesResult(
        resource_templates=list(
            chain.from_iterable(
                page.resource_templates for page in pages if isinstance(page, ListResourceTemplatesResult)
            )
        ),
        next_cursor=next_cursor,
        ttl_ms=ttl_ms,
        cache_scope="private",
        _meta=dict(meta) if meta is not None else None,
    )


class _Expiring(Protocol):
    @property
    def ttl_ms(self) -> int: ...

    def model_copy(self, *, update: Mapping[str, object] | None = None, deep: bool = False) -> Self: ...


_DiscoveryPage = TypeVar("_DiscoveryPage", bound=_Expiring)
_DiscoveryKey: TypeAlias = tuple[str, str | None]
_DISCOVERY_CACHE_LIMIT: Final = 1024
_DISCOVERY_ENTRY: Final = TypeAdapter(tuple[float, bytes])


class _DiscoveryCache(Generic[_DiscoveryPage]):
    def __init__(
        self,
        ttl: float,
        clock: Callable[[], float],
        adapter: TypeAdapter[_DiscoveryPage],
        max_entry_kb: int = 64,
        storable: Callable[[_DiscoveryPage], bool] | None = None,
    ) -> None:
        self._ttl = ttl
        self._clock = clock
        self._adapter = adapter
        self._storable = storable
        self._entries = InMemoryCache(
            max_size_in_memory=_DISCOVERY_CACHE_LIMIT, max_size_per_item=max_entry_kb, clock=clock
        )
        self._pending: dict[_DiscoveryKey, asyncio.Task[_DiscoveryPage]] = {}
        self._waiters: dict[asyncio.Task[_DiscoveryPage], int] = {}  # mutable-ok: constant-time waiter accounting
        self._oversized_server_ids: frozenset[str] = frozenset()
        self.invalidations = 0

    def _entry_keys(self, server_id: str) -> tuple[str, ...]:
        prefix: Final = f"[{json.dumps(server_id)},"
        entries: Final = cast(  # cast-ok: private cache keys are JSON strings
            "Mapping[str, object]", self._entries.cache_dict
        )
        keys: Final = tuple(entries)
        return tuple(entry_key for entry_key in keys if entry_key.startswith(prefix))

    def holds(self, server_id: str) -> bool:
        return bool(self._entry_keys(server_id))

    def invalidate(self, server_id: str) -> None:
        self.invalidations += 1
        self._oversized_server_ids = self._oversized_server_ids - {server_id}
        for entry_key in self._entry_keys(server_id):
            self._entries.delete_cache(entry_key)
        for key in tuple(self._pending):
            if key[0] == server_id:
                self._pending.pop(key)

    @staticmethod
    def _observe_completion(task: asyncio.Task[_DiscoveryPage]) -> None:
        if not task.cancelled():
            task.exception()

    async def get(
        self, key: _DiscoveryKey, fetch: Callable[[], Awaitable[_DiscoveryPage]], *, capped: bool = True
    ) -> _DiscoveryPage:
        if self._ttl <= 0:
            return await fetch()
        entry: Final[object] = self._entries.get_cache(json.dumps(key))
        if entry is not None:
            expires_at, payload = _DISCOVERY_ENTRY.validate_python(entry)
            remaining: Final = max(0, int((expires_at - self._clock()) * 1000))
            if remaining > 0:
                return self._adapter.validate_json(payload).model_copy(update={"ttl_ms": remaining})
            self._entries.delete_cache(json.dumps(key))
        pending: Final = self._pending.get(key)
        if pending is not None:
            return await self._await_fetch(key, pending)
        if len(self._pending) >= _DISCOVERY_CACHE_LIMIT:
            return await fetch()
        task: Final = asyncio.create_task(self._fetch(key, fetch, capped))
        self._pending[key] = task
        task.add_done_callback(self._observe_completion)
        return await self._await_fetch(key, task)

    async def _await_fetch(self, key: _DiscoveryKey, task: asyncio.Task[_DiscoveryPage]) -> _DiscoveryPage:
        self._waiters[task] = self._waiters.get(task, 0) + 1
        try:
            return (await asyncio.shield(task)).model_copy(deep=True)
        finally:
            remaining: Final = self._waiters[task] - 1
            if remaining:
                self._waiters[task] = remaining
            else:
                self._waiters.pop(task)
                if self._pending.get(key) is task:
                    self._pending.pop(key)
                if not task.done():
                    task.cancel()

    async def _fetch(
        self, key: _DiscoveryKey, fetch: Callable[[], Awaitable[_DiscoveryPage]], capped: bool
    ) -> _DiscoveryPage:
        try:
            items: Final = await fetch()
            if self._pending.get(key) is asyncio.current_task():
                self.keep(key, items, capped=capped)
            return items
        finally:
            if self._pending.get(key) is asyncio.current_task():
                self._pending.pop(key)

    def keep(self, key: _DiscoveryKey, items: _DiscoveryPage, *, capped: bool = True) -> None:
        declared: Final = items.ttl_ms / 1000
        ttl: Final = min(self._ttl, declared) if capped else declared
        if ttl <= 0 or (self._storable is not None and not self._storable(items)):
            return
        entry: Final = _DISCOVERY_ENTRY.dump_json((self._clock() + ttl, self._adapter.dump_json(items)))
        if self._entries.check_value_size(entry):
            self._entries.set_cache(json.dumps(key), entry, ttl=ttl)
            return
        server_id: Final = key[0]
        if server_id in self._oversized_server_ids:
            return
        self._oversized_server_ids = self._oversized_server_ids | {server_id}
        verbose_logger.warning(
            "MCP server %s returned a %s KB listing, above the %s KB cache entry limit. It is fetched on every request",
            server_id,
            len(entry) // 1024,
            self._entries.max_size_per_item,
        )
