"""Authoritative MCP catalog snapshots shared by legacy lookup adapters."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from collections import UserDict
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, MutableMapping, Sequence
from contextlib import ExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import partial, wraps
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Generic, ParamSpec, TypeAlias, TypeVar, cast

from mcp.types import CacheableResult
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy._experimental.mcp_server.result_conversion import age_freshness, aggregate_freshness

if TYPE_CHECKING:
    from mcp.types import ListToolsResult, PaginatedRequestParams, PaginatedResult

    from litellm.proxy._experimental.mcp_server.contracts import CatalogListRequest, CatalogListResult, OperationContext
    from litellm.proxy._experimental.mcp_server.db import OAuthCredentialPayload
    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import AggregateToolListing, ServerOutcome
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
    from litellm.types.mcp_server.mcp_server_manager import MCPServer
    from litellm.types.mcp_server.tool_registry import MCPTool

_P = ParamSpec("_P")
_R = TypeVar("_R")
_Page = TypeVar("_Page", bound="PaginatedResult")
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
        from litellm.proxy._experimental.mcp_server.db import get_mcp_catalog_revision

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
        from litellm.proxy._experimental.mcp_server.tool_registry import global_mcp_tool_registry

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
        from litellm.proxy._experimental.mcp_server.utils import normalize_server_name

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
        from litellm.proxy._experimental.mcp_server.db import get_mcp_catalog_revision
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
        from litellm.proxy._experimental.mcp_server.db import LiteLLM_MCPServerTable
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
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
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            config_ids_capturing_db_identifiers,
            warn_on_shared_identifier_prefixes,
        )
        from litellm.proxy.management_endpoints.mcp_management_endpoints import (
            get_prisma_client_or_throw,
        )

        verbose_logger.debug("Loading MCP servers from database into registry...")

        prisma_client: Final = get_prisma_client_or_throw("Database not connected. Connect a database to your proxy")
        # Load only "active", legacy "approved", and NULL (no approval workflow) rows.
        # Pending/rejected servers are excluded at the DB level so we never load them.
        from litellm.proxy._experimental.mcp_server.db import get_runtime_mcp_server_rows

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
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import global_mcp_server_manager

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
    from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Error
    from litellm.proxy._experimental.mcp_server.state_tokens import open_state, seal_state

    ordered: Final = tuple(sorted(server_ids))
    purpose: Final = "mcp.catalog.list.v1:" + method
    if cursor is None:
        state = _ListingState(
            caller_scope=caller_scope,
            snapshot=snapshot,
            expires_at=now + 600,
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
    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import SERVER_OUTCOMES_META_KEY

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

    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import SERVER_OUTCOMES_META_KEY

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


async def get_filtered_server_tools(
    server: MCPServer,
    *,
    context: OperationContext,
    allowed_mcp_servers: Sequence[MCPServer],
    prefetched_oauth_creds: Mapping[str, OAuthCredentialPayload],
    params: PaginatedRequestParams | None = None,
    record_listing: bool = False,
    listing_updates: ExitStack | None = None,
) -> tuple[ListToolsResult, ServerOutcome]:
    from mcp.types import ListToolsResult

    from litellm.proxy._experimental.mcp_server.exceptions import MCPUpstreamAuthError
    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import ServerListOk, classify_list_exception
    from litellm.proxy._experimental.mcp_server.operations import (
        _get_byok_credential,
        _get_user_oauth_extra_headers_from_db,
        _prepare_mcp_server_headers,
        apply_display_name_overrides,
        filter_tools_by_allowed_tools,
        filter_tools_by_key_team_permissions,
        global_mcp_server_manager,
    )
    from litellm.types.mcp import MCPAuth

    user_api_key_auth, mcp_auth_header, _, mcp_server_auth_headers, oauth2_headers, raw_headers, client_ip = (
        context.legacy_auth()
    )
    mcp_proxy_mode: Final = context.mcp_proxy_mode
    if server is None:
        return ListToolsResult(tools=[]), ServerListOk(tool_count=0)

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
    from litellm.proxy._experimental.mcp_server.outbound_credentials.adapter import (  # noqa: PLC0415
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

    # If still no OAuth2 token, fall back to pre-fetched creds (non-stale-client path)
    elif not migrated_to_v2 and extra_headers is None and server.auth_type == MCPAuth.oauth2:
        extra_headers = await _get_user_oauth_extra_headers_from_db(
            server,
            user_api_key_auth,
            prefetched_creds=prefetched_oauth_creds,
        )

    catalog_auth_header: Final = server_auth_header
    if server.is_byok and server.auth_type != MCPAuth.oauth2 and server_auth_header is None:
        server_auth_header = await _get_byok_credential(server, user_api_key_auth)

    try:
        from litellm.proxy.proxy_server import proxy_logging_obj

        listed_generation: Final = global_mcp_server_manager.listed_tools_generation(server.server_id)
        if params is None:
            page = ListToolsResult(
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
                )
            )
        else:
            page = await global_mcp_server_manager.get_tools_page(
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
                catalog_auth_header=catalog_auth_header,
                record_listing=False,
                listing_updates=listing_updates,
            )
        tools: Final = page.tools
        filtered_tools = filter_tools_by_allowed_tools(tools, server)

        filtered_tools = await filter_tools_by_key_team_permissions(
            tools=filtered_tools,
            server_id=server.server_id,
            user_api_key_auth=user_api_key_auth,
        )

        from litellm.proxy._experimental.mcp_server.mcp_server_manager import ListedToolsCaller
        from litellm.proxy._experimental.mcp_server.utils import strip_known_server_prefix

        record: Final = partial(
            global_mcp_server_manager.record_listed_tools,
            server,
            [tool.model_copy(update={"name": strip_known_server_prefix(tool.name, server)}) for tool in filtered_tools],
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

        if mcp_proxy_mode:
            from litellm.proxy._experimental.mcp_server.tool_search import with_mcp_proxy_identity

            filtered_tools = [with_mcp_proxy_identity(tool, server.server_id) for tool in filtered_tools]
        else:
            filtered_tools = apply_display_name_overrides(filtered_tools, server)

        verbose_logger.debug(
            "Successfully fetched %s tools from server %s, %s after filtering",
            len(tools),
            server.name,
            len(filtered_tools),
        )
        return page.model_copy(update={"tools": filtered_tools}), ServerListOk(tool_count=len(filtered_tools))
    except MCPUpstreamAuthError as e:
        # Absorb so one unauthenticated server does not empty every other server's
        # tools. Surfacing the upstream 401 to the client as a re-auth challenge is
        # intentionally not done here: raising from this list handler cannot produce a
        # 401 + WWW-Authenticate (the MCP session manager serializes it as a JSON-RPC
        # error). Single-server routes surface it via the request-scope preemptive
        # check in _raise_preemptive_401_for_unauthenticated_servers instead.
        verbose_logger.debug("MCP list_tools: omitting %s; it needs upstream auth", server.name)
        return ListToolsResult(tools=[]), classify_list_exception(e)
    except Exception as e:
        verbose_logger.exception("Error getting tools from server %s: %s", server.name, e)
        return ListToolsResult(tools=[]), classify_list_exception(e)


def _caller_scope(context: OperationContext, servers: Sequence[MCPServer]) -> str:
    from litellm.proxy._experimental.mcp_server.utils import upstream_credential_headers

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


async def aggregate_gateway_tools(
    context: OperationContext,
    params: PaginatedRequestParams,
    allowed: Sequence[MCPServer],
    prefetched: Mapping[str, OAuthCredentialPayload],
    *,
    record_listing: bool = False,
    enforce_rate_limits: bool = True,
) -> AggregateToolListing:
    import time

    from mcp.types import ListToolsResult, PaginatedRequestParams
    from pydantic import TypeAdapter

    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import (
        SERVER_OUTCOMES_META_KEY,
        AggregateToolListing,
        ServerOutcome,
        classify_list_exception,
    )
    from litellm.proxy._experimental.mcp_server.operations import (
        _aggregate_server_key,
        _mcp_server_rate_limit_rejection,
        global_mcp_server_manager,
    )
    from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError

    async with global_mcp_server_manager.catalog.operation() as snapshot:
        servers: Final = {server.server_id: server for server in allowed}
        listing_updates: Final = ExitStack()
        rejections: Final[list[ProxyRateLimitError]] = []  # mutable-ok: concurrent fetches share first-page errors

        async def fetch(server_id: str, cursor: str | None) -> ListToolsResult:
            if enforce_rate_limits:
                error: Final = await _mcp_server_rate_limit_rejection(servers[server_id], context.user_api_key_auth)
                if error is not None:
                    if cursor is not None:
                        raise error
                    rejections.append(error)
                    return ListToolsResult(
                        tools=[],
                        _meta={
                            SERVER_OUTCOMES_META_KEY: {
                                _aggregate_server_key(servers[server_id]): classify_list_exception(error).model_dump(
                                    mode="json"
                                )
                            }
                        },
                    )
            result, outcome = await get_filtered_server_tools(
                servers[server_id],
                context=context,
                allowed_mcp_servers=allowed,
                prefetched_oauth_creds=prefetched,
                params=PaginatedRequestParams(cursor=cursor),
                record_listing=record_listing,
                listing_updates=listing_updates,
            )
            if cursor is not None and outcome.tag != "ok":
                from mcp.shared.exceptions import MCPError
                from mcp.types import INVALID_PARAMS

                raise MCPError(code=INVALID_PARAMS, message="Upstream continuation failed; start a fresh listing")
            return result.model_copy(
                update={
                    "meta": {
                        **(result.meta or {}),
                        SERVER_OUTCOMES_META_KEY: {
                            _aggregate_server_key(servers[server_id]): outcome.model_dump(mode="json")
                        },
                    }
                }
            )

        result: Final = await list_tools_page(
            cursor=params.cursor,
            caller_scope=_caller_scope(context, allowed),
            snapshot=snapshot.identity,
            server_ids=tuple(servers),
            fetch=fetch,
            now=int(time.time()),
        )
        if params.cursor is None and servers and len(rejections) == len(servers):
            raise rejections[0]
        listing_updates.close()
        return AggregateToolListing(
            tools=result.tools,
            outcomes=TypeAdapter(dict[str, ServerOutcome]).validate_python(
                (result.meta or {}).get(SERVER_OUTCOMES_META_KEY, {})
            ),
            next_cursor=result.next_cursor,
            ttl_ms=result.ttl_ms,
        )


async def list_gateway_tools(
    context: OperationContext, params: PaginatedRequestParams, *, log_list_tools_to_spendlogs: bool = True
) -> ListToolsResult:
    from mcp.types import ListToolsResult

    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import SERVER_OUTCOMES_META_KEY, outcome_wire_value
    from litellm.proxy._experimental.mcp_server.operations import _list_mcp_tools

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
    return ListToolsResult(
        tools=listing.tools,
        next_cursor=listing.next_cursor,
        ttl_ms=listing.ttl_ms,
        cache_scope="private",
        _meta={
            SERVER_OUTCOMES_META_KEY: {key: outcome_wire_value(outcome) for key, outcome in listing.outcomes.items()}
        }
        if listing.outcomes
        else None,
    )


async def list_gateway_catalog(
    context: OperationContext, request: CatalogListRequest, *, log_list_tools_to_spendlogs: bool = True
) -> CatalogListResult:
    import time

    from mcp.types import (
        ListToolsRequest,
        PaginatedRequestParams,
    )

    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler
    from litellm.proxy._experimental.mcp_server.operations import (
        _get_allowed_mcp_servers,
        global_mcp_server_manager,
        raise_denied_scoped_mcp_access,
    )
    from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError

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

            from litellm.proxy._experimental.mcp_server.faults.list_outcomes import (
                SERVER_OUTCOMES_META_KEY,
                classify_list_exception,
            )
            from litellm.proxy._experimental.mcp_server.operations import (
                _aggregate_server_key,
                _mcp_server_rate_limit_rejection,
            )

            error: Final = await _mcp_server_rate_limit_rejection(server, caller)
            if error is not None:
                if cursor is not None:
                    raise error
                rejections.append(error)
                return combine_optional_catalog(
                    request,
                    (),
                    None,
                    {
                        SERVER_OUTCOMES_META_KEY: {
                            _aggregate_server_key(server): classify_list_exception(error).model_dump(mode="json")
                        }
                    },
                )

            try:
                page: Final = await fetch_optional_catalog_page(context, request, server, allowed, cursor)
                return page.model_copy(
                    update={
                        "meta": {
                            key: value for key, value in (page.meta or {}).items() if key != SERVER_OUTCOMES_META_KEY
                        }
                    }
                )
            except Exception as error:
                if cursor is not None:
                    raise MCPError(
                        code=INVALID_PARAMS, message="Upstream continuation failed; start a fresh listing"
                    ) from error
                return combine_optional_catalog(
                    request,
                    (),
                    None,
                    {
                        "litellm.ai/server_outcomes": {
                            _aggregate_server_key(server): classify_list_exception(error).model_dump(mode="json")
                        }
                    },
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
        from litellm.proxy._experimental.mcp_server.faults.list_outcomes import (
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

    from litellm.proxy._experimental.mcp_server.operations import _prepare_mcp_server_headers, global_mcp_server_manager

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

    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import SERVER_OUTCOMES_META_KEY

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


_DiscoveryPage = TypeVar("_DiscoveryPage", bound=CacheableResult)
_DiscoveryKey: TypeAlias = tuple[str, str | None]
_DISCOVERY_CACHE_LIMIT: Final = 1024
_DISCOVERY_ENTRY: Final = TypeAdapter(tuple[float, bytes])


class _DiscoveryCache(Generic[_DiscoveryPage]):
    def __init__(self, ttl: float, clock: Callable[[], float], adapter: TypeAdapter[_DiscoveryPage]) -> None:
        self._ttl = ttl
        self._clock = clock
        self._adapter = adapter
        self._entries = InMemoryCache(max_size_in_memory=_DISCOVERY_CACHE_LIMIT, max_size_per_item=64, clock=clock)
        self._pending: dict[_DiscoveryKey, asyncio.Task[_DiscoveryPage]] = {}
        self._waiters: dict[asyncio.Task[_DiscoveryPage], int] = {}  # mutable-ok: constant-time waiter accounting

    def invalidate(self, server_id: str) -> None:
        prefix: Final = f"[{json.dumps(server_id)},"
        keys: Final = cast(  # cast-ok: private cache contains only JSON string keys
            "tuple[str, ...]", tuple(self._entries.cache_dict)
        )
        for entry_key in keys:
            if entry_key.startswith(prefix):
                self._entries.delete_cache(entry_key)
        for key in tuple(self._pending):
            if key[0] == server_id:
                self._pending.pop(key)

    @staticmethod
    def _observe_completion(task: asyncio.Task[_DiscoveryPage]) -> None:
        if not task.cancelled():
            task.exception()

    async def get(self, key: _DiscoveryKey, fetch: Callable[[], Awaitable[_DiscoveryPage]]) -> _DiscoveryPage:
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
        task: Final = asyncio.create_task(self._fetch(key, fetch))
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

    async def _fetch(self, key: _DiscoveryKey, fetch: Callable[[], Awaitable[_DiscoveryPage]]) -> _DiscoveryPage:
        try:
            items: Final = await fetch()
            ttl: Final = min(self._ttl, items.ttl_ms / 1000)
            if ttl > 0 and self._pending.get(key) is asyncio.current_task():
                self._entries.set_cache(
                    json.dumps(key),
                    _DISCOVERY_ENTRY.dump_json((self._clock() + ttl, self._adapter.dump_json(items))),
                    ttl=ttl,
                )
            return items
        finally:
            if self._pending.get(key) is asyncio.current_task():
                self._pending.pop(key)
