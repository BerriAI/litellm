import asyncio
import json
import queue
from collections.abc import Awaitable, Callable, Mapping, Sequence
from itertools import count
from types import SimpleNamespace
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest
from mcp.server.context import ServerRequestContext
from mcp.shared.exceptions import MCPError
from mcp.types import (
    ListPromptsRequest,
    ListPromptsResult,
    ListResourcesRequest,
    ListResourcesResult,
    ListResourceTemplatesRequest,
    ListResourceTemplatesResult,
    ListToolsRequest,
    ListToolsResult,
    PaginatedRequestParams,
    Tool,
)
from pydantic import JsonValue, TypeAdapter
from starlette.requests import Request

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.mcp_server import catalog
from litellm.proxy.mcp_server import operations as mcp_operations
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.exceptions import MCPUpstreamAuthError
from litellm.proxy.mcp_server.faults.list_outcomes import (
    SERVER_OUTCOMES_META_KEY,
    AggregateToolListing,
    ServerListFault,
    ServerListOk,
)
from litellm.proxy.mcp_server.mcp_context import active_mcp_request_ctx_var, mcp_gateway_server_name
from litellm.proxy.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy.mcp_server.scope_challenge import SCOPE_RESPONSE_KEY, OAuthScopeResponse
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from tests.unit.proxy.rust_auth_stores import install_rust_auth_stores

CatalogKind = Literal["tools", "prompts", "resources", "templates"]
OptionalCatalogResult = ListPromptsResult | ListResourcesResult | ListResourceTemplatesResult


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selected,revoked,expected",
    [
        ("narrow", False, ("alpha-read_item",)),
        ("full", False, ("alpha-read_item", "alpha-write_item", "beta-read_item", "beta-write_item")),
        (None, False, ("alpha-read_item", "alpha-write_item", "beta-read_item", "beta-write_item")),
        ("narrow", True, ()),
    ],
)
async def test_catalog_refresh_keeps_selected_toolset_with_current_grants_and_contents(
    monkeypatch: pytest.MonkeyPatch,
    selected: str | None,
    revoked: bool,
    expected: tuple[str, ...],
) -> None:
    from fastapi import HTTPException

    from litellm.caching.dual_cache import DualCache
    from litellm.proxy import proxy_server
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable
    from litellm.proxy.mcp_server import operations
    from litellm.proxy.mcp_server.server import apply_toolset_scope
    from litellm.proxy.mcp_server.tool_registry import global_mcp_tool_registry
    from litellm.types.mcp_server.mcp_toolset import MCPToolset, MCPToolsetTool

    monkeypatch.setenv("LITELLM_SALT_KEY", "toolset-scope-pagination-test")
    monkeypatch.setattr(catalog.time, "time", lambda: 1000.0)
    grants: Final = LiteLLM_ObjectPermissionTable(
        object_permission_id="catalog-grants", mcp_toolsets=["narrow", "full"]
    )
    live_grants: Final = grants.model_copy(update={"mcp_toolsets": ["full"]}) if revoked else grants
    tools: Final[tuple[MCPToolsetTool, ...]] = (
        {"server_id": "alpha", "tool_name": "read_item"},
        {"server_id": "alpha", "tool_name": "write_item"},
        {"server_id": "beta", "tool_name": "read_item"},
        {"server_id": "beta", "tool_name": "write_item"},
    )
    toolsets: Final = (
        MCPToolset(toolset_id="narrow", toolset_name="narrow", tools=[tools[0]]),
        MCPToolset(toolset_id="full", toolset_name="full", tools=list(tools)),
    )

    async def read_toolsets(where: Mapping[str, Mapping[str, Sequence[str]]]) -> tuple[MCPToolset, ...]:
        return tuple(toolset for toolset in toolsets if toolset.toolset_id in where["toolset_id"]["in"])

    tables: Final = SimpleNamespace(litellm_mcptoolsettable=SimpleNamespace(find_many=read_toolsets))
    database: Final = SimpleNamespace(db=tables, writer_db=tables)
    stores: Final = install_rust_auth_stores(monkeypatch)
    stores.key.rows["catalog-key-hash"] = {
        "token": "catalog-key-hash",
        "object_permission_id": grants.object_permission_id,
    }
    stores.object_permission.seed(grants.object_permission_id, live_grants)
    cache: Final = DualCache()
    monkeypatch.setattr(proxy_server, "prisma_client", database)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "general_settings", {"supported_db_objects": []})
    await cache.async_set_cache(key="toolset_perms:narrow", value={"alpha": ["read_item", "write_item"]})
    manager: Final = operations.global_mcp_server_manager
    for server_id in ("alpha", "beta"):
        monkeypatch.setitem(
            manager.registry,
            server_id,
            MCPServer(server_id=server_id, name=server_id, transport=MCPTransport.http, spec_path="/catalog.yaml"),
        )

    async def tool_handler() -> str:
        return "ok"

    for tool in tools:
        global_mcp_tool_registry.register_tool(
            name=f"{tool['server_id']}-{tool['tool_name']}",
            description=tool["tool_name"],
            input_schema={"type": "object"},
            handler=tool_handler,
        )
    original: Final = UserAPIKeyAuth(
        api_key="catalog-key-hash",
        object_permission=grants,
        object_permission_id=grants.object_permission_id,
    )
    original.via_virtual_key = True
    caller: Final = await apply_toolset_scope(original, selected) if selected is not None else original
    before: Final = caller.model_dump()
    context: Final = operations.prepare_context(user_api_key_auth=caller)
    if revoked:
        with pytest.raises(HTTPException, match="does not have access to toolset") as denied:
            await catalog.list_gateway_catalog(context, ListToolsRequest(), log_list_tools_to_spendlogs=False)
        assert denied.value.status_code == 403
    else:
        result: Final = await catalog.list_gateway_catalog(
            context, ListToolsRequest(), log_list_tools_to_spendlogs=False
        )
        assert isinstance(result, ListToolsResult)
        assert tuple(sorted(tool.name for tool in result.tools)) == expected
        if selected == "narrow":
            from litellm.proxy.mcp_server.outbound_credentials.result import Error
            from litellm.proxy.mcp_server.state_tokens import seal_state

            servers: Final = tuple(manager.registry[server_id] for server_id in ("alpha", "beta"))
            full_context: Final = operations.prepare_context(user_api_key_auth=original)
            narrow_scope, _ = await catalog._tool_listing_scope(context, servers, {})
            full_scope, _ = await catalog._tool_listing_scope(full_context, servers, {})
            assert narrow_scope != full_scope
            now: Final = 1000
            async with manager.catalog.operation() as snapshot:
                state: Final = catalog._ListingState(
                    caller_scope=full_scope,
                    snapshot=snapshot.identity,
                    expires_at=now + 600,
                    positions=tuple(
                        catalog._UpstreamPosition(
                            server_id=server.server_id,
                            cursor=catalog._tool_page_cursor("a" * 32, 1),
                        )
                        for server in servers
                    ),
                )
            sealed = seal_state(
                catalog._JSON_VALUE.validate_python(state.model_dump(mode="json")),
                purpose="mcp.catalog.list.v1:tools/list",
                expires_at=now + 600,
                now=now,
            )
            assert not isinstance(sealed, Error)
            with pytest.raises(MCPError, match="scope or snapshot changed"):
                await catalog.aggregate_gateway_tools(
                    context,
                    PaginatedRequestParams(cursor=sealed.ok),
                    servers,
                    {},
                    enforce_rate_limits=False,
                    use_tool_list_cache=False,
                )
    assert caller.model_dump() == before
    assert original.object_permission == grants


def page(name: str, cursor: str | None = None, revision: str = "stable") -> ListToolsResult:
    return ListToolsResult(
        tools=[Tool(name=name, input_schema={"type": "object"})],
        next_cursor=cursor,
        meta={"revision": revision},
    )


class MemoryToolPageStore:
    available = True

    def __init__(self) -> None:
        self.values: dict[str, JsonValue] = {}

    async def get(self, key: str) -> JsonValue | None:
        return self.values.get(key)

    async def set(self, key: str, value: JsonValue, _ttl: float) -> None:
        if key in self.values and ":list-status:" not in key:
            raise AssertionError(f"Tool page key was overwritten: {key}")
        self.values[key] = value


def rate_limit_catalog_setup(
    monkeypatch: pytest.MonkeyPatch, rejected_server_ids: frozenset[str]
) -> tuple[tuple[MCPServer, MCPServer], OperationContext, AsyncMock]:
    from litellm.proxy import proxy_server
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.mcp_server import operations

    monkeypatch.setenv("LITELLM_SALT_KEY", "catalog-rate-limit-test-key")
    servers: Final = (
        MCPServer(server_id="catalog-a", name="catalog-a", transport=MCPTransport.http),
        MCPServer(server_id="catalog-b", name="catalog-b", transport=MCPTransport.http),
    )
    caller: Final = UserAPIKeyAuth(api_key="catalog-rate-limit-key", user_id="catalog-rate-limit-user")

    async def enforce_rate_limit(_user: UserAPIKeyAuth | None, server: MCPServer) -> None:
        if server.server_id in rejected_server_ids:
            raise ProxyRateLimitError(detail=f"{server.server_id} RPM exceeded")

    limiter: Final = AsyncMock(side_effect=enforce_rate_limit)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", SimpleNamespace(enforce_mcp_server_rate_limits=limiter))
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    monkeypatch.setattr(
        operations.global_mcp_server_manager, "registry", {server.server_id: server for server in servers}
    )
    monkeypatch.setattr(operations, "_get_allowed_mcp_servers", AsyncMock(return_value=list(servers)))
    context: Final = operations.prepare_context(
        user_api_key_auth=caller,
        mcp_servers=[server.server_id for server in servers],
    )
    return servers, context, limiter


def optional_catalog_page(kind: CatalogKind, server_id: str, next_cursor: str | None = None) -> OptionalCatalogResult:
    from mcp import types

    if kind == "prompts":
        return types.ListPromptsResult(prompts=[types.Prompt(name=f"{server_id}-item")], next_cursor=next_cursor)
    if kind == "resources":
        return types.ListResourcesResult(
            resources=[types.Resource(name=f"{server_id}-item", uri=f"https://example.com/{server_id}")],
            next_cursor=next_cursor,
        )
    return types.ListResourceTemplatesResult(
        resource_templates=[
            types.ResourceTemplate(name=f"{server_id}-item", uri_template=f"https://example.com/{server_id}/{{name}}")
        ],
        next_cursor=next_cursor,
    )


async def run_catalog_listing(
    kind: CatalogKind,
    context: OperationContext,
    servers: Sequence[MCPServer],
    cursor: str | None = None,
) -> AggregateToolListing | OptionalCatalogResult:
    from mcp import types

    if kind == "tools":
        return await catalog.aggregate_gateway_tools(context, PaginatedRequestParams(cursor=cursor), servers, {})
    request: Final = {
        "prompts": types.ListPromptsRequest,
        "resources": types.ListResourcesRequest,
        "templates": types.ListResourceTemplatesRequest,
    }[kind](params=PaginatedRequestParams(cursor=cursor))
    return await catalog.list_gateway_catalog(context, request)


async def listing(
    fetch,
    cursor: str | None = None,
    *,
    caller_scope: str = "caller-and-scope",
    snapshot: str = "registry-generation",
    servers: Sequence[str] = ("a", "b"),
    now: int = 100,
) -> ListToolsResult:
    return await catalog.list_tools_page(
        cursor=cursor,
        caller_scope=caller_scope,
        snapshot=snapshot,
        server_ids=tuple(servers),
        fetch=fetch,
        now=now,
    )


@pytest.mark.asyncio
async def test_listing_continues_on_another_replica_and_cursor_is_reusable(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "shared-test-key")

    async def replica_a(server_id: str, cursor: str | None) -> ListToolsResult:
        assert cursor is None
        return page(server_id + "1", "page-two" if server_id == "a" else None)

    async def replica_b(server_id: str, cursor: str | None) -> ListToolsResult:
        assert (server_id, cursor) == ("a", "page-two")
        return page("a2")

    first = await listing(replica_a, servers=("b", "a"))
    assert [tool.name for tool in first.tools] == ["a1", "b1"]
    assert first.next_cursor
    for _ in range(2):
        second = await listing(replica_b, first.next_cursor)
        assert [tool.name for tool in second.tools] == ["a2"]
        assert second.next_cursor is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"caller_scope": "other-caller"},
        {"snapshot": "new-registry-generation"},
        {"servers": ("b",)},
        {"now": 1000},
    ],
)
async def test_continuation_rejects_changed_binding_before_upstream_dispatch(monkeypatch, changes):
    monkeypatch.setenv("LITELLM_SALT_KEY", "shared-test-key")

    async def first_page(server_id: str, cursor: str | None) -> ListToolsResult:
        return page(server_id, "next")

    async def forbidden_dispatch(server_id: str, cursor: str | None) -> ListToolsResult:
        pytest.fail("Rejected continuation must not contact upstream")

    first = await listing(first_page)
    with pytest.raises(MCPError, match=r"fresh listing|expired"):
        await listing(forbidden_dispatch, first.next_cursor, **changes)


@pytest.mark.asyncio
async def test_complete_single_page_needs_no_key_but_continuation_does(monkeypatch):
    monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)

    async def complete(server_id: str, cursor: str | None) -> ListToolsResult:
        return page(server_id)

    async def incomplete(server_id: str, cursor: str | None) -> ListToolsResult:
        return page(server_id, "next")

    result = await listing(complete)
    assert len(result.tools) == 2
    assert result.next_cursor is None
    with pytest.raises(MCPError, match="LITELLM_SALT_KEY"):
        await listing(incomplete)


@pytest.mark.asyncio
async def test_repeated_upstream_cursor_requires_restart(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "shared-test-key")

    async def repeated(server_id: str, cursor: str | None) -> ListToolsResult:
        return page(server_id, "repeated")

    first = await listing(repeated)
    with pytest.raises(MCPError, match=r"repeated.*cursor"):
        await listing(repeated, first.next_cursor)


@pytest.mark.asyncio
async def test_changed_upstream_revision_requires_restart(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "shared-test-key")

    async def original(server_id: str, cursor: str | None) -> ListToolsResult:
        return page(server_id, "next")

    async def revised(server_id: str, cursor: str | None) -> ListToolsResult:
        return page(server_id, revision="changed")

    first = await listing(original)
    with pytest.raises(MCPError, match="fresh listing"):
        await listing(revised, first.next_cursor)


@pytest.mark.asyncio
async def test_failed_upstream_remains_visible_when_other_sources_continue(monkeypatch):
    from litellm.proxy.mcp_server.faults.list_outcomes import SERVER_OUTCOMES_META_KEY

    monkeypatch.setenv("LITELLM_SALT_KEY", "shared-test-key")

    async def fetch(server_id: str, cursor: str | None) -> ListToolsResult:
        if server_id == "a":
            return ListToolsResult(tools=[], meta={SERVER_OUTCOMES_META_KEY: {"a": {"tag": "timeout"}}})
        return page("b2" if cursor else "b1", None if cursor else "next").model_copy(update={"ttl_ms": 9000})

    first = await listing(fetch)
    assert [tool.name for tool in first.tools] == ["b1"]
    assert first.meta[SERVER_OUTCOMES_META_KEY]["a"] == {"tag": "timeout"}
    assert first.ttl_ms == 0
    second = await listing(fetch, first.next_cursor)
    assert [tool.name for tool in second.tools] == ["b2"]
    assert second.meta[SERVER_OUTCOMES_META_KEY]["a"] == {"tag": "timeout"}
    assert second.ttl_ms == 0


@pytest.mark.asyncio
async def test_cursor_cannot_be_reused_for_a_different_catalog_kind(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "shared-test-key")

    async def first_page(server_id: str, cursor: str | None) -> ListToolsResult:
        return page(server_id, "next")

    async def forbidden_dispatch(server_id: str, cursor: str | None) -> ListToolsResult:
        pytest.fail("Wrong-kind cursor must not contact upstream")

    first = await listing(first_page)
    with pytest.raises(MCPError, match="Invalid pagination state"):
        await catalog.paginate_catalog(
            method="resources/list",
            cursor=first.next_cursor,
            caller_scope="caller-and-scope",
            snapshot="registry-generation",
            server_ids=("a", "b"),
            fetch=forbidden_dispatch,
            now=100,
        )


@pytest.mark.asyncio
async def test_authenticated_state_with_invalid_catalog_schema_is_rejected(monkeypatch):
    from litellm.proxy.mcp_server.outbound_credentials.result import Ok
    from litellm.proxy.mcp_server.state_tokens import seal_state

    monkeypatch.setenv("LITELLM_SALT_KEY", "shared-test-key")
    token = seal_state({"not": "catalog state"}, purpose="mcp.catalog.list.v1:tools/list", expires_at=200, now=100)
    assert isinstance(token, Ok)

    async def forbidden_dispatch(server_id, cursor):
        pytest.fail("Malformed catalog state must not dispatch")

    with pytest.raises(MCPError, match="Invalid pagination state"):
        await listing(forbidden_dispatch, token.ok)


@pytest.mark.asyncio
async def test_following_page_does_not_extend_original_expiry(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "shared-test-key")

    async def fetch(server_id, cursor):
        return page(server_id, "second" if cursor is None else "third")

    first = await listing(fetch, now=100)
    second = await listing(fetch, first.next_cursor, now=699)
    assert second.next_cursor
    with pytest.raises(MCPError, match="expired"):
        await listing(fetch, second.next_cursor, now=700)


@pytest.mark.asyncio
async def test_page_limit_rejects_an_unending_upstream(monkeypatch):
    from litellm import constants

    monkeypatch.setenv("LITELLM_SALT_KEY", "shared-test-key")
    monkeypatch.setattr(constants, "MCP_TOOL_LISTING_MAX_PAGES", 2)

    async def fetch(server_id, cursor):
        return page(server_id, "second" if cursor is None else "third")

    first = await listing(fetch)
    with pytest.raises(MCPError, match="limit"):
        await listing(fetch, first.next_cursor)


@pytest.mark.parametrize(
    "change",
    [
        {"_caller": None},
        {"mcp_auth_header": "another upstream credential"},
        {"mcp_servers": ("another-scope",)},
        {"client_ip": "192.0.2.2"},
        {"raw_headers": {"Authorization": "another bearer"}},
        {"oauth2_headers": {"Authorization": "another upstream bearer"}},
        {"mcp_server_auth_headers": {"a": {"Authorization": "another per-server bearer"}}},
        {"protocol_version": "2024-11-05"},
        {"mcp_proxy_mode": True},
    ],
)
def test_caller_binding_covers_identity_scope_and_forwarded_credentials(change):
    from dataclasses import replace

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.mcp_server.contracts import OperationContext

    original = OperationContext(_caller=UserAPIKeyAuth(api_key="synthetic-key", user_id="owner"))
    assert catalog._caller_scope(original, ()) != catalog._caller_scope(replace(original, **change), ())


def test_caller_binding_ignores_transport_headers_and_header_case():
    from dataclasses import replace

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.mcp_server.contracts import OperationContext

    original = OperationContext(
        _caller=UserAPIKeyAuth(user_id="owner"), raw_headers={"Authorization": "Bearer synthetic"}
    )
    retry = replace(original, raw_headers={"authorization": "Bearer synthetic", "mcp-session-id": "replica-b-session"})
    assert catalog._caller_scope(original, ()) == catalog._caller_scope(retry, ())


@pytest.mark.asyncio
async def test_upstream_pages_overlap_and_keep_deterministic_order(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "shared-test-key")
    both_started = asyncio.Event()
    started = set()

    async def fetch(server_id, cursor):
        started.add(server_id)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        return page(server_id)

    result = await listing(fetch, servers=("b", "a"))
    assert [tool.name for tool in result.tools] == ["a", "b"]


@pytest.mark.asyncio
async def test_cursor_history_allows_the_full_supported_page_count(monkeypatch):
    from litellm.constants import MCP_TOOL_LISTING_MAX_PAGES

    monkeypatch.setenv("LITELLM_SALT_KEY", "shared-test-key")

    async def fetch(server_id, cursor):
        index = int(cursor or "0")
        return page(str(index), str(index + 1) if index + 1 < MCP_TOOL_LISTING_MAX_PAGES else None)

    cursor = None
    for index in range(MCP_TOOL_LISTING_MAX_PAGES):
        result = await listing(fetch, cursor, servers=("a",))
        assert result.tools[0].name == str(index)
        cursor = result.next_cursor
        assert bool(cursor) == (index + 1 < MCP_TOOL_LISTING_MAX_PAGES)


@pytest.mark.asyncio
async def test_failed_page_cancels_and_drains_other_upstream_requests(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "shared-test-key")
    pending_started = asyncio.Event()
    pending_closed = asyncio.Event()

    async def fetch(server_id, cursor):
        if server_id == "a":
            await asyncio.wait_for(pending_started.wait(), timeout=1)
            raise ValueError("upstream unavailable")
        pending_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            pending_closed.set()
        return page(server_id)

    with pytest.raises(ValueError, match="upstream unavailable"):
        await listing(fetch)
    assert pending_closed.is_set()


@pytest.mark.asyncio
async def test_gateway_tools_walks_session_bound_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setenv("LITELLM_SALT_KEY", "session-bound-tools-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    transport, calls = _session_bound_tools_transport()
    server = MCPServer(
        server_id="bound",
        name="bound",
        url="https://bound.example/mcp",
        transport=MCPTransport.http,
        protocol_version="2025-11-25",
    )
    manager = MCPServerManager(upstream_transport=transport)
    monkeypatch.setattr(manager, "registry", {server.server_id: server})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)

    result: Final = await catalog.aggregate_gateway_tools(
        mcp_operations.prepare_context(),
        PaginatedRequestParams(),
        [server],
        {},
        enforce_rate_limits=False,
        use_tool_list_cache=False,
    )

    assert [tool.name for tool in result.tools] == ["bound-bound-tool-0", "bound-bound-tool-1"]
    assert result.next_cursor is None
    requests: Final = tuple(calls.get_nowait() for _ in range(calls.qsize()))
    tool_requests: Final = tuple(request for request in requests if request[1] == "tools/list")
    assert len(tool_requests) == 2
    assert tool_requests[0][2] is not None
    assert tool_requests[0][3] is None
    assert tool_requests[1][2] == tool_requests[0][2]
    assert tool_requests[1][3] == f"{tool_requests[0][2]}:1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "salt_key",
    (pytest.param("session-bound-tools-pages-test", id="salt-present"), pytest.param(None, id="salt-missing")),
)
async def test_gateway_tools_without_redis_returns_complete_session_walk_without_cursor(
    monkeypatch: pytest.MonkeyPatch, salt_key: str | None
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    if salt_key is None:
        monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
    else:
        monkeypatch.setenv("LITELLM_SALT_KEY", salt_key)
    monkeypatch.setattr(catalog.time, "time", lambda: 1000.0)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    transport, calls = _session_bound_tools_transport(tool_count=205, page_size=30, ttl_ms=0)
    server = MCPServer(
        server_id="bound",
        name="bound",
        url="https://bound.example/mcp",
        transport=MCPTransport.http,
        protocol_version="2025-11-25",
    )
    manager = MCPServerManager(upstream_transport=transport)
    monkeypatch.setattr(manager, "registry", {server.server_id: server})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)
    context = mcp_operations.prepare_context(UserAPIKeyAuth(api_key="tools-page-caller"))
    prefetched = {"bound": {"access_token": "token-a"}}

    first = await catalog.aggregate_gateway_tools(
        context,
        PaginatedRequestParams(),
        [server],
        prefetched,
        enforce_rate_limits=False,
        use_tool_list_cache=False,
    )

    assert tuple(tool.name for tool in first.tools) == tuple(f"bound-bound-tool-{index}" for index in range(205))
    assert first.next_cursor is None
    assert first.ttl_ms == 0
    assert first.outcomes["bound"] == ServerListOk(tool_count=205)
    initial_requests: Final = tuple(calls.get_nowait() for _ in range(calls.qsize()))
    upstream_pages: Final = tuple(request for request in initial_requests if request[1] == "tools/list")
    assert len(upstream_pages) == (205 + 29) // 30
    assert len({request[2] for request in upstream_pages}) == 1
    assert tuple(request[3] for request in upstream_pages) == (
        None,
        *(f"{upstream_pages[0][2]}:{offset}" for offset in range(30, 205, 30)),
    )

    assert calls.empty()


@pytest.mark.asyncio
async def test_redis_tool_pages_replay_and_reject_changed_caller_or_oauth_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setenv("LITELLM_SALT_KEY", "session-bound-tools-pages-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(catalog, "_TOOL_PAGE_STORE", MemoryToolPageStore())
    transport, calls = _session_bound_tools_transport(tool_count=3, page_size=1)
    server = MCPServer(server_id="bound", name="bound", url="https://bound.example/mcp", transport=MCPTransport.http)
    manager = MCPServerManager(upstream_transport=transport)
    monkeypatch.setattr(manager, "registry", {server.server_id: server})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)
    context = mcp_operations.prepare_context(UserAPIKeyAuth(api_key="tools-page-caller"))
    prefetched = {"bound": {"access_token": "token-a"}}

    async def listing(caller, credentials, cursor=None):
        return await catalog.aggregate_gateway_tools(
            caller,
            PaginatedRequestParams(cursor=cursor),
            [server],
            credentials,
            enforce_rate_limits=False,
            use_tool_list_cache=False,
        )

    first = await listing(context, prefetched)
    assert first.next_cursor
    await asyncio.gather(*tuple(catalog._TOOL_PRODUCERS.values()))
    initial = tuple(calls.get_nowait() for _ in range(calls.qsize()))
    assert len([call for call in initial if call[1] == "tools/list"]) == 3
    second = await listing(context, prefetched, first.next_cursor)
    replay = await listing(context, prefetched, first.next_cursor)
    assert tuple(tool.name for tool in second.tools) == ("bound-bound-tool-1",)
    assert replay.tools == second.tools
    assert second.next_cursor
    last = await listing(context, prefetched, second.next_cursor)
    assert tuple(tool.name for tool in last.tools) == ("bound-bound-tool-2",)
    assert last.next_cursor is None
    with pytest.raises(MCPError, match="Pagination scope or snapshot changed"):
        await listing(
            mcp_operations.prepare_context(UserAPIKeyAuth(api_key="different-caller")), prefetched, first.next_cursor
        )
    with pytest.raises(MCPError, match="Pagination scope or snapshot changed"):
        await listing(context, {"bound": {"access_token": "token-b"}}, first.next_cursor)
    assert calls.empty()


@pytest.mark.asyncio
async def test_gateway_tools_rejects_stale_cursor_before_upstream_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.mcp_server.outbound_credentials.result import Error
    from litellm.proxy.mcp_server.state_tokens import seal_state

    monkeypatch.setenv("LITELLM_SALT_KEY", "stale-tool-cursor-test")
    monkeypatch.setattr(catalog.time, "time", lambda: 1000.0)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    transport, calls = _session_bound_tools_transport()
    server = MCPServer(
        server_id="bound",
        name="bound",
        url="https://bound.example/mcp",
        transport=MCPTransport.http,
    )
    manager = MCPServerManager(upstream_transport=transport)
    monkeypatch.setattr(manager, "registry", {server.server_id: server})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)
    context = mcp_operations.prepare_context()
    caller_scope, _ = await catalog._tool_listing_scope(context, [server], {})
    async with manager.catalog.operation() as snapshot:
        state = catalog._ListingState(
            caller_scope=caller_scope,
            snapshot=snapshot.identity,
            expires_at=1600,
            positions=(catalog._UpstreamPosition(server_id=server.server_id, cursor="bound-session:1"),),
        )
    sealed = seal_state(
        catalog._JSON_VALUE.validate_python(state.model_dump(mode="json")),
        purpose="mcp.catalog.list.v1:tools/list",
        expires_at=1600,
        now=1000,
    )
    assert not isinstance(sealed, Error)

    with pytest.raises(MCPError, match="page is unavailable; start a fresh listing") as rejected:
        await catalog.aggregate_gateway_tools(
            context,
            PaginatedRequestParams(cursor=sealed.ok),
            [server],
            {},
            enforce_rate_limits=False,
            use_tool_list_cache=False,
        )

    assert rejected.value.code == -32602
    assert calls.empty()


@pytest.mark.asyncio
async def test_gateway_tools_without_redis_keeps_healthy_full_walk_when_peer_page_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    monkeypatch.setenv("LITELLM_SALT_KEY", "failed-page-two-tools-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    transport, calls = _session_bound_tools_transport(
        failing_servers=frozenset({"broken"}), tool_count=205, page_size=30, ttl_ms=0
    )
    servers: Final = (
        MCPServer(
            server_id="healthy",
            name="healthy",
            url="https://healthy.example/mcp",
            transport=MCPTransport.http,
        ),
        MCPServer(
            server_id="broken",
            name="broken",
            url="https://broken.example/mcp",
            transport=MCPTransport.http,
        ),
    )
    manager = MCPServerManager(upstream_transport=transport)
    monkeypatch.setattr(manager, "registry", {server.server_id: server for server in servers})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)

    result: Final = await catalog.aggregate_gateway_tools(
        mcp_operations.prepare_context(),
        PaginatedRequestParams(),
        servers,
        {},
        record_listing=True,
        enforce_rate_limits=False,
        use_tool_list_cache=False,
    )

    assert tuple(tool.name for tool in result.tools) == tuple(f"healthy-healthy-tool-{index}" for index in range(205))
    assert result.next_cursor is None
    assert result.ttl_ms == 0
    assert result.outcomes["broken"].tag != "ok"
    assert "untrusted upstream message" not in str(result.outcomes["broken"].model_dump())
    assert manager.get_listed_tool(servers[0], "healthy-tool-0") is not None
    assert manager.get_listed_tool(servers[0], "healthy-tool-204") is not None
    assert manager.get_listed_tool(servers[1], "broken-tool-0") is None
    requests: Final = tuple(calls.get_nowait() for _ in range(calls.qsize()))
    tool_requests: Final = tuple(request for request in requests if request[1] == "tools/list")
    healthy_requests: Final = tuple(request for request in tool_requests if request[0] == "healthy")
    broken_requests: Final = tuple(request for request in tool_requests if request[0] == "broken")
    assert len(healthy_requests) == (205 + 29) // 30
    assert len(broken_requests) == 2
    assert len({request[2] for request in healthy_requests}) == 1
    assert healthy_requests[0][3] is None
    assert healthy_requests[1][3] == f"{healthy_requests[0][2]}:30"
    assert len({request[2] for request in broken_requests}) == 1
    assert broken_requests[0][3] is None
    assert broken_requests[1][3] == f"{broken_requests[0][2]}:30"

    assert calls.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_state", ("missing", "expired"))
async def test_gateway_tool_page_missing_or_expired_requires_restart(
    monkeypatch: pytest.MonkeyPatch, cache_state: str
) -> None:
    monkeypatch.setattr(catalog.time, "time", lambda: 1000.0)
    snapshot_id = "a" * 32
    server_id = "bound"
    caller_scope = "b" * 64
    store = MemoryToolPageStore()
    page_index = 1
    page_key = catalog._tool_page_cache_key(caller_scope, server_id, snapshot_id, page_index)
    await store.set(
        catalog._tool_page_status_key(caller_scope, server_id, snapshot_id),
        catalog._ToolProducerStatus(state="complete", deadline_at=1000, expires_at=1600, next_page_index=2).model_dump(
            mode="json"
        ),
        600,
    )
    if cache_state == "missing":
        assert await store.get(page_key) is None
    else:
        await store.set(
            page_key,
            catalog._CachedToolPage(
                server_id=server_id,
                snapshot_id=snapshot_id,
                page_index=page_index,
                captured_at=900,
                expires_at=999,
                result=ListToolsResult(tools=[]),
            ).model_dump(mode="json"),
            600,
        )

    with pytest.raises(MCPError, match="start a fresh listing") as rejected:
        await catalog._read_tool_page(
            catalog._tool_page_cursor(snapshot_id, page_index),
            server_id=server_id,
            server_key=server_id,
            caller_scope=caller_scope,
            store=store,
            initial_read_timeout=30,
        )
    assert rejected.value.code == -32602


@pytest.mark.asyncio
async def test_gateway_continuation_uses_server_timeout_for_pending_page_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    from litellm.proxy import proxy_server
    from litellm.proxy.mcp_server import operations as mcp_operations
    from litellm.proxy.mcp_server.outbound_credentials.result import Error
    from litellm.proxy.mcp_server.state_tokens import seal_state

    monkeypatch.setenv("LITELLM_SALT_KEY", "server-timeout-page-read-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", object())
    server: Final = MCPServer(
        server_id="slow",
        name="slow",
        url="https://slow.example/mcp",
        timeout=60,
        transport=MCPTransport.http,
    )
    manager: Final = MCPServerManager()
    monkeypatch.setattr(manager, "registry", {server.server_id: server})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)
    context: Final = mcp_operations.prepare_context()
    now: Final = int(time.time())
    snapshot_id: Final = "e" * 32

    async with manager.catalog.operation() as snapshot:
        caller_scope, _prepared = await catalog._tool_listing_scope(context, (server,), {})
        state: Final = catalog._ListingState(
            caller_scope=caller_scope,
            snapshot=snapshot.identity,
            expires_at=now + 600,
            positions=(
                catalog._UpstreamPosition(
                    server_id=server.server_id,
                    cursor=catalog._tool_page_cursor(snapshot_id, 1),
                    revision="fixture-v1",
                ),
            ),
        )
    sealed: Final = seal_state(
        catalog._JSON_VALUE.validate_python(state.model_dump(mode="json")),
        purpose="mcp.catalog.list.v1:tools/list",
        expires_at=now + 600,
        now=now,
    )
    assert not isinstance(sealed, Error)

    clock: Final = SimpleNamespace(monotonic=0.0, wall=time.time())
    page_key: Final = catalog._tool_page_cache_key(caller_scope, server.server_id, snapshot_id, 1)
    status_key: Final = catalog._tool_page_status_key(caller_scope, server.server_id, snapshot_id)

    class DelayedPageStore(MemoryToolPageStore):
        async def get(self, key: str) -> JsonValue | None:
            if key == page_key and clock.monotonic < 15:
                return None
            return await super().get(key)

    store: Final = DelayedPageStore()
    store.values[page_key] = catalog._CachedToolPage(
        server_id=server.server_id,
        snapshot_id=snapshot_id,
        page_index=1,
        captured_at=clock.wall,
        expires_at=clock.wall + 120,
        result=page("continued", revision="fixture-v1"),
    ).model_dump(mode="json")
    store.values[status_key] = catalog._ToolProducerStatus(
        state="pending",
        deadline_at=clock.wall + 60,
        expires_at=clock.wall + 120,
        next_page_index=1,
        meta={"revision": "fixture-v1"},
    ).model_dump(mode="json")
    monkeypatch.setattr(catalog, "_TOOL_PAGE_STORE", store)
    monkeypatch.setattr("litellm.constants.MCP_TOOL_LISTING_TIMEOUT", 10)
    monkeypatch.setattr(catalog, "_TOOL_PAGE_POLL_INTERVAL", 5)
    monkeypatch.setattr(
        catalog,
        "time",
        SimpleNamespace(time=lambda: clock.wall, monotonic=lambda: clock.monotonic),
    )

    async def advance_time(seconds: float) -> None:
        clock.wall += seconds
        clock.monotonic += seconds

    monkeypatch.setattr(catalog.asyncio, "sleep", advance_time)

    result: Final = await catalog.aggregate_gateway_tools(
        context,
        PaginatedRequestParams(cursor=sealed.ok),
        (server,),
        {},
        enforce_rate_limits=False,
        use_tool_list_cache=False,
    )

    assert tuple(tool.name for tool in result.tools) == ("continued",)
    assert result.next_cursor is None
    assert clock.monotonic == 15


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deadline_offset", "expiry_offset"),
    (
        pytest.param(60, 12, id="page-expiry"),
        pytest.param(12, 60, id="dead-worker-deadline"),
    ),
)
async def test_gateway_pending_page_read_stops_at_status_deadline(
    monkeypatch: pytest.MonkeyPatch,
    deadline_offset: int,
    expiry_offset: int,
) -> None:
    import time

    snapshot_id: Final = "f" * 32
    server_id = "bounded"
    server_key = server_id
    caller_scope = "c" * 64
    clock: Final = SimpleNamespace(monotonic=0.0, wall=time.time())
    page_key: Final = catalog._tool_page_cache_key(caller_scope, server_id, snapshot_id, 1)
    status_key: Final = catalog._tool_page_status_key(caller_scope, server_id, snapshot_id)
    base_time: Final = clock.wall

    class DelayedPageStore(MemoryToolPageStore):
        async def get(self, key: str) -> JsonValue | None:
            if key == page_key and clock.monotonic < 15:
                return None
            return await super().get(key)

    store: Final = DelayedPageStore()
    store.values[page_key] = catalog._CachedToolPage(
        server_id=server_id,
        snapshot_id=snapshot_id,
        page_index=1,
        captured_at=base_time,
        expires_at=base_time + 60,
        result=page("too-late"),
    ).model_dump(mode="json")
    store.values[status_key] = catalog._ToolProducerStatus(
        state="pending",
        deadline_at=base_time + deadline_offset,
        expires_at=base_time + expiry_offset,
        next_page_index=1,
    ).model_dump(mode="json")
    monkeypatch.setattr(catalog, "_TOOL_PAGE_POLL_INTERVAL", 5)
    monkeypatch.setattr(
        catalog,
        "time",
        SimpleNamespace(time=lambda: clock.wall, monotonic=lambda: clock.monotonic),
    )

    async def advance_time(seconds: float) -> None:
        clock.wall += seconds
        clock.monotonic += seconds

    monkeypatch.setattr(catalog.asyncio, "sleep", advance_time)

    with pytest.raises(MCPError, match="start a fresh listing") as rejected:
        await catalog._read_tool_page(
            catalog._tool_page_cursor(snapshot_id, 1),
            server_id=server_id,
            server_key=server_key,
            caller_scope=caller_scope,
            store=store,
            initial_read_timeout=60,
        )

    assert rejected.value.code == -32602
    assert clock.monotonic == 12


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "tool_count", "page_size"),
    (
        pytest.param("start", 3, 1, id="start-status-write"),
        pytest.param("first-page", 3, 1, id="first-page-write"),
        pytest.param("final-single-page", 3, 3, id="final-single-page-write"),
    ),
)
async def test_gateway_tool_page_publication_failure_falls_back_to_complete_listing(
    monkeypatch: pytest.MonkeyPatch, failure: str, tool_count: int, page_size: int
) -> None:
    from litellm.proxy import proxy_server

    class FailedToolPageStore(MemoryToolPageStore):
        available = True

        def __init__(self) -> None:
            super().__init__()

        async def set(self, key: str, value: JsonValue, ttl: float) -> None:
            is_status_write: Final = ":list-status:" in key
            is_first_page_write: Final = ":list-page:" in key and key.endswith(":0")
            if (failure == "start" and is_status_write) or (
                failure in ("first-page", "final-single-page") and is_first_page_write
            ):
                raise OSError("cache unavailable")
            await super().set(key, value, ttl)

    monkeypatch.setenv("LITELLM_SALT_KEY", "tool-page-publish-failure-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", object())
    monkeypatch.setattr(catalog, "_TOOL_PAGE_STORE", FailedToolPageStore())
    transport, calls = _session_bound_tools_transport(tool_count=tool_count, page_size=page_size, ttl_ms=0)
    server = MCPServer(
        server_id="bound",
        name="bound",
        url="https://bound.example/mcp",
        transport=MCPTransport.http,
    )
    manager = MCPServerManager(upstream_transport=transport)
    monkeypatch.setattr(manager, "registry", {server.server_id: server})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)

    result: Final = await catalog.aggregate_gateway_tools(
        mcp_operations.prepare_context(),
        PaginatedRequestParams(),
        [server],
        {},
        record_listing=True,
        enforce_rate_limits=False,
        use_tool_list_cache=False,
    )

    assert tuple(tool.name for tool in result.tools) == tuple(
        f"bound-bound-tool-{index}" for index in range(tool_count)
    )
    assert result.next_cursor is None
    assert result.outcomes["bound"].tag == "ok"
    assert manager.get_listed_tool(server, "bound-tool-0") is not None
    requests: Final = tuple(calls.get_nowait() for _ in range(calls.qsize()))
    tool_requests: Final = tuple(request for request in requests if request[1] == "tools/list")
    assert len(tool_requests) == (tool_count + page_size - 1) // page_size
    assert len({request[2] for request in tool_requests}) == 1
    assert calls.empty()


@pytest.mark.asyncio
async def test_gateway_tool_page_producer_timeout_bounds_initial_redis_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm import constants
    from litellm.proxy import proxy_server

    started = asyncio.Event()

    class HangingToolPageStore:
        available = True

        async def get(self, _key: str) -> object | None:
            return None

        async def set(self, _key: str, _value: object, _ttl: float) -> None:
            started.set()
            await asyncio.Event().wait()

    monkeypatch.setenv("LITELLM_SALT_KEY", "tool-page-timeout-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", object())
    monkeypatch.setattr(catalog, "_TOOL_PAGE_STORE", HangingToolPageStore())
    monkeypatch.setattr(constants, "MCP_TOOL_LISTING_TIMEOUT", 0.01)
    transport, calls = _session_bound_tools_transport()
    server = MCPServer(
        server_id="bound",
        name="bound",
        url="https://bound.example/mcp",
        transport=MCPTransport.http,
    )
    manager = MCPServerManager(upstream_transport=transport)
    monkeypatch.setattr(manager, "registry", {server.server_id: server})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)

    with pytest.raises(MCPError, match="producer stopped before its first page") as rejected:
        await catalog.aggregate_gateway_tools(
            mcp_operations.prepare_context(),
            PaginatedRequestParams(),
            [server],
            {},
            record_listing=True,
            enforce_rate_limits=False,
            use_tool_list_cache=False,
        )

    assert rejected.value.code == -32603
    assert started.is_set()
    assert manager.get_listed_tool(server, "bound-tool-0") is None
    assert calls.empty()


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_tool_walk_and_closes_its_session(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setenv("LITELLM_SALT_KEY", "shutdown-tool-producer-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(catalog, "_TOOL_PAGE_STORE", MemoryToolPageStore())
    transport, calls = _session_bound_tools_transport()
    blocked = asyncio.Event()

    async def respond(request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST":
            payload = json.loads(request.content)
            if payload.get("method") == "tools/list" and (payload.get("params") or {}).get("cursor"):
                blocked.set()
                await asyncio.Event().wait()
        return await transport.handle_async_request(request)

    server = MCPServer(server_id="bound", name="bound", url="https://bound.example/mcp", transport=MCPTransport.http)
    manager = MCPServerManager(upstream_transport=httpx2.MockTransport(respond))
    monkeypatch.setattr(manager, "registry", {server.server_id: server})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)
    first = await catalog.aggregate_gateway_tools(
        mcp_operations.prepare_context(),
        PaginatedRequestParams(),
        [server],
        {},
        record_listing=True,
        enforce_rate_limits=False,
        use_tool_list_cache=False,
    )
    assert first.next_cursor
    await blocked.wait()
    await catalog.shutdown_tool_page_producers()
    requests = tuple(calls.get_nowait() for _ in range(calls.qsize()))
    initialized = {session for _, method, session, _ in requests if method == "tools/list"}
    closed = {session for _, method, session, _ in requests if method == "DELETE"}
    assert len(initialized) == 1
    assert None not in initialized
    assert initialized <= closed
    assert not catalog._TOOL_PRODUCERS
    delivered_tool = manager.get_listed_tool(server, "bound-tool-0")
    assert delivered_tool is not None
    assert delivered_tool.name == "bound-tool-0"
    assert delivered_tool.input_schema == {"type": "object"}
    assert manager.get_listed_tool(server, "bound-tool-1") is None
    assert manager.get_mcp_server_from_tool_name("bound-tool-0") is None


@pytest.mark.asyncio
async def test_manager_reuses_and_expires_complete_tool_discovery_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    class Clock:
        now: float = 0

        def __call__(self) -> float:
            return self.now

    monkeypatch.setenv("LITELLM_SALT_KEY", "tool-discovery-cache-test")
    monkeypatch.setenv("LITELLM_MCP_DISCOVERY_CACHE_TTL", "60")
    monkeypatch.setenv("LITELLM_MCP_TOOLS_LIST_CACHE_TTL", "10")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    transport, calls = _session_bound_tools_transport(tool_count=2, page_size=1, ttl_ms=0)
    server = MCPServer(
        server_id="bound",
        name="bound",
        url="https://bound.example/mcp",
        transport=MCPTransport.http,
    )
    manager = MCPServerManager(upstream_transport=transport)
    clock = Clock()
    monkeypatch.setattr(manager._tool_discovery_cache, "_clock", clock)
    monkeypatch.setattr(manager._tool_discovery_cache._entries, "_clock", clock)
    monkeypatch.setattr(manager, "registry", {server.server_id: server})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)
    context = mcp_operations.prepare_context()

    first = await catalog.aggregate_gateway_tools(
        context,
        PaginatedRequestParams(),
        [server],
        {},
        enforce_rate_limits=False,
        use_tool_list_cache=True,
    )
    first_calls = calls.qsize()
    second = await catalog.aggregate_gateway_tools(
        context,
        PaginatedRequestParams(),
        [server],
        {},
        enforce_rate_limits=False,
        use_tool_list_cache=True,
    )
    assert first.tools == second.tools
    assert calls.qsize() == first_calls
    clock.now = 10
    third = await catalog.aggregate_gateway_tools(
        context,
        PaginatedRequestParams(),
        [server],
        {},
        enforce_rate_limits=False,
        use_tool_list_cache=True,
    )
    assert third.tools == first.tools
    assert calls.qsize() == first_calls * 2


@pytest.mark.asyncio
async def test_streaming_tool_pages_preserve_upstream_page_sizes_and_empty_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp.types import Tool

    monkeypatch.setattr(catalog.time, "time", lambda: 1000.0)
    store = MemoryToolPageStore()
    first_page: Final = asyncio.get_running_loop().create_future()
    publisher: Final = catalog._StreamingToolPagePublisher(
        store, "c" * 64, "bounded", "a" * 32, "bounded", 1600, first_page
    )
    await publisher.start(deadline_at=1300)
    tools: Final = tuple(Tool(name=f"tool-{index}", inputSchema={"type": "object"}) for index in range(205))
    await publisher.publish(ListToolsResult(tools=list(tools), next_cursor="upstream-next"))
    first: Final = first_page.result()
    assert first.tools == list(tools)
    assert first.next_cursor == catalog._tool_page_cursor("a" * 32, 1)

    await publisher.publish(ListToolsResult(tools=[]))
    await publisher.finish(ServerListOk(tool_count=len(tools)), failed=False)
    pages: Final = tuple(
        catalog._CACHED_TOOL_PAGE.validate_python(store.values[key])
        for key in (
            catalog._tool_page_cache_key("c" * 64, "bounded", "a" * 32, 0),
            catalog._tool_page_cache_key("c" * 64, "bounded", "a" * 32, 1),
        )
    )
    assert tuple(len(page.result.tools) for page in pages) == (205, 0)
    assert pages[-1].result.next_cursor is None
    status: Final = catalog._ToolProducerStatus.model_validate(
        store.values[catalog._tool_page_status_key("c" * 64, "bounded", "a" * 32)]
    )
    assert status.state == "complete"


@pytest.mark.asyncio
async def test_streaming_tool_pages_preserve_large_upstream_page_without_splitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp.types import Tool

    monkeypatch.setattr(catalog.time, "time", lambda: 1000.0)
    store = MemoryToolPageStore()
    first_page: Final = asyncio.get_running_loop().create_future()
    publisher: Final = catalog._StreamingToolPagePublisher(
        store, "c" * 64, "bounded", "b" * 32, "bounded", 1600, first_page
    )
    await publisher.start(deadline_at=1300)
    tools: Final = (
        Tool(name="large-a", inputSchema={"type": "object", "description": "a" * 270_000}),
        Tool(name="large-b", inputSchema={"type": "object", "description": "b" * 270_000}),
    ) + tuple(Tool(name=f"small-{index}", inputSchema={}) for index in range(203))
    await publisher.publish(ListToolsResult(tools=[], next_cursor="upstream-next"))
    await publisher.publish(ListToolsResult(tools=list(tools)))
    await publisher.finish(ServerListOk(tool_count=len(tools)), failed=False)
    cursor: Final = first_page.result().next_cursor
    assert cursor is not None
    page: Final = await catalog._read_tool_page(
        cursor,
        server_id="bounded",
        server_key="bounded",
        caller_scope="c" * 64,
        store=store,
        initial_read_timeout=30,
    )
    assert page.tools == list(tools)
    assert page.next_cursor is None
    assert len(page.model_dump_json().encode()) > 512 * 1024


@pytest.mark.asyncio
async def test_streaming_tool_page_publisher_reads_back_an_indivisible_oversized_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp.types import Tool

    monkeypatch.setattr(catalog.time, "time", lambda: 1000.0)
    store = MemoryToolPageStore()
    snapshot_id: Final = "d" * 32
    oversized: Final = Tool(
        name="oversized",
        inputSchema={
            "type": "object",
            "description": "x" * (512 * 1024),
        },
    )
    first_page: Final = asyncio.get_running_loop().create_future()
    publisher: Final = catalog._StreamingToolPagePublisher(
        store, "c" * 64, "bounded", snapshot_id, "bounded", 1600, first_page
    )
    await publisher.start(deadline_at=1300)
    await publisher.publish(ListToolsResult(tools=[oversized]))
    await publisher.finish(ServerListOk(tool_count=1), failed=False)

    cached: Final = catalog._CACHED_TOOL_PAGE.validate_python(
        store.values[catalog._tool_page_cache_key("c" * 64, "bounded", snapshot_id, 0)]
    )
    assert first_page.result().tools == [oversized]
    assert cached.result.tools == [oversized]
    assert len(cached.model_dump_json().encode()) > 512 * 1024


def _session_bound_tools_transport(
    *,
    failing_servers: frozenset[str] = frozenset(),
    auth_failing_servers: frozenset[str] = frozenset(),
    tool_count: int = 2,
    page_size: int = 1,
    ttl_ms: int = 9000,
    tool_description: str | None = None,
    revision: str | None = None,
) -> tuple[httpx2.MockTransport, queue.SimpleQueue[tuple[str, str, str | None, str | None]]]:
    session_ids: Final = count()
    calls: Final[queue.SimpleQueue[tuple[str, str, str | None, str | None]]] = queue.SimpleQueue()
    payload_adapter: Final = TypeAdapter(dict[str, JsonValue])

    def cursor_offset(session_id: str | None, cursor: str | None) -> int | None:
        if cursor is None:
            return 0
        if session_id is None:
            return None
        cursor_session, separator, raw_offset = cursor.rpartition(":")
        if not separator or cursor_session != session_id or not raw_offset.isdecimal():
            return None
        return int(raw_offset)

    def respond(request: httpx2.Request) -> httpx2.Response:
        server_id: Final = request.url.host.partition(".")[0]
        session_id: Final = request.headers.get("mcp-session-id")
        if request.method != "POST":
            calls.put((server_id, request.method, session_id, None))
            return httpx2.Response(200)

        payload: Final = payload_adapter.validate_json(request.content)
        method: Final = payload.get("method")
        params_value: Final = payload.get("params")
        params: Final = params_value if isinstance(params_value, dict) else {}
        cursor_value: Final = params.get("cursor")
        cursor: Final = cursor_value if isinstance(cursor_value, str) else None
        calls.put((server_id, method if isinstance(method, str) else "", session_id, cursor))
        request_id: Final = payload.get("id")

        if method == "initialize":
            init_params: Final = params
            protocol_version: Final = init_params.get("protocolVersion")
            session: Final = f"{server_id}-session-{next(session_ids)}"
            return httpx2.Response(
                200,
                headers={"content-type": "application/json", "mcp-session-id": session},
                json={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": protocol_version,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": server_id, "version": "1"},
                    },
                },
            )
        if method == "notifications/initialized":
            return httpx2.Response(202)
        if method != "tools/list":
            return httpx2.Response(405)
        if session_id is None:
            return httpx2.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32602, "message": "missing initialized session"},
                },
            )
        offset: Final = cursor_offset(session_id, cursor)
        if offset is None:
            return httpx2.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32602, "message": "cursor session mismatch"},
                },
            )
        if server_id in failing_servers and offset > 0:
            return httpx2.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": "untrusted upstream message"},
                },
            )
        if server_id in auth_failing_servers and offset > 0:
            return httpx2.Response(
                401,
                headers={"www-authenticate": 'Bearer error="invalid_token"'},
                json={"error": "unauthorized"},
            )

        end: Final = min(offset + page_size, tool_count)
        return httpx2.Response(
            200,
            headers={"content-type": "application/json", "mcp-session-id": session_id},
            json={
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": f"{server_id}-tool-{index}",
                            "inputSchema": {"type": "object"},
                            **({"description": tool_description} if tool_description is not None else {}),
                        }
                        for index in range(offset, end)
                    ],
                    "nextCursor": f"{session_id}:{end}" if end < tool_count else None,
                    **({"_meta": {"revision": revision}} if revision is not None else {}),
                    "ttlMs": ttl_ms,
                    "cacheScope": "public",
                },
            },
        )

    return httpx2.MockTransport(respond), calls


@pytest.mark.asyncio
async def test_failed_peer_continuation_preserves_healthy_peer_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    from mcp.types import PaginatedRequestParams

    from litellm.proxy import proxy_server
    from litellm.proxy.mcp_server import operations as mcp_operations

    monkeypatch.setenv("LITELLM_SALT_KEY", "peer-continuation-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", object())
    monkeypatch.setattr(catalog, "_TOOL_PAGE_STORE", MemoryToolPageStore())
    transport, calls = _session_bound_tools_transport(
        failing_servers=frozenset({"broken"}), tool_count=2, page_size=1, ttl_ms=0, revision="fixture-v1"
    )
    servers: Final = tuple(
        MCPServer(
            server_id=server_id,
            name=server_id,
            url=f"https://{server_id}.example/mcp",
            transport=MCPTransport.http,
        )
        for server_id in ("broken", "healthy")
    )
    manager: Final = MCPServerManager(upstream_transport=transport)
    monkeypatch.setattr(manager, "registry", {server.server_id: server for server in servers})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)
    context: Final = mcp_operations.prepare_context()

    initial: Final = await catalog.aggregate_gateway_tools(
        context,
        PaginatedRequestParams(),
        servers,
        {},
        record_listing=True,
        enforce_rate_limits=False,
        use_tool_list_cache=False,
    )
    assert initial.next_cursor is not None
    await asyncio.gather(*tuple(catalog._TOOL_PRODUCERS.values()))

    continuation: Final = await catalog.aggregate_gateway_tools(
        context,
        PaginatedRequestParams(cursor=initial.next_cursor),
        servers,
        {},
        record_listing=True,
        enforce_rate_limits=False,
        use_tool_list_cache=False,
    )
    assert tuple(tool.name for tool in continuation.tools) == ("healthy-healthy-tool-1",)
    assert continuation.next_cursor is None
    assert continuation.outcomes["broken"].tag != "ok"
    assert continuation.outcomes["healthy"].tag == "ok"
    assert manager.get_listed_tool(servers[0], "broken-tool-0") is not None
    assert manager.get_listed_tool(servers[0], "broken-tool-1") is None

    requests: Final = tuple(calls.get_nowait() for _ in range(calls.qsize()))
    tool_requests: Final = tuple(request for request in requests if request[1] == "tools/list")
    assert len(tool_requests) == 4
    for server_id in ("broken", "healthy"):
        session_ids: Final = {request[2] for request in tool_requests if request[0] == server_id}
        assert len(session_ids) == 1
        assert None not in session_ids
    assert calls.empty()


@pytest.mark.asyncio
async def test_tool_page_producer_outlives_request_without_inheriting_catalog_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.mcp_server import operations as mcp_operations

    monkeypatch.setenv("LITELLM_SALT_KEY", "producer-context-isolation-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", object())
    page_started: Final = asyncio.Event()
    release_page: Final = asyncio.Event()
    producer_snapshot: Final[asyncio.Future[catalog.CatalogSnapshot | None]] = (
        asyncio.get_running_loop().create_future()
    )

    class BlockingPageStore(MemoryToolPageStore):
        async def set(self, key: str, value: JsonValue, ttl: float) -> None:
            if ":list-page:" in key and key.endswith(":1"):
                if not producer_snapshot.done():
                    producer_snapshot.set_result(manager.catalog.current())
                page_started.set()
                await release_page.wait()
            await super().set(key, value, ttl)

    store: Final = BlockingPageStore()
    monkeypatch.setattr(catalog, "_TOOL_PAGE_STORE", store)
    transport, calls = _session_bound_tools_transport(tool_count=2, page_size=1, ttl_ms=0)
    server: Final = MCPServer(
        server_id="bound",
        name="bound",
        url="https://bound.example/mcp",
        transport=MCPTransport.http,
    )
    manager: Final = MCPServerManager(upstream_transport=transport)
    monkeypatch.setattr(manager, "registry", {server.server_id: server})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)
    context: Final = mcp_operations.prepare_context(UserAPIKeyAuth(api_key="producer-context-caller"))
    page_started_waiter: Final = asyncio.create_task(page_started.wait())

    try:
        async with manager.catalog.operation() as request_snapshot:
            active_lease: Final = manager.catalog._operation.get()
            assert active_lease is not None
            request_lease: Final = active_lease[1]
            first: Final = await catalog.aggregate_gateway_tools(
                context,
                PaginatedRequestParams(),
                (server,),
                {},
                enforce_rate_limits=False,
                use_tool_list_cache=False,
            )
            assert first.next_cursor is not None
            producers: Final = tuple(catalog._TOOL_PRODUCERS.values())
            assert len(producers) == 1
            producer: Final = producers[0]
            await asyncio.wait((page_started_waiter, producer), return_when=asyncio.FIRST_COMPLETED)
            assert page_started.is_set()

        assert request_lease.is_set()
        assert not producer.done()
        assert producer_snapshot.result() is not None
        assert producer_snapshot.result() is not request_snapshot
        release_page.set()
        await producer

        continuation: Final = await catalog.aggregate_gateway_tools(
            context,
            PaginatedRequestParams(cursor=first.next_cursor),
            (server,),
            {},
            enforce_rate_limits=False,
            use_tool_list_cache=False,
        )
        assert tuple(tool.name for tool in first.tools + continuation.tools) == (
            "bound-bound-tool-0",
            "bound-bound-tool-1",
        )
        assert continuation.next_cursor is None
        requests: Final = tuple(calls.get_nowait() for _ in range(calls.qsize()))
        assert sum(request[1] == "tools/list" for request in requests) == 2
    finally:
        release_page.set()
        page_started_waiter.cancel()
        await asyncio.gather(page_started_waiter, return_exceptions=True)
        await catalog.shutdown_tool_page_producers()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server_ids", "expected_tools"),
    (
        pytest.param(("broken", "healthy"), ("healthy-healthy-tool-1",), id="healthy-peer"),
        pytest.param(("broken",), None, id="all-peers-unavailable"),
    ),
)
async def test_missing_cached_page_fails_only_its_server_and_never_refetches(
    monkeypatch: pytest.MonkeyPatch,
    server_ids: tuple[str, ...],
    expected_tools: tuple[str, ...] | None,
) -> None:
    import hashlib

    from litellm.proxy import proxy_server
    from litellm.proxy.mcp_server import operations as mcp_operations

    monkeypatch.setenv("LITELLM_SALT_KEY", "missing-cached-page-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", object())
    store: Final = MemoryToolPageStore()
    monkeypatch.setattr(catalog, "_TOOL_PAGE_STORE", store)
    transport, calls = _session_bound_tools_transport(tool_count=2, page_size=1, ttl_ms=0, revision="fixture-v1")
    servers: Final = tuple(
        MCPServer(
            server_id=server_id,
            name=server_id,
            url=f"https://{server_id}.example/mcp",
            transport=MCPTransport.http,
        )
        for server_id in server_ids
    )
    manager: Final = MCPServerManager(upstream_transport=transport)
    monkeypatch.setattr(manager, "registry", {server.server_id: server for server in servers})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)
    context: Final = mcp_operations.prepare_context()

    initial: Final = await catalog.aggregate_gateway_tools(
        context,
        PaginatedRequestParams(),
        servers,
        {},
        enforce_rate_limits=False,
        use_tool_list_cache=False,
    )
    assert initial.next_cursor is not None
    await asyncio.gather(*tuple(catalog._TOOL_PRODUCERS.values()))

    broken_hash: Final = hashlib.sha256(b"broken").hexdigest()
    broken_page_key: Final = next(
        key
        for key in store.values
        if ":list-page:" in key and f":{broken_hash}:" in key and key.endswith(":1")
    )
    snapshot_id: Final = broken_page_key.rsplit(":", 2)[1]
    broken_status_key: Final = next(
        key
        for key in store.values
        if ":list-status:" in key and f":{broken_hash}:" in key and key.endswith(snapshot_id)
    )
    status: Final = catalog._ToolProducerStatus.model_validate(store.values[broken_status_key])
    assert status.meta is not None
    assert status.meta.get("revision") == "fixture-v1"
    assert store.values.pop(broken_page_key) is not None

    continuation: ListToolsResult | None = None
    if expected_tools is None:
        with pytest.raises(MCPError, match="start a fresh listing"):
            continuation = await catalog.aggregate_gateway_tools(
                context,
                PaginatedRequestParams(cursor=initial.next_cursor),
                servers,
                {},
                enforce_rate_limits=False,
                use_tool_list_cache=False,
            )
    else:
        continuation = await catalog.aggregate_gateway_tools(
            context,
            PaginatedRequestParams(cursor=initial.next_cursor),
            servers,
            {},
            enforce_rate_limits=False,
            use_tool_list_cache=False,
        )

    assert (continuation is None) == (expected_tools is None)
    assert (
        continuation is None
        or (
            tuple(tool.name for tool in continuation.tools) == expected_tools
            and continuation.next_cursor is None
            and continuation.outcomes["broken"].tag != "ok"
            and ("healthy" not in server_ids or continuation.outcomes["healthy"].tag == "ok")
        )
    )
    requests: Final = tuple(calls.get_nowait() for _ in range(calls.qsize()))
    tool_requests: Final = tuple(request for request in requests if request[1] == "tools/list")
    assert len(tool_requests) == 2 * len(server_ids)
    assert calls.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("display_name", (None, "friendly-tool"))
@pytest.mark.parametrize("mcp_proxy_mode", (False, True))
async def test_cached_continuation_records_tools_on_the_receiving_worker(
    monkeypatch: pytest.MonkeyPatch, display_name: str | None, mcp_proxy_mode: bool
) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.mcp_server import operations as mcp_operations
    from litellm.proxy.mcp_server.mcp_server_manager import ListedToolsCaller

    monkeypatch.setenv("LITELLM_SALT_KEY", "receiving-worker-cached-page-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", object())
    store: Final = MemoryToolPageStore()
    monkeypatch.setattr(catalog, "_TOOL_PAGE_STORE", store)
    transport, calls = _session_bound_tools_transport(tool_count=2, page_size=1, ttl_ms=0, revision="fixture-v1")
    server: Final = MCPServer(
        server_id="bound",
        name="bound",
        url="https://bound.example/mcp",
        transport=MCPTransport.http,
    )
    server.tool_name_to_display_name = {"bound-tool-1": display_name} if display_name is not None else None
    first_manager: Final = MCPServerManager(upstream_transport=transport)
    monkeypatch.setattr(first_manager, "registry", {server.server_id: server})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", first_manager)
    auth: Final = UserAPIKeyAuth(api_key="cached-page-owner", user_id="cached-page-user")
    context: Final = mcp_operations.prepare_context(auth, mcp_proxy_mode=mcp_proxy_mode)

    first: Final = await catalog.aggregate_gateway_tools(
        context,
        PaginatedRequestParams(),
        (server,),
        {},
        record_listing=True,
        enforce_rate_limits=False,
        use_tool_list_cache=False,
    )
    assert first.next_cursor is not None
    await asyncio.gather(*tuple(catalog._TOOL_PRODUCERS.values()))
    initial_requests: Final = tuple(calls.get_nowait() for _ in range(calls.qsize()))
    assert sum(request[1] == "tools/list" for request in initial_requests) == 2

    receiving_manager: Final = MCPServerManager(upstream_transport=transport)
    monkeypatch.setattr(receiving_manager, "registry", {server.server_id: server})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", receiving_manager)
    async with receiving_manager.catalog.operation():
        _, prepared_headers = await catalog._tool_listing_scope(context, (server,), {})
    listed_caller: Final = ListedToolsCaller(
        user_api_key_auth=auth,
        mcp_auth_header=prepared_headers[server.server_id][2],
        raw_headers=context.raw_headers,
        oauth2_headers=context.oauth2_headers,
    )
    other_auth: Final = UserAPIKeyAuth(api_key="different-cached-page-owner", user_id="different-user")
    other_context: Final = mcp_operations.prepare_context(other_auth)
    other_caller: Final = ListedToolsCaller(
        user_api_key_auth=other_auth,
        mcp_auth_header=prepared_headers[server.server_id][2],
        raw_headers=other_context.raw_headers,
        oauth2_headers=other_context.oauth2_headers,
    )
    assert receiving_manager.get_listed_tool(server, "bound-tool-1", listed_caller) is None

    continuation: Final = await catalog.aggregate_gateway_tools(
        context,
        PaginatedRequestParams(cursor=first.next_cursor),
        (server,),
        {},
        record_listing=True,
        enforce_rate_limits=False,
        use_tool_list_cache=False,
    )

    listed_tool_name: Final = "bound-tool-1"
    listed_tool: Final = receiving_manager.get_listed_tool(server, listed_tool_name, listed_caller)
    assert listed_tool is not None
    assert listed_tool.name == listed_tool_name
    assert listed_tool.input_schema == {"type": "object"}
    assert "litellm.ai/proxy_tool_identity" not in (listed_tool.meta or {})
    assert receiving_manager.get_listed_tool(server, listed_tool_name, other_caller) is None
    expected_name: Final = (
        "bound-bound-tool-1" if mcp_proxy_mode or display_name is None else display_name
    )
    assert continuation.tools[0].name == expected_name
    assert "bound-bound-tool-1" in receiving_manager.tool_name_to_mcp_server_name_mapping
    assert "bound-tool-1" not in receiving_manager.tool_name_to_mcp_server_name_mapping
    assert calls.empty()


@pytest.mark.asyncio
async def test_gateway_returns_catalog_larger_than_sixteen_mib_through_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp.types import PaginatedRequestParams

    from litellm.proxy import proxy_server

    monkeypatch.setenv("LITELLM_SALT_KEY", "large-tool-catalog-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", object())
    monkeypatch.setattr(catalog, "_TOOL_PAGE_STORE", MemoryToolPageStore())
    tool_count: Final = 17
    transport, calls = _session_bound_tools_transport(
        tool_count=tool_count, page_size=1, tool_description="x" * (1024 * 1024)
    )
    server: Final = MCPServer(
        server_id="bound",
        name="bound",
        url="https://bound.example/mcp",
        transport=MCPTransport.http,
    )
    manager: Final = MCPServerManager(upstream_transport=transport)
    monkeypatch.setattr(manager, "registry", {server.server_id: server})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)
    context: Final = mcp_operations.prepare_context(UserAPIKeyAuth(api_key="large-catalog-caller"))

    first: Final = await catalog.aggregate_gateway_tools(
        context,
        PaginatedRequestParams(),
        [server],
        {},
        enforce_rate_limits=False,
        use_tool_list_cache=False,
    )
    assert first.next_cursor is not None
    producers: Final = tuple(catalog._TOOL_PRODUCERS.values())
    assert len(producers) == 1
    await asyncio.gather(*producers)

    async def read_remaining(cursor: str | None) -> tuple[AggregateToolListing, ...]:
        if cursor is None:
            return ()
        page: Final = await catalog.aggregate_gateway_tools(
            context,
            PaginatedRequestParams(cursor=cursor),
            [server],
            {},
            enforce_rate_limits=False,
            use_tool_list_cache=False,
        )
        return (page, *await read_remaining(page.next_cursor))

    pages: Final = (first, *await read_remaining(first.next_cursor))
    tools: Final = tuple(tool for page in pages for tool in page.tools)
    tool_requests: Final = tuple(
        request for request in (calls.get_nowait() for _ in range(calls.qsize())) if request[1] == "tools/list"
    )
    assert len(pages) == tool_count
    assert len(tools) == tool_count
    assert sum(len(tool.model_dump_json().encode()) for tool in tools) > 16 * 1024 * 1024
    assert all(page.next_cursor is not None for page in pages[:-1])
    assert pages[-1].next_cursor is None
    assert len(tool_requests) == tool_count
    session_id: Final = tool_requests[0][2]
    assert session_id is not None
    assert {request[2] for request in tool_requests} == {session_id}
    assert tuple(request[3] for request in tool_requests) == (
        None,
        *(f"{session_id}:{index}" for index in range(1, tool_count)),
    )


@pytest.mark.asyncio
async def test_gateway_falls_back_to_complete_listing_when_tool_page_producer_quota_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp.types import PaginatedRequestParams

    from litellm.proxy import proxy_server

    monkeypatch.setenv("LITELLM_SALT_KEY", "tool-page-quota-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", object())
    monkeypatch.setattr(catalog, "_TOOL_PAGE_STORE", MemoryToolPageStore())
    transport, calls = _session_bound_tools_transport(tool_count=3, page_size=1)
    server: Final = MCPServer(
        server_id="bound",
        name="bound",
        url="https://bound.example/mcp",
        transport=MCPTransport.http,
    )
    manager: Final = MCPServerManager(upstream_transport=transport)
    monkeypatch.setattr(manager, "registry", {server.server_id: server})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)
    context: Final = mcp_operations.prepare_context(UserAPIKeyAuth(api_key="quota-caller"))
    release: Final = asyncio.Event()

    async def reserve_slot() -> None:
        await release.wait()

    reserved: Final = {
        ("reserved", str(index), "snapshot"): asyncio.create_task(reserve_slot())
        for index in range(catalog._TOOL_PAGE_PRODUCER_MAX)
    }
    monkeypatch.setattr(catalog, "_TOOL_PRODUCERS", reserved)
    try:
        result: Final = await catalog.aggregate_gateway_tools(
            context,
            PaginatedRequestParams(),
            [server],
            {},
            enforce_rate_limits=False,
            use_tool_list_cache=False,
        )
        assert tuple(tool.name for tool in result.tools) == tuple(f"bound-bound-tool-{index}" for index in range(3))
        assert result.next_cursor is None
        assert result.outcomes["bound"].tag == "ok"
        assert tuple(catalog._TOOL_PRODUCERS) == tuple(reserved)
        requests: Final = tuple(calls.get_nowait() for _ in range(calls.qsize()))
        tool_requests: Final = tuple(request for request in requests if request[1] == "tools/list")
        assert len(tool_requests) == 3
        assert len({request[2] for request in tool_requests}) == 1
    finally:
        release.set()
        await asyncio.gather(*reserved.values())


@pytest.mark.asyncio
async def test_gateway_storage_fallback_still_challenges_later_upstream_auth_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp.types import PaginatedRequestParams

    from litellm.proxy import proxy_server

    class FailedStartStore(MemoryToolPageStore):
        async def set(self, key: str, value: JsonValue, ttl: float) -> None:
            if ":list-status:" in key:
                raise OSError("cache unavailable")
            await super().set(key, value, ttl)

    monkeypatch.setenv("LITELLM_SALT_KEY", "tool-page-fallback-auth-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", object())
    monkeypatch.setattr(catalog, "_TOOL_PAGE_STORE", FailedStartStore())
    transport, calls = _session_bound_tools_transport(
        auth_failing_servers=frozenset(("bound",)), tool_count=3, page_size=1
    )
    server: Final = MCPServer(
        server_id="bound",
        name="docs",
        alias="docs",
        url="https://bound.example/mcp",
        transport=MCPTransport.http,
        auth_type=MCPAuth.true_passthrough,
    )
    manager: Final = MCPServerManager(upstream_transport=transport)
    monkeypatch.setattr(manager, "registry", {server.server_id: server})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)
    gate: Final = OAuthScopeResponse(AsyncMock(), _GATEWAY, "/docs/mcp", reauthorize_as="docs")
    context: Final = mcp_operations.prepare_context(
        UserAPIKeyAuth(token="h", user_id="u1"),
        oauth2_headers={"Authorization": "Bearer rejected"},
    )
    request_token: Final = active_mcp_request_ctx_var.set(_request_context(gate, "tools/list"))
    name_token: Final = mcp_gateway_server_name.set("docs")
    try:
        result: Final = await catalog.aggregate_gateway_tools(
            context,
            PaginatedRequestParams(),
            [server],
            {},
            enforce_rate_limits=False,
            use_tool_list_cache=False,
        )
    finally:
        mcp_gateway_server_name.reset(name_token)
        active_mcp_request_ctx_var.reset(request_token)

    assert result.tools == []
    assert result.next_cursor is None
    assert result.outcomes["docs"].tag == "auth_required"
    assert not gate.pending
    requests: Final = tuple(calls.get_nowait() for _ in range(calls.qsize()))
    tool_requests: Final = tuple(request for request in requests if request[1] == "tools/list")
    assert len(tool_requests) == 2
    assert tool_requests[0][2] == tool_requests[1][2]
    assert tool_requests[0][3] is None
    assert tool_requests[1][3] == f"{tool_requests[0][2]}:1"


@pytest.mark.asyncio
async def test_streamed_listing_records_guarded_authorized_underlying_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler
    from mcp.types import Tool

    monkeypatch.setenv("LITELLM_SALT_KEY", "guarded-streamed-tools-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", object())
    monkeypatch.setattr(catalog, "_TOOL_PAGE_STORE", MemoryToolPageStore())
    transport, _calls = _session_bound_tools_transport(tool_count=3, page_size=3)
    server = MCPServer(
        server_id="bound",
        name="bound",
        url="https://bound.example/mcp",
        transport=MCPTransport.http,
        tool_name_to_display_name={"bound-tool-0": "renamed"},
    )
    manager = MCPServerManager(upstream_transport=transport)
    monkeypatch.setattr(manager, "registry", {server.server_id: server})
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)
    guarded_pages: list[tuple[str, ...]] = []

    async def guard_catalog(
        *,
        server: MCPServer,
        tools: Sequence[Tool],
        **_kwargs: object,
    ) -> tuple[Tool, ...]:
        assert server.server_id == "bound"
        guarded_pages.append(tuple(tool.name for tool in tools))
        return tuple(
            tool.model_copy(update={"description": "guarded description"})
            for tool in tools
            if tool.name != "bound-tool-1"
        )

    async def granted_tools(server_id: str, user_api_key_auth: object | None) -> set[str]:
        assert server_id == "bound"
        return {"bound-tool-0"}

    monkeypatch.setattr(manager, "_guard_tool_catalog", guard_catalog)
    monkeypatch.setattr(MCPRequestHandler, "get_allowed_tools_for_server", granted_tools)
    result: Final = await catalog.aggregate_gateway_tools(
        mcp_operations.prepare_context(),
        PaginatedRequestParams(),
        [server],
        {},
        record_listing=True,
        enforce_rate_limits=False,
        use_tool_list_cache=False,
    )
    await asyncio.gather(*tuple(catalog._TOOL_PRODUCERS.values()))

    assert tuple(tool.name for tool in result.tools) == ("renamed",)
    assert result.tools[0].description == "guarded description"
    assert guarded_pages == [("bound-tool-0", "bound-tool-1", "bound-tool-2")]
    listed: Final = manager.get_listed_tool(server, "bound-tool-0")
    assert listed is not None
    assert listed.description == "guarded description"
    assert manager.get_listed_tool(server, "bound-tool-1") is None
    assert manager.get_listed_tool(server, "bound-tool-2") is None
    assert manager.get_mcp_server_from_tool_name("bound-tool-1") is None
    assert manager.get_mcp_server_from_tool_name("bound-tool-2") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_name,result_name,field",
    [
        ("ListPromptsRequest", "ListPromptsResult", "prompts"),
        ("ListResourcesRequest", "ListResourcesResult", "resources"),
        ("ListResourceTemplatesRequest", "ListResourceTemplatesResult", "resource_templates"),
    ],
)
async def test_optional_gateway_catalog_reports_initial_failure_and_rejects_failed_continuation(
    monkeypatch, request_name, result_name, field
):
    from unittest.mock import AsyncMock

    from mcp import types

    from litellm.proxy import proxy_server
    from litellm.proxy.mcp_server import operations
    from litellm.proxy.mcp_server.faults.list_outcomes import SERVER_OUTCOMES_META_KEY
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    monkeypatch.setenv("LITELLM_SALT_KEY", "catalog-failure-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    server = MCPServer(server_id="pages", name="pages", transport=MCPTransport.http)
    monkeypatch.setattr(operations.global_mcp_server_manager, "registry", {server.server_id: server})
    monkeypatch.setattr(operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[server]))
    fetch = AsyncMock(side_effect=TimeoutError("upstream secret"))
    monkeypatch.setattr(catalog, "fetch_optional_catalog_page", fetch)
    context = operations.prepare_context()
    request = getattr(types, request_name)
    failed = await catalog.list_gateway_catalog(context, request())
    assert getattr(failed, field) == [] and failed.next_cursor is None
    assert next(iter(failed.meta[SERVER_OUTCOMES_META_KEY].values()))["status"] == "timeout"
    assert "upstream secret" not in failed.model_dump_json()
    fetch.side_effect = [
        getattr(types, result_name)(**{field: [], "next_cursor": "next"}),
        TimeoutError("upstream secret"),
    ]
    first = await catalog.list_gateway_catalog(context, request())
    assert first.next_cursor
    with pytest.raises(MCPError, match="Upstream continuation failed; start a fresh listing") as denied:
        await catalog.list_gateway_catalog(
            context, request(params=types.PaginatedRequestParams(cursor=first.next_cursor))
        )
    assert "upstream secret" not in str(denied.value)
    assert fetch.await_count == 3
    assert fetch.await_args.args[-1] == "next"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tools", "prompts", "resources", "templates"])
async def test_first_catalog_page_keeps_admitted_items_and_reports_rate_limited_servers(
    monkeypatch: pytest.MonkeyPatch, kind: CatalogKind
) -> None:
    from mcp import types

    servers, context, limiter = rate_limit_catalog_setup(monkeypatch, frozenset({"catalog-a"}))

    async def fetch_tools(server: MCPServer, **_kwargs: object) -> tuple[ListToolsResult, ServerListOk]:
        return ListToolsResult(
            tools=[types.Tool(name=f"{server.server_id}-item", inputSchema={"type": "object"})]
        ), ServerListOk(tool_count=1)

    async def fetch_optional(
        _context: OperationContext,
        _request: ListPromptsRequest | ListResourcesRequest | ListResourceTemplatesRequest,
        server: MCPServer,
        _allowed: Sequence[MCPServer],
        _cursor: str | None,
    ) -> OptionalCatalogResult:
        return optional_catalog_page(kind, server.server_id)

    if kind == "tools":
        monkeypatch.setattr(catalog, "get_filtered_server_tools", fetch_tools)
    else:
        monkeypatch.setattr(catalog, "fetch_optional_catalog_page", fetch_optional)

    result: Final = await run_catalog_listing(kind, context, servers)
    if isinstance(result, AggregateToolListing):
        assert [tool.name for tool in result.tools] == ["catalog-b-item"]
        assert result.outcomes["catalog-a"].tag == "rate_limited"
    else:
        field: Final = {
            "prompts": "prompts",
            "resources": "resources",
            "templates": "resource_templates",
        }[kind]
        assert [item.name for item in getattr(result, field)] == ["catalog-b-item"]
        assert result.meta[SERVER_OUTCOMES_META_KEY]["catalog-a"]["status"] == "rate_limited"
    assert {call.args[1].server_id for call in limiter.await_args_list} == {"catalog-a", "catalog-b"}
    assert len(limiter.await_args_list) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tools", "prompts", "resources", "templates"])
async def test_first_catalog_page_raises_when_every_server_is_rate_limited(
    monkeypatch: pytest.MonkeyPatch, kind: CatalogKind
) -> None:
    servers, context, limiter = rate_limit_catalog_setup(monkeypatch, frozenset({"catalog-a", "catalog-b"}))
    fetch_tools: Final = AsyncMock(return_value=(ListToolsResult(tools=[]), ServerListOk(tool_count=0)))
    fetch_optional: Final = AsyncMock(return_value=optional_catalog_page(kind, "catalog-a"))
    if kind == "tools":
        monkeypatch.setattr(catalog, "get_filtered_server_tools", fetch_tools)
    else:
        monkeypatch.setattr(catalog, "fetch_optional_catalog_page", fetch_optional)

    with pytest.raises(ProxyRateLimitError, match="RPM exceeded"):
        await run_catalog_listing(kind, context, servers)

    if kind == "tools":
        fetch_tools.assert_not_awaited()
    else:
        fetch_optional.assert_not_awaited()
    assert len(limiter.await_args_list) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tools", "prompts", "resources", "templates"])
async def test_continuation_rate_limit_raises_without_refetching_completed_servers(
    monkeypatch: pytest.MonkeyPatch, kind: CatalogKind
) -> None:
    from litellm.proxy import proxy_server
    from mcp import types

    servers, context, limiter = rate_limit_catalog_setup(monkeypatch, frozenset())

    async def fetch_tools(
        server: MCPServer,
        *,
        on_page: Callable[[ListToolsResult], Awaitable[None]] | None = None,
        **_kwargs: object,
    ) -> tuple[ListToolsResult, ServerListOk]:
        tool_count: Final = 205 if server.server_id == "catalog-a" else 1
        result: Final = ListToolsResult(
            tools=[
                types.Tool(name=f"{server.server_id}-item-{index}", inputSchema={"type": "object"})
                for index in range(tool_count)
            ],
        )
        if on_page is not None:
            if server.server_id == "catalog-a":
                page_boundary: Final = len(result.tools) // 2
                await on_page(
                    result.model_copy(update={"tools": result.tools[:page_boundary], "next_cursor": "upstream-next"})
                )
                await on_page(result.model_copy(update={"tools": result.tools[page_boundary:]}))
            else:
                await on_page(result)
        return result, ServerListOk(tool_count=tool_count)

    async def fetch_optional(
        _context: OperationContext,
        _request: ListPromptsRequest | ListResourcesRequest | ListResourceTemplatesRequest,
        server: MCPServer,
        _allowed: Sequence[MCPServer],
        cursor: str | None,
    ) -> OptionalCatalogResult:
        return optional_catalog_page(
            kind,
            server.server_id,
            "next" if server.server_id == "catalog-a" and cursor is None else None,
        )

    if kind == "tools":
        monkeypatch.setattr(proxy_server, "redis_usage_cache", object())
        monkeypatch.setattr(catalog, "_TOOL_PAGE_STORE", MemoryToolPageStore())
        monkeypatch.setattr(catalog, "get_filtered_server_tools", fetch_tools)
    else:
        monkeypatch.setattr(catalog, "fetch_optional_catalog_page", fetch_optional)

    first_page: Final = await run_catalog_listing(kind, context, servers)
    if kind == "tools":
        assert isinstance(first_page, AggregateToolListing)
        assert first_page.next_cursor
        await asyncio.gather(*tuple(catalog._TOOL_PRODUCERS.values()))
        limiter.side_effect = ProxyRateLimitError(detail="catalog-a RPM exceeded")
        with pytest.raises(ProxyRateLimitError, match="catalog-a RPM exceeded"):
            await run_catalog_listing(kind, context, servers, first_page.next_cursor)
        charged_server_ids: Final = [call.args[1].server_id for call in limiter.await_args_list]
        assert charged_server_ids.count("catalog-a") == 2
        assert charged_server_ids.count("catalog-b") == 1
        return

    assert first_page.next_cursor is not None
    limiter.side_effect = ProxyRateLimitError(detail="catalog-a RPM exceeded")
    with pytest.raises(ProxyRateLimitError, match="catalog-a RPM exceeded"):
        await run_catalog_listing(kind, context, servers, first_page.next_cursor)

    charged_server_ids: Final = [call.args[1].server_id for call in limiter.await_args_list]
    assert charged_server_ids.count("catalog-a") == 2
    assert charged_server_ids.count("catalog-b") == 1


@pytest.mark.parametrize("kind", ("prompts", "resources", "templates"))
@pytest.mark.parametrize("ttls,expected", (((9000, 4000), 4000), ((9000, 0), 0)))
def test_optional_catalog_preserves_conservative_freshness(kind: str, ttls: tuple[int, int], expected: int) -> None:
    from mcp.types import (
        ListPromptsRequest,
        ListPromptsResult,
        ListResourcesRequest,
        ListResourcesResult,
        ListResourceTemplatesRequest,
        ListResourceTemplatesResult,
    )

    request, result_type, field = {
        "prompts": (ListPromptsRequest(), ListPromptsResult, "prompts"),
        "resources": (ListResourcesRequest(), ListResourcesResult, "resources"),
        "templates": (ListResourceTemplatesRequest(), ListResourceTemplatesResult, "resource_templates"),
    }[kind]
    pages: Final = tuple(result_type(**{field: []}, ttl_ms=ttl, cache_scope="public") for ttl in ttls)
    result: Final = catalog.combine_optional_catalog(request, pages, None, None)
    assert result.ttl_ms == expected
    assert result.cache_scope == "private"


@pytest.mark.asyncio
async def test_tool_catalog_preserves_upstream_freshness() -> None:
    async def fetch(server_id: str, cursor: str | None) -> ListToolsResult:
        return page(server_id).model_copy(update={"ttl_ms": 9000, "cache_scope": "public"})

    result: Final = await listing(fetch)
    assert 0 < result.ttl_ms <= 9000
    assert result.cache_scope == "private"


@pytest.mark.asyncio
@pytest.mark.parametrize("upstream_ttl,limit,expected_ttl", [(1000, 60, 1000), (90000, 1, 1000), (0, 60, 0)])
async def test_discovery_cache_uses_upstream_freshness_and_configured_cap(
    upstream_ttl: int, limit: float, expected_ttl: int
) -> None:
    from unittest.mock import AsyncMock

    from mcp.types import ListPromptsResult, Prompt
    from pydantic import TypeAdapter

    class Clock:
        now: float = 0.0

        def __call__(self) -> float:
            return self.now

    clock: Final = Clock()
    cache: Final = catalog._DiscoveryCache(limit, clock, TypeAdapter(ListPromptsResult))
    fetch: Final = AsyncMock(return_value=ListPromptsResult(prompts=[Prompt(name="fresh")], ttl_ms=upstream_ttl))
    first: Final = await cache.get(("server", "caller"), fetch)
    assert first.prompts[0].name == "fresh"
    clock.now = 0.5  # rebind-ok: advance the injected test clock without sleeping
    second: Final = await cache.get(("server", "caller"), fetch)
    assert second.prompts[0].name == "fresh"
    if expected_ttl:
        assert fetch.await_count == 1
        assert second.ttl_ms == 500
        second.prompts[0].name = "caller edit"  # rebind-ok: prove caller mutation cannot alter retained results
    else:
        assert fetch.await_count == 2
    clock.now = 1.0  # rebind-ok: reach the exact expiry boundary without sleeping
    assert (await cache.get(("server", "caller"), fetch)).prompts[0].name == "fresh"
    assert fetch.await_count == (2 if expected_ttl else 3)


def test_partial_optional_catalog_never_advertises_freshness() -> None:
    from mcp.types import ListPromptsRequest, ListPromptsResult

    from litellm.proxy.mcp_server.faults.list_outcomes import SERVER_OUTCOMES_META_KEY

    result: Final = catalog.combine_optional_catalog(
        ListPromptsRequest(),
        [ListPromptsResult(prompts=[], ttl_ms=9000)],
        None,
        {SERVER_OUTCOMES_META_KEY: {"failed-earlier": {"tag": "timeout"}}},
    )
    assert result.ttl_ms == 0
    assert result.cache_scope == "private"


_GATEWAY: Final = "https://gateway.example"
_UNAUTHORIZED: Final = ServerListFault(tag="auth_required", status_code=401)


def _server(
    auth_type: MCPAuth,
    name: str = "docs",
    oauth2_flow: str | None = None,
    delegate_auth_to_upstream: bool = False,
    authentication_token: str | None = None,
    alias: str | None = None,
) -> MCPServer:
    return MCPServer(
        server_id=f"{name}-id",
        name=name,
        alias=alias or name,
        url="https://upstream.example/mcp",
        transport=MCPTransport.http,
        auth_type=auth_type,
        oauth2_flow=oauth2_flow,
        delegate_auth_to_upstream=delegate_auth_to_upstream,
        authentication_token=authentication_token,
    )


_VAULTED: Final = _server(MCPAuth.oauth2, oauth2_flow="authorization_code")
_PEER: Final = _server(MCPAuth.none, name="wiki")
_STATIC: Final = _server(MCPAuth.api_key, authentication_token="k")
_REAUTHORIZABLE: Final = [
    pytest.param(_VAULTED, id="vaulted-per-user-oauth"),
    pytest.param(_server(MCPAuth.true_passthrough), id="true-passthrough"),
    pytest.param(_server(MCPAuth.oauth_delegate), id="oauth-delegate"),
]
_NOT_REAUTHORIZABLE: Final = [
    pytest.param(_server(MCPAuth.none), id="no-auth"),
    pytest.param(_STATIC, id="static-api-key"),
    pytest.param(_server(MCPAuth.bearer_token, authentication_token="k"), id="static-bearer"),
    pytest.param(_server(MCPAuth.oauth2, oauth2_flow="client_credentials"), id="client-credentials"),
    pytest.param(_server(MCPAuth.oauth2, delegate_auth_to_upstream=True), id="legacy-upstream-delegate"),
]


def _request_context(gate: OAuthScopeResponse, method: str) -> ServerRequestContext:
    return ServerRequestContext(
        session=MagicMock(),
        lifespan_context={},
        protocol_version="2025-11-25",
        method=method,
        request=Request({"type": "http", SCOPE_RESPONSE_KEY: gate}),
    )


def _challenged(
    server: MCPServer,
    outcome: ServerListFault | ServerListOk,
    listed: Sequence[MCPServer] | None = None,
    reauthorize_as: str | None = "docs",
    method: str = "tools/list",
) -> bool:
    gate: Final = OAuthScopeResponse(AsyncMock(), _GATEWAY, "/docs/mcp", reauthorize_as=reauthorize_as)
    token: Final = active_mcp_request_ctx_var.set(_request_context(gate, method))
    try:
        catalog.challenge_rejected_credential(server, outcome, [server] if listed is None else listed)
    finally:
        active_mcp_request_ctx_var.reset(token)
    return not gate.pending


@pytest.mark.parametrize("server", _REAUTHORIZABLE)
def test_a_rejected_credential_the_caller_can_renew_is_challenged(server: MCPServer) -> None:
    assert _challenged(server, _UNAUTHORIZED)


@pytest.mark.parametrize("server", _NOT_REAUTHORIZABLE)
def test_a_rejected_credential_the_caller_cannot_renew_stays_absorbed(server: MCPServer) -> None:
    assert not _challenged(server, _UNAUTHORIZED)


@pytest.mark.parametrize(
    "outcome",
    [
        ServerListOk(tool_count=3),
        ServerListFault(tag="forbidden", status_code=403),
        ServerListFault(tag="upstream_error", status_code=500),
        ServerListFault(tag="unreachable"),
    ],
)
def test_only_an_unauthorized_outcome_is_challenged(outcome: ServerListFault | ServerListOk) -> None:
    assert not _challenged(_VAULTED, outcome)


def test_a_listing_that_spans_several_servers_is_not_challenged() -> None:
    assert not _challenged(_VAULTED, _UNAUTHORIZED, [_VAULTED, _PEER])


def test_a_request_that_is_not_a_single_server_listing_is_not_challenged() -> None:
    assert not _challenged(_VAULTED, _UNAUTHORIZED, reauthorize_as=None)


@pytest.mark.parametrize("method", ["tools/call", "server/discover", "initialize"])
def test_a_request_the_sdk_parsed_as_something_other_than_a_listing_is_not_challenged(method: str) -> None:
    assert not _challenged(_VAULTED, _UNAUTHORIZED, method=method)


@pytest.mark.asyncio
async def test_a_rejected_stored_token_is_answered_with_the_gateways_own_challenge() -> None:
    send: Final = AsyncMock()
    gate: Final = OAuthScopeResponse(send, _GATEWAY, "/docs/mcp", reauthorize_as="docs")
    token: Final = active_mcp_request_ctx_var.set(_request_context(gate, "tools/list"))
    try:
        started: Final = asyncio.create_task(gate({"type": "http.response.start", "status": 200, "headers": []}))
        await asyncio.sleep(0)
        catalog.challenge_rejected_credential(_VAULTED, _UNAUTHORIZED, [_VAULTED])
        await asyncio.wait_for(started, 1)
    finally:
        active_mcp_request_ctx_var.reset(token)

    start: Final = send.await_args_list[0].args[0]
    assert start["status"] == 401
    assert dict(start["headers"])[b"www-authenticate"].decode() == (
        'Bearer resource_metadata="https://gateway.example/.well-known/oauth-protected-resource/docs/mcp"'
    )


def _upstream_rejecting_the_token(_request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(401, headers={"www-authenticate": 'Bearer error="invalid_token"'})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server", "peers", "challenged"),
    [
        pytest.param(_server(MCPAuth.true_passthrough), (), True, id="true-passthrough"),
        pytest.param(_server(MCPAuth.oauth_delegate), (), True, id="oauth-delegate"),
        pytest.param(_STATIC, (), False, id="static-api-key"),
        pytest.param(_server(MCPAuth.true_passthrough), (_PEER,), False, id="selected-group-of-two"),
    ],
)
async def test_tools_listing_challenges_a_single_server_whose_token_the_upstream_rejected(
    monkeypatch: pytest.MonkeyPatch, server: MCPServer, peers: Sequence[MCPServer], challenged: bool
) -> None:
    manager: Final = MCPServerManager(upstream_transport=httpx2.MockTransport(_upstream_rejecting_the_token))
    monkeypatch.setattr(mcp_operations, "global_mcp_server_manager", manager)
    gate: Final = OAuthScopeResponse(AsyncMock(), _GATEWAY, "/docs/mcp", reauthorize_as="docs")
    context: Final = mcp_operations.prepare_context(
        UserAPIKeyAuth(token="h", user_id="u1"), oauth2_headers={"Authorization": "Bearer rejected"}
    )
    request_token: Final = active_mcp_request_ctx_var.set(_request_context(gate, "tools/list"))
    name_token: Final = mcp_gateway_server_name.set("docs")
    try:
        listing, outcome = await catalog.get_filtered_server_tools(
            server, context=context, allowed_mcp_servers=[server, *peers], prefetched_oauth_creds={}
        )
    finally:
        mcp_gateway_server_name.reset(name_token)
        active_mcp_request_ctx_var.reset(request_token)

    assert listing.tools == []
    assert outcome.tag == "auth_required"
    assert (not gate.pending) is challenged


@pytest.mark.asyncio
@pytest.mark.parametrize("request_name", ["ListPromptsRequest", "ListResourcesRequest", "ListResourceTemplatesRequest"])
@pytest.mark.parametrize(
    ("server", "challenged"),
    [
        pytest.param(_VAULTED, True, id="vaulted-per-user-oauth"),
        pytest.param(_STATIC, False, id="static-api-key"),
    ],
)
async def test_optional_listing_challenges_a_server_whose_stored_token_was_rejected(
    monkeypatch: pytest.MonkeyPatch, request_name: str, server: MCPServer, challenged: bool
) -> None:
    from mcp import types

    from litellm.proxy import proxy_server

    monkeypatch.setenv("LITELLM_SALT_KEY", "listing-reauth-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(mcp_operations.global_mcp_server_manager, "registry", {server.server_id: server})
    monkeypatch.setattr(mcp_operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[server]))
    monkeypatch.setattr(
        catalog, "fetch_optional_catalog_page", AsyncMock(side_effect=MCPUpstreamAuthError(401, None, "docs"))
    )
    gate: Final = OAuthScopeResponse(AsyncMock(), _GATEWAY, "/docs/mcp", reauthorize_as="docs")
    token: Final = active_mcp_request_ctx_var.set(_request_context(gate, "prompts/list"))
    try:
        result: Final = await catalog.list_gateway_catalog(
            mcp_operations.prepare_context(), getattr(types, request_name)()
        )
    finally:
        active_mcp_request_ctx_var.reset(token)

    assert next(iter(result.meta[SERVER_OUTCOMES_META_KEY].values())) == {"status": "auth_required", "http_status": 401}
    assert (not gate.pending) is challenged


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_name", "result_name", "items"),
    [
        ("ListPromptsRequest", "ListPromptsResult", "prompts"),
        ("ListResourcesRequest", "ListResourcesResult", "resources"),
        ("ListResourceTemplatesRequest", "ListResourceTemplatesResult", "resource_templates"),
    ],
)
async def test_optional_listing_challenges_when_the_token_is_rejected_on_a_later_page(
    monkeypatch: pytest.MonkeyPatch, request_name: str, result_name: str, items: str
) -> None:
    from mcp import types

    from litellm.proxy import proxy_server

    monkeypatch.setenv("LITELLM_SALT_KEY", "listing-reauth-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(mcp_operations.global_mcp_server_manager, "registry", {_VAULTED.server_id: _VAULTED})
    monkeypatch.setattr(mcp_operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[_VAULTED]))
    first_page: Final = getattr(types, result_name)(**{items: [], "next_cursor": "next"})
    monkeypatch.setattr(
        catalog,
        "fetch_optional_catalog_page",
        AsyncMock(side_effect=[first_page, MCPUpstreamAuthError(401, None, "docs")]),
    )
    gate: Final = OAuthScopeResponse(AsyncMock(), _GATEWAY, "/docs/mcp", reauthorize_as="docs")
    token: Final = active_mcp_request_ctx_var.set(_request_context(gate, "prompts/list"))
    request: Final = getattr(types, request_name)
    try:
        first: Final = await catalog.list_gateway_catalog(mcp_operations.prepare_context(), request())
        assert first.next_cursor
        assert gate.pending
        with pytest.raises(MCPError, match="Upstream continuation failed"):
            await catalog.list_gateway_catalog(
                mcp_operations.prepare_context(),
                request(params=types.PaginatedRequestParams(cursor=first.next_cursor)),
            )
    finally:
        active_mcp_request_ctx_var.reset(token)

    assert not gate.pending


@pytest.mark.parametrize(
    ("request_path", "resource"),
    [("/mcp/docs", "litellm/mcp/docs"), ("/docs/mcp", "litellm/docs/mcp")],
)
def test_the_challenge_names_the_metadata_route_under_a_server_root_path(
    monkeypatch: pytest.MonkeyPatch, request_path: str, resource: str
) -> None:
    monkeypatch.setenv("SERVER_ROOT_PATH", "/litellm")

    challenge: Final = MCPUpstreamAuthError(401, None, "docs").to_http_exception(_GATEWAY, request_path)

    assert (challenge.headers or {})["www-authenticate"] == (
        f'Bearer resource_metadata="https://gateway.example/.well-known/oauth-protected-resource/{resource}"'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_path", "addressed", "resource"),
    [
        pytest.param("/docs/mcp", "docs", "docs/mcp", id="per-server-path"),
        pytest.param("/Docs/mcp", "Docs", "Docs/mcp", id="other-case"),
        pytest.param("/docs_internal/mcp", "docs_internal", "docs_internal/mcp", id="name-not-alias"),
        pytest.param("/mcp/Docs", "Docs", "mcp/Docs", id="standard-path-other-case"),
        pytest.param("/mcp/a:b@c+d,e", "a:b@c+d,e", "mcp/a:b@c+d,e", id="path-legal-characters-stay-as-typed"),
        pytest.param("/mcp/a b%c", "a b%c", "mcp/a%20b%25c", id="path-illegal-characters-are-encoded"),
        pytest.param("/mcp/日本", "日本", "mcp/%E6%97%A5%E6%9C%AC", id="non-ascii-name"),
        pytest.param("/日本/mcp", "日本", "%E6%97%A5%E6%9C%AC/mcp", id="non-ascii-name-legacy-path"),
    ],
)
async def test_the_challenge_names_the_url_the_client_addressed(
    request_path: str, addressed: str, resource: str
) -> None:
    send: Final = AsyncMock()
    gate: Final = OAuthScopeResponse(send, _GATEWAY, request_path, reauthorize_as=addressed)
    aliased: Final = _server(MCPAuth.oauth2, name="docs_internal", oauth2_flow="authorization_code", alias="docs")
    token: Final = active_mcp_request_ctx_var.set(_request_context(gate, "tools/list"))
    try:
        started: Final = asyncio.create_task(gate({"type": "http.response.start", "status": 200, "headers": []}))
        await asyncio.sleep(0)
        catalog.challenge_rejected_credential(aliased, _UNAUTHORIZED, [aliased])
        await asyncio.wait_for(started, 1)
    finally:
        active_mcp_request_ctx_var.reset(token)

    start: Final = send.await_args_list[0].args[0]
    assert dict(start["headers"])[b"www-authenticate"].decode() == (
        f'Bearer resource_metadata="https://gateway.example/.well-known/oauth-protected-resource/{resource}"'
    )
