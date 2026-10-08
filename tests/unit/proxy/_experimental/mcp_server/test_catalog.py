import asyncio
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Final, Literal
from unittest.mock import AsyncMock

import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import (
    ListPromptsRequest,
    ListPromptsResult,
    ListResourcesRequest,
    ListResourcesResult,
    ListResourceTemplatesRequest,
    ListResourceTemplatesResult,
    ListToolsResult,
    PaginatedRequestParams,
    Tool,
)

from litellm.proxy._experimental.mcp_server import catalog
from litellm.proxy._experimental.mcp_server.contracts import OperationContext
from litellm.proxy._experimental.mcp_server.faults.list_outcomes import (
    SERVER_OUTCOMES_META_KEY,
    AggregateToolListing,
    ServerListOk,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

CatalogKind = Literal["tools", "prompts", "resources", "templates"]
OptionalCatalogResult = ListPromptsResult | ListResourcesResult | ListResourceTemplatesResult


def page(name: str, cursor: str | None = None, revision: str = "stable") -> ListToolsResult:
    return ListToolsResult(
        tools=[Tool(name=name, input_schema={"type": "object"})],
        next_cursor=cursor,
        meta={"revision": revision},
    )


def rate_limit_catalog_setup(
    monkeypatch: pytest.MonkeyPatch, rejected_server_ids: frozenset[str]
) -> tuple[tuple[MCPServer, MCPServer], OperationContext, AsyncMock]:
    from litellm.proxy import proxy_server
    from litellm.proxy._experimental.mcp_server import operations

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
    monkeypatch.setattr(operations.global_mcp_server_manager, "registry", {server.server_id: server for server in servers})
    monkeypatch.setattr(operations, "_get_allowed_mcp_servers", AsyncMock(return_value=list(servers)))
    context: Final = operations.prepare_context(
        user_api_key_auth=caller,
        mcp_servers=[server.server_id for server in servers],
    )
    return servers, context, limiter


def optional_catalog_page(
    kind: CatalogKind, server_id: str, next_cursor: str | None = None
) -> OptionalCatalogResult:
    from mcp import types

    if kind == "prompts":
        return types.ListPromptsResult(
            prompts=[types.Prompt(name=f"{server_id}-item")], next_cursor=next_cursor
        )
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
        return await catalog.aggregate_gateway_tools(
            context, PaginatedRequestParams(cursor=cursor), servers, {}
        )
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
    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import SERVER_OUTCOMES_META_KEY

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
    from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Ok
    from litellm.proxy._experimental.mcp_server.state_tokens import seal_state

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

    from litellm.proxy._experimental.mcp_server.contracts import OperationContext
    from litellm.proxy._types import UserAPIKeyAuth

    original = OperationContext(_caller=UserAPIKeyAuth(api_key="synthetic-key", user_id="owner"))
    assert catalog._caller_scope(original, ()) != catalog._caller_scope(replace(original, **change), ())


def test_caller_binding_ignores_transport_headers_and_header_case():
    from dataclasses import replace

    from litellm.proxy._experimental.mcp_server.contracts import OperationContext
    from litellm.proxy._types import UserAPIKeyAuth

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
async def test_gateway_tools_continuation_rejects_an_upstream_failure(monkeypatch):
    from unittest.mock import AsyncMock

    from mcp.types import PaginatedRequestParams

    from litellm.proxy import proxy_server
    from litellm.proxy._experimental.mcp_server import operations
    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import ServerListOk, classify_list_exception
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    monkeypatch.setenv("LITELLM_SALT_KEY", "catalog-failure-test")
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    server = MCPServer(server_id="pages", name="pages", transport=MCPTransport.http)
    monkeypatch.setattr(operations.global_mcp_server_manager, "registry", {server.server_id: server})
    fetch = AsyncMock(side_effect=[
        (page("first", "next"), ServerListOk(tool_count=1)),
        (ListToolsResult(tools=[]), classify_list_exception(TimeoutError("upstream secret"))),
    ])
    monkeypatch.setattr(catalog, "get_filtered_server_tools", fetch)
    context = operations.prepare_context()
    first = await catalog.aggregate_gateway_tools(context, PaginatedRequestParams(), [server], {})
    assert first.next_cursor and [tool.name for tool in first.tools] == ["first"]
    with pytest.raises(MCPError, match="Upstream continuation failed; start a fresh listing") as denied:
        await catalog.aggregate_gateway_tools(context, PaginatedRequestParams(cursor=first.next_cursor), [server], {})
    assert "upstream secret" not in str(denied.value)
    assert fetch.await_count == 2
    assert fetch.await_args.kwargs["params"].cursor == "next"


@pytest.mark.asyncio
@pytest.mark.parametrize("request_name,result_name,field", [
    ("ListPromptsRequest", "ListPromptsResult", "prompts"),
    ("ListResourcesRequest", "ListResourcesResult", "resources"),
    ("ListResourceTemplatesRequest", "ListResourceTemplatesResult", "resource_templates"),
])
async def test_optional_gateway_catalog_reports_initial_failure_and_rejects_failed_continuation(monkeypatch, request_name, result_name, field):
    from unittest.mock import AsyncMock

    from mcp import types

    from litellm.proxy import proxy_server
    from litellm.proxy._experimental.mcp_server import operations
    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import SERVER_OUTCOMES_META_KEY
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
    fetch.side_effect = [getattr(types, result_name)(**{field: [], "next_cursor": "next"}), TimeoutError("upstream secret")]
    first = await catalog.list_gateway_catalog(context, request())
    assert first.next_cursor
    with pytest.raises(MCPError, match="Upstream continuation failed; start a fresh listing") as denied:
        await catalog.list_gateway_catalog(context, request(params=types.PaginatedRequestParams(cursor=first.next_cursor)))
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
    from mcp import types

    servers, context, limiter = rate_limit_catalog_setup(monkeypatch, frozenset())

    async def fetch_tools(server: MCPServer, **kwargs: object) -> tuple[ListToolsResult, ServerListOk]:
        params: Final = PaginatedRequestParams.model_validate(kwargs["params"])
        next_cursor: Final = "next" if server.server_id == "catalog-a" and params.cursor is None else None
        return (
            ListToolsResult(
                tools=[types.Tool(name=f"{server.server_id}-item", inputSchema={"type": "object"})],
                next_cursor=next_cursor,
            ),
            ServerListOk(tool_count=1),
        )

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
        monkeypatch.setattr(catalog, "get_filtered_server_tools", fetch_tools)
    else:
        monkeypatch.setattr(catalog, "fetch_optional_catalog_page", fetch_optional)

    first_page: Final = await run_catalog_listing(kind, context, servers)
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
    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import SERVER_OUTCOMES_META_KEY

    result: Final = catalog.combine_optional_catalog(
        ListPromptsRequest(),
        [ListPromptsResult(prompts=[], ttl_ms=9000)],
        None,
        {SERVER_OUTCOMES_META_KEY: {"failed-earlier": {"tag": "timeout"}}},
    )
    assert result.ttl_ms == 0
    assert result.cache_scope == "private"

