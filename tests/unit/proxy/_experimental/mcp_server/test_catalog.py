import asyncio
from collections.abc import Sequence

import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import ListToolsResult, Tool

from litellm.proxy._experimental.mcp_server import catalog


def page(name: str, cursor: str | None = None, revision: str = "stable") -> ListToolsResult:
    return ListToolsResult(
        tools=[Tool(name=name, input_schema={"type": "object"})],
        next_cursor=cursor,
        meta={"revision": revision},
    )


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
        return page("b2" if cursor else "b1", None if cursor else "next")

    first = await listing(fetch)
    assert [tool.name for tool in first.tools] == ["b1"]
    assert first.meta[SERVER_OUTCOMES_META_KEY]["a"] == {"tag": "timeout"}
    second = await listing(fetch, first.next_cursor)
    assert [tool.name for tool in second.tools] == ["b2"]
    assert second.meta[SERVER_OUTCOMES_META_KEY]["a"] == {"tag": "timeout"}


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
