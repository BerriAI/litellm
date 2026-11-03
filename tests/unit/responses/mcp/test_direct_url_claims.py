from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import httpx2
import pytest
from mcp.types import CallToolRequestParams, CallToolResult
from mcp.types import Tool as MCPTool

from litellm.exceptions import APIError, UnsupportedParamsError
from litellm.responses.mcp.direct_url_claims import (
    DirectMCPServer,
    allowed_tool_names,
    list_direct_tools,
    normalized_mcp_url,
    plan_hosted_mcp_claims,
)
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

DEEPWIKI: Final = MCPServer(
    server_id="srv-deepwiki",
    name="deepwiki",
    server_name="deepwiki",
    url="https://mcp.deepwiki.com/mcp",
    transport=MCPTransport.http,
)
DEEPWIKI_MIRROR: Final = MCPServer(
    server_id="srv-mirror",
    name="wiki-mirror",
    alias="wiki-mirror",
    url="https://MCP.deepwiki.com/mcp/",
    transport=MCPTransport.http,
)
FUNCTION_TOOL: Final = {"type": "function", "name": "noop", "parameters": {"type": "object"}}


def _mcp_tool(server_url: str, extra: Mapping[str, object] = MappingProxyType({})) -> Mapping[str, object]:
    return {"type": "mcp", "server_label": "wiki", "server_url": server_url, "require_approval": "never", **extra}


def _plan(
    tools: Sequence[Mapping[str, object]],
    servers: Sequence[MCPServer] = (DEEPWIKI,),
    allowed_hosts: Sequence[str] = (),
    drop_params: object = None,
):
    return plan_hosted_mcp_claims(
        tools, servers, allowed_hosts, drop_params=drop_params, model="deepseek-chat", route="deepseek"
    )


@pytest.mark.parametrize(
    ("url", "normalized"),
    [
        ("HTTPS://MCP.DeepWiki.com:443/mcp/", "https://mcp.deepwiki.com/mcp"),
        ("https://mcp.deepwiki.com./mcp#section", "https://mcp.deepwiki.com/mcp"),
        ("http://mcp.example.com:8080/mcp?tenant=a", "http://mcp.example.com:8080/mcp?tenant=a"),
        ("http://mcp.example.com:80", "http://mcp.example.com"),
        ("https://[2001:db8::1]:8443/mcp", "https://[2001:db8::1]:8443/mcp"),
        ("ftp://mcp.example.com/mcp", None),
        ("https://user:secret@mcp.example.com/mcp", None),
        ("https://mcp.example.com:port/mcp", None),
        ("litellm_proxy/mcp/deepwiki", None),
    ],
)
def test_normalized_mcp_url_matches_urls_that_name_the_same_endpoint(url: str, normalized: str | None):
    assert normalized_mcp_url(url) == normalized


def test_a_url_neither_registered_nor_allowlisted_is_left_alone():
    assert _plan([_mcp_tool("https://mcp.unknown.example.com/mcp")], allowed_hosts=("mcp.example.com",)) is None


def test_a_registered_url_runs_as_its_gateway_server_under_the_callers_label():
    claims: Final = _plan([FUNCTION_TOOL, _mcp_tool("https://Mcp.DeepWiki.com/mcp/", {"headers": {"X-Team": "a"}})])

    assert claims is not None
    assert [dict(tool) for tool in claims.tools] == [
        FUNCTION_TOOL,
        {**_mcp_tool("litellm_proxy/mcp/deepwiki"), "headers": {"X-Team": "a"}},
    ]
    assert claims.direct == ()
    assert claims.server_labels == {"deepwiki": "wiki", "srv-deepwiki": "wiki"}


def test_a_registered_server_known_only_by_its_id_is_reached_through_the_id():
    unnamed: Final = MCPServer(
        server_id="srv-unnamed",
        name="srv-unnamed",
        url="https://mcp.unnamed.example.com/mcp",
        transport=MCPTransport.http,
    )
    claims: Final = _plan([_mcp_tool("https://mcp.unnamed.example.com/mcp")], servers=(unnamed,))

    assert claims is not None
    assert [tool["server_url"] for tool in claims.tools] == ["litellm_proxy/mcp/srv-unnamed"]


def test_a_url_two_registered_servers_share_runs_only_the_one_the_label_names():
    tools: Final = [_mcp_tool("https://mcp.deepwiki.com/mcp")]
    named: Final = _plan([{**tools[0], "server_label": "wiki-mirror"}], servers=(DEEPWIKI, DEEPWIKI_MIRROR))

    assert _plan(tools, servers=(DEEPWIKI, DEEPWIKI_MIRROR)) is None
    assert named is not None
    assert [tool["server_url"] for tool in named.tools] == ["litellm_proxy/mcp/wiki-mirror"]


def test_an_allowlisted_host_is_run_directly_with_only_the_callers_own_credentials():
    claims: Final = _plan(
        [
            FUNCTION_TOOL,
            _mcp_tool(
                "https://mcp.example.com/mcp",
                {"headers": {"X-Team": "a"}, "authorization": "caller-token", "allowed_tools": ["search"]},
            ),
        ],
        allowed_hosts=("mcp.example.com",),
    )

    assert claims is not None
    assert [dict(tool) for tool in claims.tools] == [FUNCTION_TOOL]
    assert claims.direct == (
        DirectMCPServer(
            server_label="wiki",
            server_url="https://mcp.example.com/mcp",
            headers={"X-Team": "a", "Authorization": "Bearer caller-token"},
            allowed_tools=frozenset({"search"}),
        ),
    )


def test_a_callers_authorization_header_is_not_replaced_by_the_authorization_field():
    claims: Final = _plan(
        [
            _mcp_tool(
                "https://mcp.example.com/mcp", {"headers": {"authorization": "Basic abc"}, "authorization": "other"}
            )
        ],
        allowed_hosts=("mcp.example.com",),
    )

    assert claims is not None
    assert [server.headers for server in claims.direct] == [{"authorization": "Basic abc"}]


def test_routing_and_framing_headers_the_caller_sets_are_not_sent_to_the_direct_server():
    caller_headers: Final = {
        "Host": "admin.internal.example.com",
        "X-Forwarded-For": "127.0.0.1",
        "x-forwarded-host": "admin.internal.example.com",
        "Forwarded": "for=127.0.0.1",
        "Connection": "close",
        "Transfer-Encoding": "chunked",
        "X-Team": "a",
    }
    claims: Final = _plan(
        [_mcp_tool("https://mcp.example.com/mcp", {"headers": caller_headers})], allowed_hosts=("mcp.example.com",)
    )

    assert claims is not None
    assert [server.headers for server in claims.direct] == [{"X-Team": "a"}]


@pytest.mark.parametrize(
    "server_url",
    ["https://mcp.deepwiki.com/./mcp", "https://mcp.deepwiki.com/a/../mcp", "https://mcp.deepwiki.com/mcp?x=1"],
)
def test_a_url_on_a_registered_servers_host_never_bypasses_that_server(server_url: str):
    assert _plan([_mcp_tool(server_url)], allowed_hosts=("mcp.deepwiki.com",)) is None


@pytest.mark.parametrize(
    ("allowed_host", "server_url", "claimed"),
    [
        ("mcp.example.com", "https://mcp.example.com/mcp", True),
        ("mcp.example.com", "http://mcp.example.com/mcp", False),
        ("mcp.example.com", "https://mcp.example.com:6379/mcp", False),
        ("mcp.example.com:8443", "https://mcp.example.com:8443/mcp", True),
        ("mcp.example.com:8443", "http://mcp.example.com:8443/mcp", False),
        ("http://mcp.example.com:8080", "http://mcp.example.com:8080/mcp", True),
    ],
)
def test_an_allowlist_entry_without_a_scheme_admits_only_https(allowed_host: str, server_url: str, claimed: bool):
    assert (_plan([_mcp_tool(server_url)], allowed_hosts=(allowed_host,)) is not None) is claimed


def test_a_gateway_tool_the_caller_sent_is_kept_as_sent_beside_a_claimed_one():
    proxy_tool: Final = _mcp_tool("litellm_proxy/mcp/deepwiki")
    claims: Final = _plan([proxy_tool, _mcp_tool("https://mcp.example.com/mcp")], allowed_hosts=("mcp.example.com",))

    assert _plan([proxy_tool]) is None
    assert claims is not None
    assert [dict(tool) for tool in claims.tools] == [proxy_tool]


@pytest.mark.parametrize("approval", ["always", {"never": {"tool_names": ["search"]}}, None])
@pytest.mark.parametrize(
    ("server_url", "allowed_hosts"),
    [("https://mcp.deepwiki.com/mcp", ()), ("https://mcp.example.com/mcp", ("mcp.example.com",))],
    ids=["registered", "allowlisted"],
)
def test_a_claimed_tool_that_needs_approval_answers_400_or_is_left_for_the_drop(
    approval: object, server_url: str, allowed_hosts: tuple[str, ...]
):
    tool: Final = {**_mcp_tool(server_url), "require_approval": approval}

    with pytest.raises(UnsupportedParamsError) as refused:
        _plan([tool], allowed_hosts=allowed_hosts)

    assert refused.value.status_code == 400
    assert "`wiki`" in refused.value.message
    assert _plan([tool], allowed_hosts=allowed_hosts, drop_params=True) is None


def test_a_tool_that_needs_approval_on_an_unclaimed_url_is_not_answered_here():
    assert _plan([{**_mcp_tool("https://mcp.unknown.example.com/mcp"), "require_approval": "always"}]) is None


@pytest.mark.parametrize(
    ("allowed_tools", "names"),
    [
        (None, None),
        (["search", "fetch"], frozenset({"search", "fetch"})),
        ({"tool_names": ["search"]}, frozenset({"search"})),
        ({"read_only": True}, frozenset()),
        ("search", frozenset()),
    ],
    ids=["absent", "list", "filter", "read_only_filter", "malformed"],
)
def test_allowed_tool_names_fails_closed_on_a_filter_it_cannot_read(
    allowed_tools: object, names: frozenset[str] | None
):
    assert allowed_tool_names({"allowed_tools": allowed_tools}) == names


@dataclass(frozen=True, slots=True)
class _ListingClient:
    tools: tuple[MCPTool, ...]
    failure: Exception | None = None

    async def list_tools(self, raise_on_error: bool = False) -> Sequence[MCPTool]:
        if self.failure is not None and raise_on_error:
            raise self.failure
        return () if self.failure is not None else self.tools

    async def call_tool(self, call_tool_request_params: CallToolRequestParams) -> CallToolResult:
        return CallToolResult(content=[])


def _direct_server(allowed_tools: frozenset[str] | None = None) -> DirectMCPServer:
    return DirectMCPServer(
        server_label="wiki", server_url="https://mcp.example.com/mcp", headers={}, allowed_tools=allowed_tools
    )


@pytest.mark.asyncio
async def test_direct_tools_are_listed_under_the_callers_label_within_allowed_tools():
    client: Final = _ListingClient(
        (MCPTool(name="search", inputSchema={"type": "object"}), MCPTool(name="delete", inputSchema={"type": "object"}))
    )
    server: Final = _direct_server(frozenset({"search"}))

    listing: Final = await list_direct_tools([server], lambda _server: client)

    assert [tool.name for tool in listing.tools] == ["wiki-search"]
    assert {name: (tool.server, tool.tool.name, tool.client) for name, tool in listing.by_name.items()} == {
        "wiki-search": (server, "search", client)
    }
    assert [tool.name for tool in listing.server_tools["wiki"]] == ["search"]


@pytest.mark.asyncio
async def test_a_direct_server_that_cannot_list_its_tools_answers_424():
    client: Final = _ListingClient((), failure=httpx2.ConnectError("refused"))

    with pytest.raises(APIError) as failed:
        await list_direct_tools([_direct_server()], lambda _server: client)

    assert failed.value.status_code == 424
    assert "`wiki`" in failed.value.message
