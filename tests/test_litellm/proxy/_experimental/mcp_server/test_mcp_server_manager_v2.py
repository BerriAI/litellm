"""Tests for the v2 egress manager: factory, skeleton, and the egress override (step 6)."""

import base64
import contextlib
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
    _make_global_mcp_server_manager,
)
from litellm.proxy._experimental.mcp_server.mcp_server_manager_v2 import (
    MCPServerManagerV2,
)
from litellm.proxy._types import MCPTransport
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def test_v2_is_a_manager_subclass():
    assert issubclass(MCPServerManagerV2, MCPServerManager)


def test_factory_constructs_the_v2_manager():
    # v2 is the egress implementation; there is no opt-in flag.
    assert isinstance(_make_global_mcp_server_manager(), MCPServerManagerV2)


@contextlib.contextmanager
def _serve_echo():
    """A no-auth streamable-http FastMCP server with one `echo` tool, in a background thread."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("mgr-echo-test", stateless_http=True)

    @mcp.tool()
    def echo(text: str) -> str:
        return f"echo: {text}"

    @mcp.prompt()
    def greeting(name: str) -> str:
        return f"Hello, {name}"

    @mcp.resource("echo://info")
    def info() -> str:
        return "echo server info"

    @mcp.resource("echo://item/{item_id}")
    def item(item_id: str) -> str:
        return f"item {item_id}"

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    url = f"http://127.0.0.1:{port}/mcp"
    server = uvicorn.Server(
        uvicorn.Config(
            mcp.streamable_http_app(), host="127.0.0.1", port=port, log_level="error"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        try:
            httpx.get(url, timeout=0.3)
            break
        except httpx.ConnectError:
            time.sleep(0.05)
        except Exception:
            break
    try:
        yield url
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture
def echo_server_url():
    with _serve_echo() as url:
        yield url


@pytest.mark.asyncio
async def test_v2_override_lists_tools_via_upstream_connection(echo_server_url):
    # The `none`-mode list path goes through resolve() + UpstreamConnection (v2), and the tools come
    # back namespaced via the inherited _create_prefixed_tools.
    manager = MCPServerManagerV2()
    server = MCPServer(
        server_id="echo1",
        name="echo1",
        transport=MCPTransport.http,
        url=echo_server_url,
        auth_type=MCPAuth.none,
    )
    tools = await manager._get_tools_from_server(server, add_prefix=True)
    assert any(t.name.endswith("echo") for t in tools)


@pytest.mark.asyncio
async def test_v2_override_lists_prompts_via_upstream_connection(echo_server_url):
    # The none-mode prompts path goes through resolve() + UpstreamConnection.list_prompts (v2),
    # namespaced via the inherited _create_prefixed_prompts.
    manager = MCPServerManagerV2()
    server = MCPServer(
        server_id="echo1",
        name="echo1",
        transport=MCPTransport.http,
        url=echo_server_url,
        auth_type=MCPAuth.none,
    )
    prompts = await manager.get_prompts_from_server(server, add_prefix=True)
    assert any("greeting" in p.name for p in prompts)


@pytest.mark.asyncio
async def test_v2_override_lists_resources_via_upstream_connection(echo_server_url):
    # The none-mode resources path goes through resolve() + UpstreamConnection.list_resources (v2).
    manager = MCPServerManagerV2()
    server = MCPServer(
        server_id="echo1",
        name="echo1",
        transport=MCPTransport.http,
        url=echo_server_url,
        auth_type=MCPAuth.none,
    )
    resources = await manager.get_resources_from_server(server, add_prefix=True)
    assert len(resources) >= 1


@pytest.mark.asyncio
async def test_v2_override_lists_resource_templates_via_upstream_connection(
    echo_server_url,
):
    # resources/templates/list path: resolve() + UpstreamConnection.list_resource_templates (v2),
    # namespaced via the inherited _create_prefixed_resource_templates.
    manager = MCPServerManagerV2()
    server = MCPServer(
        server_id="echo1",
        name="echo1",
        transport=MCPTransport.http,
        url=echo_server_url,
        auth_type=MCPAuth.none,
    )
    templates = await manager.get_resource_templates_from_server(
        server, add_prefix=True
    )
    assert len(templates) >= 1


@pytest.mark.asyncio
async def test_v2_override_reads_resource_via_upstream_connection(echo_server_url):
    # Single-result read path: resolve() + UpstreamConnection.read_resource (v2), raises on failure.
    from pydantic import AnyUrl

    manager = MCPServerManagerV2()
    server = MCPServer(
        server_id="echo1",
        name="echo1",
        transport=MCPTransport.http,
        url=echo_server_url,
        auth_type=MCPAuth.none,
    )
    result = await manager.read_resource_from_server(server, AnyUrl("echo://info"))
    assert result.contents


@pytest.mark.asyncio
async def test_v2_override_gets_prompt_via_upstream_connection(echo_server_url):
    # Single-result prompt path: resolve() + UpstreamConnection.get_prompt (v2), raises on failure.
    manager = MCPServerManagerV2()
    server = MCPServer(
        server_id="echo1",
        name="echo1",
        transport=MCPTransport.http,
        url=echo_server_url,
        auth_type=MCPAuth.none,
    )
    result = await manager.get_prompt_from_server(server, "greeting", {"name": "Tin"})
    assert result.messages


@pytest.mark.asyncio
async def test_v2_override_calls_tool_via_upstream_connection(echo_server_url):
    # Tool-call path: the factored _open_and_call_tool seam routes through resolve() +
    # UpstreamConnection.call_tool (v2) for the none mode, returning the upstream CallToolResult.
    manager = MCPServerManagerV2()
    server = MCPServer(
        server_id="echo1",
        name="echo1",
        transport=MCPTransport.http,
        url=echo_server_url,
        auth_type=MCPAuth.none,
    )
    result = await manager._open_and_call_tool(
        server,
        "echo",
        {"text": "hi"},
        mcp_auth_header=None,
        mcp_server_auth_headers=None,
        oauth2_headers=None,
        raw_headers=None,
        hook_extra_headers=None,
        host_progress_callback=None,
        user_api_key_auth=None,
    )
    assert any("echo: hi" in getattr(c, "text", "") for c in result.content)


def _passthrough_server(url):
    return MCPServer(
        server_id="pt",
        name="pt",
        transport=MCPTransport.http,
        url=url,
        auth_type=MCPAuth.oauth2,
        delegate_auth_to_upstream=True,
        client_id="cid",
        authorization_url="https://idp/auth",
        token_url="https://idp/token",
    )


@pytest.mark.asyncio
async def test_v2_passthrough_forwards_inbound_token(echo_server_url):
    # Passthrough: the caller token is extracted and threaded as inbound_token, so the v2 call
    # reaches the upstream (the no-auth echo server ignores the forwarded bearer and serves it).
    manager = MCPServerManagerV2()
    result = await manager._open_and_call_tool(
        _passthrough_server(echo_server_url),
        "echo",
        {"text": "hi"},
        mcp_auth_header=None,
        mcp_server_auth_headers=None,
        oauth2_headers={"Authorization": "Bearer caller-token"},
        raw_headers=None,
        hook_extra_headers=None,
        host_progress_callback=None,
        user_api_key_auth=None,
    )
    assert any("echo: hi" in getattr(c, "text", "") for c in result.content)


@pytest.mark.asyncio
async def test_v2_passthrough_without_token_fails_closed(echo_server_url):
    # No caller token -> inbound_token is None -> the passthrough arm fails closed (401), surfaced
    # as MCPUpstreamAuthError. Before the inbound-token plumbing, the token was never threaded
    # through, so every passthrough call hit this path.
    from litellm.proxy._experimental.mcp_server.exceptions import MCPUpstreamAuthError

    manager = MCPServerManagerV2()
    with pytest.raises(MCPUpstreamAuthError):
        await manager._open_and_call_tool(
            _passthrough_server(echo_server_url),
            "echo",
            {"text": "hi"},
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers=None,
            hook_extra_headers=None,
            host_progress_callback=None,
            user_api_key_auth=None,
        )


# --- _v2_connection header-assembly + isolation matrix ---
# Run the full _v2_connection seam per mode and assert the EXACT upstream headers (resolve()'s auth
# merged with the metadata headers), so a mode can never silently attach a wrong, missing, or extra
# header, and the caller's credentials never leak where they shouldn't.


def _egress_server(**kwargs):
    return MCPServer(
        server_id="s",
        name="s",
        transport=MCPTransport.http,
        url="https://up.example/mcp",
        **kwargs,
    )


def _final_headers(conn_ok):
    # Exact upstream headers = metadata headers + what resolve()'s Auth attaches (auth wins on conflict).
    # White-box: read the connection's resolved auth + metadata headers directly.
    from litellm.proxy._experimental.mcp_server.v2_resolver_bridge import _added_headers

    return {**(conn_ok._extra_headers or {}), **_added_headers(conn_ok._auth)}


async def _connect(server, **kwargs):
    from litellm.proxy.gateway.mcp.result import Ok

    manager = MCPServerManagerV2()
    conn = await manager._v2_connection(
        server, kwargs.pop("user_api_key_auth", None), **kwargs
    )
    assert isinstance(conn, Ok), f"_v2_connection returned {conn!r}"
    return conn.ok


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"auth_type": MCPAuth.none}, {}),
        (
            {"auth_type": MCPAuth.api_key, "authentication_token": "k"},
            {"X-API-Key": "k"},
        ),
        (
            {"auth_type": MCPAuth.bearer_token, "authentication_token": "k"},
            {"Authorization": "Bearer k"},
        ),
        (
            {"auth_type": MCPAuth.authorization, "authentication_token": "Raw v"},
            {"Authorization": "Raw v"},
        ),
        (
            {"auth_type": MCPAuth.token, "authentication_token": "k"},
            {"Authorization": "token k"},
        ),
        (
            {"auth_type": MCPAuth.basic, "authentication_token": "u:p"},
            {"Authorization": f"Basic {base64.b64encode(b'u:p').decode()}"},
        ),
    ],
)
async def test_v2_connection_builds_exact_headers_per_mode(kwargs, expected):
    conn = await _connect(_egress_server(**kwargs))
    assert _final_headers(conn) == expected  # exact match -> no missing/extra headers


@pytest.mark.asyncio
async def test_v2_connection_passthrough_forwards_only_the_inbound_token():
    conn = await _connect(
        _passthrough_server("https://up.example/mcp"), subject_token="caller-tok"
    )
    assert _final_headers(conn) == {"Authorization": "Bearer caller-tok"}


@pytest.mark.asyncio
async def test_v2_connection_merges_static_headers_without_extras():
    conn = await _connect(
        _egress_server(
            auth_type=MCPAuth.api_key,
            authentication_token="k",
            static_headers={"X-Custom": "v"},
        )
    )
    assert _final_headers(conn) == {"X-API-Key": "k", "X-Custom": "v"}


@pytest.mark.asyncio
async def test_v2_connection_strips_caller_authorization_but_forwards_configured_headers():
    # The server forwards [Authorization, X-Tenant]; Authorization is the gateway credential and must
    # be stripped, X-Tenant forwarded, and the upstream credential comes from resolve().
    conn = await _connect(
        _egress_server(
            auth_type=MCPAuth.api_key,
            authentication_token="k",
            extra_headers=["Authorization", "X-Tenant"],
        ),
        raw_headers={"Authorization": "Bearer GATEWAY-CRED", "X-Tenant": "t1"},
        forward_caller_headers=True,
    )
    assert _final_headers(conn) == {"X-API-Key": "k", "X-Tenant": "t1"}


@pytest.mark.asyncio
async def test_v2_connection_does_not_leak_inbound_token_into_non_obo_modes():
    # The inbound token is always put on the Subject, but a non-OBO mode (api_key) ignores it: the
    # headers carry only the configured credential, never the caller's token (credential isolation).
    conn = await _connect(
        _egress_server(auth_type=MCPAuth.api_key, authentication_token="k"),
        subject_token="LEAK-ME",
    )
    headers = _final_headers(conn)
    assert headers == {"X-API-Key": "k"}
    assert "LEAK-ME" not in str(headers)
