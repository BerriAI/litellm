"""Tests for the v2 egress manager: factory, skeleton, and the egress override (step 6)."""

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
