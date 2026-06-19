"""Tests for the v2 MCP egress transport: the flag and the UpstreamConnection."""

import socket
import threading
import time

import httpx
import pytest
import uvicorn

from litellm.proxy._experimental.mcp_server.v2_egress import v2_egress_enabled

FLAG = "LITELLM_USE_V2_MCP_EGRESS"


def test_egress_flag_off_by_default(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert v2_egress_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on"])
def test_egress_flag_truthy_values(monkeypatch, value):
    monkeypatch.setenv(FLAG, value)
    assert v2_egress_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "", "  "])
def test_egress_flag_falsey_values(monkeypatch, value):
    monkeypatch.setenv(FLAG, value)
    assert v2_egress_enabled() is False


@pytest.fixture
def echo_server_url():
    """A no-auth streamable-http FastMCP server with one `echo` tool, in a background thread."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("egress-echo-test", stateless_http=True)

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
    for _ in range(100):  # wait until the app responds (any HTTP status means it is up)
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


@pytest.mark.asyncio
async def test_upstream_connection_lists_and_calls(echo_server_url):
    from litellm.proxy._experimental.mcp_server.v2_egress import UpstreamConnection
    from litellm.proxy.gateway.mcp.outbound_credentials.httpx_auth import NoOpAuth
    from litellm.proxy.gateway.mcp.result import Ok

    conn = UpstreamConnection(echo_server_url, auth=NoOpAuth())

    tools = await conn.list_tools()
    assert isinstance(tools, Ok)
    assert any(t.name == "echo" for t in tools.ok)

    called = await conn.call_tool("echo", {"text": "hi"})
    assert isinstance(called, Ok)
    assert called.ok.isError is False


@pytest.mark.asyncio
async def test_upstream_connection_unreachable_is_upstream_unavailable():
    from litellm.proxy._experimental.mcp_server.v2_egress import UpstreamConnection
    from litellm.proxy.gateway.mcp.result import Error

    conn = UpstreamConnection("http://127.0.0.1:1/mcp", timeout=3.0)
    result = await conn.list_tools()
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"
