"""Tests for the v2 MCP egress transport (UpstreamConnection)."""

import contextlib
import socket
import sys
import threading
import time

import httpx
import pytest
import uvicorn


@contextlib.contextmanager
def _serve(app, path="/mcp"):
    """Serve an ASGI app on a free port in a background thread; yield its endpoint url."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    url = f"http://127.0.0.1:{port}{path}"
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
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


def _echo_app(token=None):
    """A stateless streamable-http FastMCP app with one `echo` tool, optionally Bearer-gated."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("egress-test", stateless_http=True)

    @mcp.tool()
    def echo(text: str) -> str:
        return f"echo: {text}"

    @mcp.prompt()
    def greet(name: str) -> str:
        return f"Hello {name}"

    @mcp.resource("data://info")
    def info() -> str:
        return "resource-info"

    app = mcp.streamable_http_app()
    if token is not None:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse

        class _BearerCheck(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                authorized = request.headers.get("authorization") == f"Bearer {token}"
                if request.url.path.startswith("/mcp") and not authorized:
                    return JSONResponse({"error": "unauthorized"}, status_code=401)
                return await call_next(request)

        app.add_middleware(_BearerCheck)
    return app


@pytest.fixture
def echo_server_url():
    with _serve(_echo_app()) as url:
        yield url


@pytest.fixture
def protected_server_url():
    with _serve(_echo_app(token="secret-token")) as url:
        yield url, "secret-token"


@pytest.fixture
def sse_server_url():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("egress-sse-test")

    @mcp.tool()
    def echo(text: str) -> str:
        return f"echo: {text}"

    with _serve(mcp.sse_app(), path="/sse") as url:
        yield url


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


@pytest.mark.asyncio
async def test_upstream_connection_401_is_unauthorized(protected_server_url):
    from litellm.proxy._experimental.mcp_server.v2_egress import UpstreamConnection
    from litellm.proxy.gateway.mcp.outbound_credentials.httpx_auth import (
        NoOpAuth,
        StaticHeaderAuth,
    )
    from litellm.proxy.gateway.mcp.result import Error, Ok

    url, token = protected_server_url

    rejected = await UpstreamConnection(url, auth=NoOpAuth()).list_tools()
    assert isinstance(rejected, Error)
    assert rejected.error.tag == "unauthorized"

    authed = await UpstreamConnection(
        url, auth=StaticHeaderAuth(f"Bearer {token}")
    ).list_tools()
    assert isinstance(authed, Ok)
    assert any(t.name == "echo" for t in authed.ok)


@pytest.mark.asyncio
async def test_upstream_connection_prompts_and_resources(echo_server_url):
    from litellm.proxy._experimental.mcp_server.v2_egress import UpstreamConnection
    from litellm.proxy.gateway.mcp.outbound_credentials.httpx_auth import NoOpAuth
    from litellm.proxy.gateway.mcp.result import Ok

    conn = UpstreamConnection(echo_server_url, auth=NoOpAuth())

    prompts = await conn.list_prompts()
    assert isinstance(prompts, Ok)
    assert any(p.name == "greet" for p in prompts.ok)

    got = await conn.get_prompt("greet", {"name": "Ada"})
    assert isinstance(got, Ok)
    assert got.ok.messages  # the rendered prompt has at least one message

    resources = await conn.list_resources()
    assert isinstance(resources, Ok)
    target = next(r for r in resources.ok if str(r.uri) == "data://info")

    read = await conn.read_resource(target.uri)
    assert isinstance(read, Ok)
    assert read.ok.contents  # at least one content block


@pytest.mark.asyncio
async def test_upstream_connection_stdio(tmp_path):
    from litellm.proxy._experimental.mcp_server.v2_egress import UpstreamConnection
    from litellm.proxy._types import MCPTransport
    from litellm.proxy.gateway.mcp.result import Ok

    script = tmp_path / "stdio_server.py"
    script.write_text(
        "from mcp.server.fastmcp import FastMCP\n"
        "mcp = FastMCP('stdio-test')\n"
        "@mcp.tool()\n"
        "def ping() -> str:\n"
        "    return 'pong'\n"
        "mcp.run(transport='stdio')\n"
    )
    conn = UpstreamConnection(
        transport=MCPTransport.stdio, command=sys.executable, args=[str(script)]
    )
    tools = await conn.list_tools()
    assert isinstance(tools, Ok)
    assert any(t.name == "ping" for t in tools.ok)


@pytest.mark.asyncio
async def test_upstream_connection_sse(sse_server_url):
    from litellm.proxy._experimental.mcp_server.v2_egress import UpstreamConnection
    from litellm.proxy._types import MCPTransport
    from litellm.proxy.gateway.mcp.outbound_credentials.httpx_auth import NoOpAuth
    from litellm.proxy.gateway.mcp.result import Ok

    conn = UpstreamConnection(
        sse_server_url, transport=MCPTransport.sse, auth=NoOpAuth()
    )
    tools = await conn.list_tools()
    assert isinstance(tools, Ok)
    assert any(t.name == "echo" for t in tools.ok)
