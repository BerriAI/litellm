"""Tests for the v2 MCP egress transport: the flag and the UpstreamConnection."""

import contextlib
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


@contextlib.contextmanager
def _serve(app):
    """Serve an ASGI app on a free port in a background thread; yield its /mcp url."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    url = f"http://127.0.0.1:{port}/mcp"
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
