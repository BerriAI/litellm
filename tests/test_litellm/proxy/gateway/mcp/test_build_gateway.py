import httpx
from mcp.shared.memory import create_connected_server_and_client_session
from starlette.applications import Starlette
from starlette.routing import Mount

from litellm.proxy.gateway.mcp.app import build_gateway, build_server
from litellm.proxy.gateway.mcp.foundation import build_test_deps


def test_build_gateway_returns_starlette():
    app = build_gateway(build_test_deps())
    assert isinstance(app, Starlette)


def test_build_gateway_yields_independent_apps():
    a = build_gateway(build_test_deps())
    b = build_gateway(build_test_deps())
    assert a is not b


def test_mcp_endpoint_is_mounted_as_asgi_app():
    app = build_gateway(build_test_deps())
    mounts = {r.path for r in app.routes if isinstance(r, Mount)}
    assert "/mcp" in mounts


async def test_initialize_then_list_tools_returns_empty():
    server = build_server(build_test_deps())
    async with create_connected_server_and_client_session(server) as client:
        init = await client.initialize()
        assert init.serverInfo.name == "litellm-mcp-gateway"
        listed = await client.list_tools()
        assert listed.tools == []


async def test_initialize_handshake_over_real_asgi_transport():
    app = build_gateway(build_test_deps())
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0"},
        },
    }
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", follow_redirects=True) as client:
            resp = await client.post(
                "/mcp",
                headers={
                    "content-type": "application/json",
                    "accept": "application/json, text/event-stream",
                },
                json=request,
            )
    assert resp.status_code == 200
    assert "litellm-mcp-gateway" in resp.text
