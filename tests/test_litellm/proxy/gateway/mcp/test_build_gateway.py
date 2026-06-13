from mcp.shared.memory import create_connected_server_and_client_session
from starlette.applications import Starlette
from starlette.routing import Route

from litellm.proxy.gateway.mcp.app import build_gateway, build_server
from litellm.proxy.gateway.mcp.foundation import build_test_deps


def test_build_gateway_returns_starlette():
    app = build_gateway(build_test_deps())
    assert isinstance(app, Starlette)


def test_build_gateway_yields_independent_apps():
    a = build_gateway(build_test_deps())
    b = build_gateway(build_test_deps())
    assert a is not b


def test_mcp_route_is_mounted():
    app = build_gateway(build_test_deps())
    paths = {r.path for r in app.routes if isinstance(r, Route)}
    assert "/mcp" in paths


async def test_initialize_then_list_tools_returns_empty():
    server = build_server(build_test_deps())
    async with create_connected_server_and_client_session(server) as client:
        init = await client.initialize()
        assert init.serverInfo.name == "litellm-mcp-gateway"
        listed = await client.list_tools()
        assert listed.tools == []
