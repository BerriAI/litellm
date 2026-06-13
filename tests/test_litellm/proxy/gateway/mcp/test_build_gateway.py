import socket
import threading
import time

import httpx
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.memory import create_connected_server_and_client_session
from starlette.applications import Starlette
from starlette.routing import Mount

from litellm.proxy.gateway.mcp.app import build_gateway, build_server
from litellm.proxy.gateway.mcp.foundation import build_test_deps


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _LiveServer:
    def __init__(self, app: Starlette) -> None:
        self.port = _free_port()
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> str:
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("uvicorn did not start in time")
            time.sleep(0.05)
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)


def test_build_gateway_yields_independent_apps():
    a = build_gateway(build_test_deps())
    b = build_gateway(build_test_deps())
    assert a is not b


def test_gateway_mounts_a_catch_all_asgi_app():
    app = build_gateway(build_test_deps())
    mounts = {r.path for r in app.routes if isinstance(r, Mount)}
    assert "" in mounts


async def test_server_handshake_and_empty_catalog_in_memory():
    server = build_server(build_test_deps())
    async with create_connected_server_and_client_session(server) as client:
        init = await client.initialize()
        assert init.serverInfo.name == "litellm-mcp-gateway"
        listed = await client.list_tools()
        assert listed.tools == []


async def test_s0_end_to_end_over_a_real_uvicorn_socket():
    app = build_gateway(build_test_deps())
    with _LiveServer(app) as base_url:
        async with streamablehttp_client(f"{base_url}/mcp") as (read, write, _):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                assert init.serverInfo.name == "litellm-mcp-gateway"
                assert init.instructions

                listed = await session.list_tools()
                assert listed.tools == []

                result = await session.call_tool("anything", {})
                assert result.isError is True
                assert "not wired in S0" in result.content[0].text

        async with httpx.AsyncClient(follow_redirects=False) as raw:
            resp = await raw.post(
                f"{base_url}/mcp",
                headers={
                    "content-type": "application/json",
                    "accept": "application/json, text/event-stream",
                },
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            )
        assert resp.status_code == 200
