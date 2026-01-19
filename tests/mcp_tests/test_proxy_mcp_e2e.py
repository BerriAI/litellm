import asyncio
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from litellm.proxy.proxy_server import (
    app as proxy_app,
    cleanup_router_config_variables,
    initialize,
)


CONFIG_TEMPLATE_PATH = Path("tests/mcp_tests/test_configs/test_config_mcp_e2e.yaml")
PROXY_START_TIMEOUT = 30
MCP_HEADERS = {
    "Authorization": "Bearer sk-1234",
    "x-mcp-servers": "math_stdio",
}


def _initialize_proxy(config_path: str) -> None:
    cleanup_router_config_variables()
    asyncio.run(initialize(config=config_path, debug=True))


def _start_proxy_server(config_path: str) -> tuple[str, uvicorn.Server, threading.Thread, socket.socket]:
    _initialize_proxy(config_path)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()

    config = uvicorn.Config(proxy_app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve(sockets=[sock]))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    start_time = time.time()
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("Proxy server failed to start")
        if time.time() - start_time > PROXY_START_TIMEOUT:
            raise TimeoutError("Proxy server did not start in time")
        time.sleep(0.05)

    return f"http://{host}:{port}", server, thread, sock


@pytest.fixture(scope="session")
def proxy_server_url(tmp_path_factory: pytest.TempPathFactory):
    config_dir = tmp_path_factory.mktemp("mcp_e2e")
    config_path = config_dir / "config.yaml"
    config_path.write_text(CONFIG_TEMPLATE_PATH.read_text())

    server_url, server, thread, sock = _start_proxy_server(str(config_path))

    yield server_url

    server.should_exit = True
    thread.join(timeout=10)
    sock.close()


@pytest.mark.asyncio
async def test_proxy_mcp_stdio_roundtrip(proxy_server_url: str) -> None:
    async with asyncio.timeout(20):
        async with streamablehttp_client(
            url=f"{proxy_server_url}/mcp", headers=MCP_HEADERS
        ) as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                assert any(tool.name.endswith("add") for tool in tools_result.tools)

                result = await session.call_tool(
                    "add", arguments={"a": 3, "b": 4}
                )
                assert result.content
                first_content = result.content[0]
                text = getattr(first_content, "text", None)
                assert text == "7"
