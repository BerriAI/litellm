import asyncio
import os
import socket
import subprocess
import sys
import threading
import time
import typing
from pathlib import Path

import pytest
import uvicorn
import yaml
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from litellm.proxy.proxy_server import (
    app as proxy_app,
    cleanup_router_config_variables,
    initialize,
)


CONFIG_TEMPLATE_PATH = Path("tests/mcp_tests/test_configs/test_config_mcp_e2e.yaml")
MCP_SERVER_SCRIPT = Path("tests/mcp_tests/mcp_server.py")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROXY_START_TIMEOUT = 30


PROXY_AUTHORIZATION_HEADER = "Bearer sk-1234"


@pytest.fixture(scope="session", autouse=True)
def _clear_proxy_database_env() -> typing.Iterator[None]:
    """Ensure local proxy DB settings don't leak into tests."""
    mp = pytest.MonkeyPatch()
    mp.delenv("DATABASE_URL", raising=False)
    try:
        yield
    finally:
        mp.undo()


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
def math_streamable_http_server() -> str:
    host = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        _, port = sock.getsockname()

    cmd = [
        sys.executable,
        str(MCP_SERVER_SCRIPT),
        "--transport",
        "http",
        "--host",
        host,
        "--port",
        str(port),
    ]

    env = os.environ.copy()
    server_process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    start_time = time.time()
    while True:
        if server_process.poll() is not None:
            stdout, stderr = server_process.communicate()
            raise RuntimeError(
                f"Streamable HTTP MCP server exited early.\nSTDOUT: {stdout.decode()}\nSTDERR: {stderr.decode()}"
            )
        try:
            with socket.create_connection((host, port), timeout=0.1):
                break
        except OSError:
            if time.time() - start_time > PROXY_START_TIMEOUT:
                server_process.terminate()
                raise TimeoutError("Streamable HTTP MCP server did not start in time")
            time.sleep(0.05)

    yield f"http://{host}:{port}"

    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()


@pytest.fixture(scope="session")
def proxy_server_url(
    tmp_path_factory: pytest.TempPathFactory, math_streamable_http_server: str
):
    config_dir = tmp_path_factory.mktemp("mcp_e2e")
    config_path = config_dir / "config.yaml"
    config = yaml.safe_load(CONFIG_TEMPLATE_PATH.read_text())
    config["mcp_servers"]["math_streamable_http"][
        "url"
    ] = f"{math_streamable_http_server}/mcp"
    config_path.write_text(yaml.safe_dump(config))

    server_url, server, thread, sock = _start_proxy_server(str(config_path))

    yield server_url

    server.should_exit = True
    thread.join(timeout=10)
    sock.close()


class TestProxyMcpSimpleConnections:
    @pytest.mark.asyncio
    async def test_proxy_mcp_stdio_roundtrip(self, proxy_server_url: str) -> None:
        async with asyncio.timeout(20):
            async with streamablehttp_client(
                url=f"{proxy_server_url}/mcp",
                headers={
                    "Authorization": PROXY_AUTHORIZATION_HEADER,
                    "x-mcp-servers": "math_stdio",
                },
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

    @pytest.mark.asyncio
    async def test_proxy_mcp_streamable_http_roundtrip(
        self, proxy_server_url: str
    ) -> None:
        async with asyncio.timeout(20):
            async with streamablehttp_client(
                url=f"{proxy_server_url}/mcp",
                headers={
                    "Authorization": PROXY_AUTHORIZATION_HEADER,
                    "x-mcp-servers": "math_streamable_http",
                },
            ) as (read, write, _get_session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    assert any(tool.name.endswith("add") for tool in tools_result.tools)

                    result = await session.call_tool(
                        "add", arguments={"a": 5, "b": 6}
                    )
                    assert result.content
                    first_content = result.content[0]
                    text = getattr(first_content, "text", None)
                    assert text == "11"

    @pytest.mark.asyncio
    async def test_proxy_mcp_lists_all_servers_without_header(
        self, proxy_server_url: str
    ) -> None:
        async with asyncio.timeout(20):
            async with streamablehttp_client(
                url=f"{proxy_server_url}/mcp",
                headers={"Authorization": PROXY_AUTHORIZATION_HEADER},
            ) as (read, write, _get_session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    tool_names = {tool.name for tool in tools_result.tools}
                    expected_tool_names = {
                        "math_stdio-add",
                        "math_stdio-multiply",
                        "math_streamable_http-add",
                        "math_streamable_http-multiply",
                    }
                    assert expected_tool_names <= tool_names

                    async def _call_and_get_text(
                        tool_name: str, *, a: int, b: int
                    ) -> str | None:
                        result = await session.call_tool(tool_name, arguments={"a": a, "b": b})
                        assert result.content
                        first_content = result.content[0]
                        return getattr(first_content, "text", None)

                    stdio_result = await _call_and_get_text(
                        "math_stdio-add", a=2, b=3
                    )
                    streamable_result = await _call_and_get_text(
                        "math_streamable_http-add", a=4, b=5
                    )
                    assert stdio_result == "5"
                    assert streamable_result == "9"
