import asyncio
import socket
import subprocess
import sys
import threading
import time
import typing
from pathlib import Path

import httpx
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
from tests.mcp_tests.mcp_server import (
    DEFAULT_API_KEY,
    DEFAULT_AUTHORIZATION_VALUE,
    DEFAULT_BEARER_TOKEN,
    DEFAULT_CLIENT_ID,
    DEFAULT_CLIENT_SECRET,
    DEFAULT_CUSTOM_HEADER,
    DEFAULT_CUSTOM_HEADER_VALUE,
)

CONFIG_TEMPLATE_PATH = Path("tests/mcp_tests/test_configs/test_oauth2_mcp_config.yaml")
MCP_SERVER_SCRIPT = Path("tests/mcp_tests/mcp_server.py")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROXY_START_TIMEOUT = 30
PROXY_AUTHORIZATION_HEADER = "Bearer sk-1234"

# Each entry: proxy server name -> how to launch the upstream test MCP server.
SERVER_SPECS: dict[str, dict[str, typing.Optional[str]]] = {
    "math_no_auth": {"auth_mode": "none", "auth_secret": None},
    "math_api_key": {"auth_mode": "api_key", "auth_secret": DEFAULT_API_KEY},
    "math_bearer_token": {
        "auth_mode": "bearer_token",
        "auth_secret": DEFAULT_BEARER_TOKEN,
    },
    "math_authorization": {
        "auth_mode": "authorization",
        "auth_secret": DEFAULT_AUTHORIZATION_VALUE,
    },
    "math_custom_header": {
        "auth_mode": "custom_header",
        "auth_secret": DEFAULT_CUSTOM_HEADER_VALUE,
    },
    "test_oauth2_server": {"auth_mode": "oauth2", "auth_secret": None},
    "test_pkce_server": {"auth_mode": "oauth2", "auth_secret": None},
    "internal_mcp_server": {"auth_mode": "oauth2", "auth_secret": None},
}


def _initialize_proxy(config_path: str) -> None:
    cleanup_router_config_variables()
    asyncio.run(initialize(config=config_path, debug=True))


def _start_proxy_server(
    config_path: str,
) -> tuple[str, uvicorn.Server, threading.Thread, socket.socket]:
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


def _reserve_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _start_mcp_server_process(*, auth_mode: str, port: int, auth_secret: typing.Optional[str]) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(MCP_SERVER_SCRIPT),
        "--transport",
        "http",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--auth-mode",
        auth_mode,
        "--client-id",
        DEFAULT_CLIENT_ID,
        "--client-secret",
        DEFAULT_CLIENT_SECRET,
    ]
    if auth_secret is not None:
        cmd.extend(["--auth-secret", auth_secret])

    process = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    start_time = time.time()
    while True:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"MCP server exited early (auth_mode={auth_mode}).\n"
                f"STDOUT: {stdout.decode()}\nSTDERR: {stderr.decode()}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            if time.time() - start_time > PROXY_START_TIMEOUT:
                process.terminate()
                raise TimeoutError(f"MCP server did not start in time (auth_mode={auth_mode})")
            time.sleep(0.05)

    return process


@pytest.fixture(scope="session", autouse=True)
def _clear_proxy_database_env() -> typing.Iterator[None]:
    mp = pytest.MonkeyPatch()
    mp.delenv("DATABASE_URL", raising=False)
    mp.setenv("LITELLM_MASTER_KEY", "sk-1234")
    try:
        yield
    finally:
        mp.undo()


@pytest.fixture(scope="session")
def mcp_auth_servers() -> typing.Iterator[dict[str, typing.Any]]:
    servers = {name: {**spec, "port": _reserve_port()} for name, spec in SERVER_SPECS.items()}

    processes: list[subprocess.Popen] = []
    try:
        for spec in servers.values():
            process = _start_mcp_server_process(
                auth_mode=spec["auth_mode"],
                port=spec["port"],
                auth_secret=spec["auth_secret"],
            )
            spec["process"] = process
            spec["base_url"] = f"http://127.0.0.1:{spec['port']}"
            processes.append(process)
        yield servers
    finally:
        for process in processes:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


@pytest.fixture(scope="session")
def proxy_server_url(
    tmp_path_factory: pytest.TempPathFactory, mcp_auth_servers: dict[str, typing.Any]
) -> typing.Iterator[str]:
    config = yaml.safe_load(CONFIG_TEMPLATE_PATH.read_text())

    for server_name, spec in mcp_auth_servers.items():
        server_config = config["mcp_servers"][server_name]
        base_url = spec["base_url"]
        server_config["url"] = f"{base_url}/mcp"
        for endpoint_key in ("token_url", "authorization_url", "token_exchange_endpoint"):
            if endpoint_key in server_config:
                suffix = "authorize" if "authorization" in endpoint_key else "token"
                server_config[endpoint_key] = f"{base_url}/oauth/{suffix}"

    config_path = tmp_path_factory.mktemp("mcp_auth_e2e") / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    server_url, server, thread, sock = _start_proxy_server(str(config_path))
    yield server_url

    server.should_exit = True
    thread.join(timeout=10)
    sock.close()


async def _call_add_tool(
    *,
    proxy_server_url: str,
    server_name: str,
    a: int,
    b: int,
    headers: typing.Optional[dict[str, str]] = None,
) -> typing.Optional[str]:
    request_headers = {
        "Authorization": PROXY_AUTHORIZATION_HEADER,
        "x-mcp-servers": server_name,
    }
    if headers:
        request_headers.update(headers)

    async with asyncio.timeout(20):
        async with streamablehttp_client(url=f"{proxy_server_url}/mcp", headers=request_headers) as (
            read,
            write,
            _get_session_id,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                assert any(tool.name.endswith("add") for tool in tools_result.tools)

                result = await session.call_tool("add", arguments={"a": a, "b": b})
                assert result.content
                return getattr(result.content[0], "text", None)


async def _list_tool_names(
    *,
    proxy_server_url: str,
    server_name: str,
    headers: typing.Optional[dict[str, str]] = None,
) -> list[str]:
    request_headers = {
        "Authorization": PROXY_AUTHORIZATION_HEADER,
        "x-mcp-servers": server_name,
    }
    if headers:
        request_headers.update(headers)

    async with asyncio.timeout(20):
        async with streamablehttp_client(url=f"{proxy_server_url}/mcp", headers=request_headers) as (
            read,
            write,
            _get_session_id,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                return [tool.name for tool in tools_result.tools]


class TestProxyMcpAuthE2E:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("server_name", "a", "b", "expected"),
        [
            ("math_no_auth", 3, 4, "7"),
            ("math_api_key", 5, 6, "11"),
            ("math_bearer_token", 7, 8, "15"),
            ("math_authorization", 1, 2, "3"),
            ("test_oauth2_server", 9, 10, "19"),
        ],
    )
    async def test_proxy_forwards_configured_credential(self, proxy_server_url, server_name, a, b, expected) -> None:
        result = await _call_add_tool(proxy_server_url=proxy_server_url, server_name=server_name, a=a, b=b)
        assert result == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "server_name",
        ["math_api_key", "math_bearer_token", "math_authorization", "test_oauth2_server"],
    )
    async def test_upstream_rejects_unauthenticated_request(self, mcp_auth_servers, server_name) -> None:
        base_url = mcp_auth_servers[server_name]["base_url"]
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_oauth_m2m_ignores_caller_authorization(self, proxy_server_url) -> None:
        """M2M servers must never forward the caller's Authorization; the proxy
        fetches its own client_credentials token. A bogus caller token must not
        break the call (proves the proxy substitutes its own upstream token)."""
        result = await _call_add_tool(
            proxy_server_url=proxy_server_url,
            server_name="test_oauth2_server",
            a=2,
            b=3,
            headers={
                "x-litellm-api-key": "Bearer sk-1234",
                "Authorization": "Bearer caller-supplied-bogus-token",
            },
        )
        assert result == "5"

    @pytest.mark.asyncio
    async def test_custom_header_passthrough(self, proxy_server_url) -> None:
        result = await _call_add_tool(
            proxy_server_url=proxy_server_url,
            server_name="math_custom_header",
            a=4,
            b=5,
            headers={f"x-mcp-math_custom_header-{DEFAULT_CUSTOM_HEADER}": DEFAULT_CUSTOM_HEADER_VALUE},
        )
        assert result == "9"

    @pytest.mark.asyncio
    async def test_custom_header_required_for_discovery(self, proxy_server_url) -> None:
        """Without the per-server custom header the upstream rejects the request,
        so its tools must not be discoverable through the proxy."""
        tool_names = await _list_tool_names(proxy_server_url=proxy_server_url, server_name="math_custom_header")
        assert not any(name.endswith("add") for name in tool_names)

    @pytest.mark.asyncio
    async def test_obo_token_exchange(self, proxy_server_url) -> None:
        """OBO: the proxy exchanges the caller's bearer (subject_token) for a
        scoped token and uses it upstream. Per the MCP OBO docs, tools/list and
        tools/call must both work with the user token in Authorization."""
        result = await _call_add_tool(
            proxy_server_url=proxy_server_url,
            server_name="internal_mcp_server",
            a=6,
            b=7,
            headers={
                "x-litellm-api-key": "Bearer sk-1234",
                "Authorization": "Bearer user-subject-jwt",
            },
        )
        assert result == "13"
