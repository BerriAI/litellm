import asyncio
import json
import os
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
import typing
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
import pytest
import uvicorn
import yaml
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult
from starlette.requests import Request

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._experimental.mcp_server.tool_search import handle_mcp_proxy_tool
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, ProxyException, UserAPIKeyAuth
from litellm.proxy.proxy_server import (
    app as proxy_app,
)
from litellm.proxy.proxy_server import (
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
    # The FastAPI lifespan event (proxy_startup_event) re-reads master_key from
    # the LITELLM_MASTER_KEY env var, overriding whatever initialize() set from
    # the config file. We must set it here so the lifespan doesn't reset it to None.
    mp.setenv("LITELLM_MASTER_KEY", "sk-1234")
    try:
        yield
    finally:
        mp.undo()


async def _initialize_proxy(config_path: str) -> None:
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import global_mcp_server_manager

    cleanup_router_config_variables()
    await initialize(config=config_path, debug=True)
    for server_id, upstream in tuple(global_mcp_server_manager.registry.items()):
        if upstream.server_name != "math_restricted":
            continue
        global_mcp_server_manager.registry[server_id] = upstream.model_copy(
            update={"tool_name_to_display_name": {"add": "Add Numbers"}}
        )


@dataclass(frozen=True)
class ProxyRig:
    url: str
    config_path: str
    loop: asyncio.AbstractEventLoop


def _start_proxy_server(
    config_path: str,
) -> tuple[ProxyRig, uvicorn.Server, threading.Thread, socket.socket]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()

    config = uvicorn.Config(proxy_app, host=host, port=port, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)

    loop = asyncio.new_event_loop()

    async def _serve() -> None:
        from litellm.proxy._experimental.mcp_server import server as mcp_server

        await _initialize_proxy(config_path)
        async with proxy_app.router.lifespan_context(proxy_app), mcp_server.lifespan(proxy_app):
            await server.serve(sockets=[sock])

    def _run() -> None:
        with asyncio.Runner(loop_factory=lambda: loop) as runner:
            runner.run(_serve())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    start_time = time.time()
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("Proxy server failed to start")
        if time.time() - start_time > PROXY_START_TIMEOUT:
            raise TimeoutError("Proxy server did not start in time")
        time.sleep(0.05)

    return ProxyRig(f"http://{host}:{port}", config_path, loop), server, thread, sock


@contextmanager
def _math_http_server(offset: int) -> typing.Iterator[str]:
    host = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        _, port = sock.getsockname()

    with tempfile.TemporaryFile() as server_log:
        process = subprocess.Popen(
            [sys.executable, str(MCP_SERVER_SCRIPT), "--transport", "http", "--host", host, "--port", str(port)],
            cwd=str(PROJECT_ROOT),
            stdout=server_log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "MCP_ADD_OFFSET": str(offset)},
        )
        try:
            start_time = time.monotonic()
            while True:
                if process.poll() is not None:
                    server_log.seek(0)
                    raise RuntimeError(f"MCP upstream exited early: {server_log.read().decode()}")
                try:
                    with socket.create_connection((host, port), timeout=0.1):
                        break
                except OSError:
                    if time.monotonic() - start_time > PROXY_START_TIMEOUT:
                        raise TimeoutError("Streamable HTTP MCP server did not start in time")
                    time.sleep(0.05)
            yield f"http://{host}:{port}"
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.fixture(scope="session")
def math_streamable_http_server() -> typing.Iterator[str]:
    with _math_http_server(100) as url:
        yield url


@pytest.fixture(scope="session")
def math_restricted_server() -> typing.Iterator[str]:
    with _math_http_server(200) as url:
        yield url


@pytest.fixture(scope="session")
def _proxy_server(
    tmp_path_factory: pytest.TempPathFactory,
    math_streamable_http_server: str,
    math_restricted_server: str,
):
    config_dir = tmp_path_factory.mktemp("mcp_e2e")
    config_path = config_dir / "config.yaml"
    config = yaml.safe_load(CONFIG_TEMPLATE_PATH.read_text())
    config["mcp_servers"]["math_stdio"]["command"] = sys.executable
    config["mcp_servers"]["math_streamable_http"]["url"] = f"{math_streamable_http_server}/mcp"
    config["mcp_servers"]["math_restricted"]["url"] = f"{math_restricted_server}/mcp"
    config["general_settings"]["custom_auth"] = f"{__name__}.authorize_proxy_key"
    config["litellm_settings"]["callbacks"] = [f"{__name__}.proxy_call_recorder"]
    config["mcp_servers"]["math_restricted"]["mcp_info"] = {"mcp_server_cost_info": {"default_cost_per_query": 0.25}}
    config_path.write_text(yaml.safe_dump(config))

    rig, server, thread, sock = _start_proxy_server(str(config_path))

    try:
        yield rig
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        assert not thread.is_alive(), "Proxy did not shut down"


@pytest.fixture
def proxy_server_url(_proxy_server: ProxyRig, setup_and_teardown: None) -> str:
    asyncio.run_coroutine_threadsafe(_initialize_proxy(_proxy_server.config_path), _proxy_server.loop).result(
        timeout=30
    )
    return _proxy_server.url


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

                    result = await session.call_tool("add", arguments={"a": 3, "b": 4})
                    assert result.content
                    first_content = result.content[0]
                    text = getattr(first_content, "text", None)
                    assert text == "7"

    @pytest.mark.asyncio
    async def test_proxy_mcp_streamable_http_roundtrip(self, proxy_server_url: str) -> None:
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

                    result = await session.call_tool("add", arguments={"a": 5, "b": 6})
                    assert result.content
                    first_content = result.content[0]
                    text = getattr(first_content, "text", None)
                    assert text == "111"

    @pytest.mark.asyncio
    async def test_proxy_mcp_lists_all_servers_without_header(self, proxy_server_url: str) -> None:
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

                    async def _call_and_get_text(tool_name: str, *, a: int, b: int) -> str | None:
                        result = await session.call_tool(tool_name, arguments={"a": a, "b": b})
                        assert result.content
                        first_content = result.content[0]
                        return getattr(first_content, "text", None)

                    stdio_result = await _call_and_get_text("math_stdio-add", a=2, b=3)
                    streamable_result = await _call_and_get_text("math_streamable_http-add", a=4, b=5)
                    assert stdio_result == "5"
                    assert streamable_result == "109"


class TestProxyMcpStatelessBehavior:
    """
    Verify that the LiteLLM MCP proxy operates in stateless mode.

    When StreamableHTTPSessionManager is configured with stateless=True,
    independent clients must be able to connect, list tools, and call tools
    without sharing or inheriting session state from other clients.

    With stateless=False this fails because the server tracks sessions and
    expects clients to supply an mcp-session-id header obtained from a
    prior handshake — breaking clients that don't manage session IDs.

    Regression test for https://github.com/BerriAI/litellm/issues/20242
    """

    @pytest.mark.asyncio
    async def test_independent_clients_no_shared_session(self, proxy_server_url: str) -> None:
        """Two independent clients connect and operate without sharing session state."""
        async with asyncio.timeout(30):
            # --- Client A: connect, initialize, call tool ---
            async with streamablehttp_client(
                url=f"{proxy_server_url}/mcp",
                headers={
                    "Authorization": PROXY_AUTHORIZATION_HEADER,
                    "x-mcp-servers": "math_stdio",
                },
            ) as (read_a, write_a, _get_sid_a):
                async with ClientSession(read_a, write_a) as session_a:
                    await session_a.initialize()
                    result_a = await session_a.call_tool("add", arguments={"a": 10, "b": 20})
                    assert result_a.content
                    text_a = getattr(result_a.content[0], "text", None)
                    assert text_a == "30"

            # Allow proxy and MCP SDK to fully clean up the first connection before
            # opening the second. Without this, the SDK's TaskGroup can raise
            # ExceptionGroup when the server closes the connection (see MCP SDK #915).
            await asyncio.sleep(0.5)

            # --- Client B: completely independent connection ---
            async with streamablehttp_client(
                url=f"{proxy_server_url}/mcp",
                headers={
                    "Authorization": PROXY_AUTHORIZATION_HEADER,
                    "x-mcp-servers": "math_stdio",
                },
            ) as (read_b, write_b, _get_sid_b):
                async with ClientSession(read_b, write_b) as session_b:
                    await session_b.initialize()
                    tools = await session_b.list_tools()
                    assert any(t.name.endswith("add") for t in tools.tools)
                    result_b = await session_b.call_tool("add", arguments={"a": 100, "b": 200})
                    assert result_b.content
                    text_b = getattr(result_b.content[0], "text", None)
                    assert text_b == "300"


PROXY_MODE_TOOLS = frozenset({"search_tools", "get_tool_schema", "call_tool"})


def _payload(result: typing.Any) -> typing.Any:
    assert result.content, f"empty tool result: {result}"
    return json.loads(result.content[0].text)


def _proxy_session(proxy_server_url: str, **extra_headers: str):
    return streamablehttp_client(
        url=f"{proxy_server_url}/mcp/proxy",
        headers={"Authorization": PROXY_AUTHORIZATION_HEADER, **extra_headers},
    )


class TestProxyMcpSchemaDiscoveryMode:
    """Drive /mcp/proxy over the real streamable-HTTP transport with the MCP SDK client:
    the fixed three-tool surface, opaque-id discovery, schema-validated execution against
    two upstreams that expose the same tool name, and the operations the surface refuses."""

    @pytest.mark.asyncio
    async def test_initialize_and_list_expose_only_discovery_tools(self, proxy_server_url: str) -> None:
        async with asyncio.timeout(20):
            async with _proxy_session(proxy_server_url) as (read, write, _sid):
                async with ClientSession(read, write) as session:
                    init = await session.initialize()
                    assert init.capabilities.tools is not None
                    assert init.capabilities.prompts is None
                    assert init.capabilities.resources is None

                    listed = await session.list_tools()
                    assert {tool.name for tool in listed.tools} == PROXY_MODE_TOOLS

    @pytest.mark.asyncio
    async def test_search_schema_and_call_round_trip_keeps_server_identity(self, proxy_server_url: str) -> None:
        async with asyncio.timeout(30):
            async with _proxy_session(proxy_server_url) as (read, write, _sid):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    hits = _payload(await session.call_tool("search_tools", arguments={"query": "add"}))
                    by_name = {hit["name"]: hit for hit in hits}
                    assert {"math_stdio-add", "math_streamable_http-add"} <= set(by_name)
                    assert all("inputSchema" not in hit for hit in hits)
                    assert by_name["math_stdio-add"]["tool_id"] != by_name["math_streamable_http-add"]["tool_id"]

                    schema = _payload(
                        await session.call_tool(
                            "get_tool_schema", arguments={"tool_id": by_name["math_stdio-add"]["tool_id"]}
                        )
                    )
                    assert schema["name"] == "math_stdio-add"
                    assert set(schema["inputSchema"]["required"]) == {"a", "b"}
                    assert schema["outputSchema"]["properties"]["result"]["type"] == "integer"

                    stdio = await session.call_tool(
                        "call_tool",
                        arguments={"tool_id": by_name["math_stdio-add"]["tool_id"], "arguments": {"a": 3, "b": 4}},
                    )
                    http = await session.call_tool(
                        "call_tool",
                        arguments={
                            "tool_id": by_name["math_streamable_http-add"]["tool_id"],
                            "arguments": {"a": 5, "b": 6},
                        },
                    )
                    assert stdio.isError is False and stdio.content[0].text == "7"
                    assert http.isError is False and http.content[0].text == "111"

    @pytest.mark.asyncio
    async def test_server_scope_header_narrows_discovery(self, proxy_server_url: str) -> None:
        async with asyncio.timeout(20):
            async with _proxy_session(proxy_server_url, **{"x-mcp-servers": "math_streamable_http"}) as (
                read,
                write,
                _sid,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    hits = _payload(await session.call_tool("search_tools", arguments={"query": "add"}))
                    assert {hit["name"] for hit in hits} == {"math_streamable_http-add"}

    @pytest.mark.asyncio
    async def test_rejections_never_reach_upstream(self, proxy_server_url: str) -> None:
        from mcp.shared.exceptions import McpError
        from mcp.types import METHOD_NOT_FOUND

        async with asyncio.timeout(30):
            async with _proxy_session(proxy_server_url) as (read, write, _sid):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    hits = _payload(await session.call_tool("search_tools", arguments={"query": "add"}))
                    tool_id = next(hit["tool_id"] for hit in hits if hit["name"] == "math_stdio-add")

                    bad_args = await session.call_tool(
                        "call_tool", arguments={"tool_id": tool_id, "arguments": {"a": "three", "b": 4}}
                    )
                    assert bad_args.isError is True and "Invalid arguments" in bad_args.content[0].text

                    stale = await session.call_tool("get_tool_schema", arguments={"tool_id": "0" * 32})
                    assert stale.isError is True and "unauthorized tool_id" in stale.content[0].text

                    for not_an_object in ("wrong", False):
                        refused_args = await session.call_tool(
                            "call_tool", arguments={"tool_id": tool_id, "arguments": not_an_object}
                        )
                        assert refused_args.isError is True and "object" in refused_args.content[0].text

                    direct = await session.call_tool("math_stdio-add", arguments={"a": 1, "b": 2})
                    assert direct.isError is True and "unavailable on /mcp/proxy" in direct.content[0].text

                    for operation in (session.list_prompts, session.list_resources):
                        with pytest.raises(McpError) as refused:
                            await operation()
                        assert refused.value.error.code == METHOD_NOT_FOUND


async def authorize_proxy_key(request: Request, api_key: str) -> UserAPIKeyAuth:
    permissions = {
        "sk-1234": LiteLLM_ObjectPermissionTable(object_permission_id="open", mcp_servers=["math_stdio"]),
        "sk-restricted": LiteLLM_ObjectPermissionTable(
            object_permission_id="restricted", mcp_servers=["math_restricted"]
        ),
        "sk-none": LiteLLM_ObjectPermissionTable(object_permission_id="none", mcp_servers=["no-mcp-servers"]),
        "sk-add-only": LiteLLM_ObjectPermissionTable(
            object_permission_id="add-only", mcp_servers=["math_stdio"], mcp_tool_permissions={"math_stdio": ["add"]}
        ),
    }
    permission = permissions.get(api_key)
    if permission is None:
        raise ProxyException(message="Unknown test key", type="authentication_error", param=None, code=401)
    return UserAPIKeyAuth(api_key=api_key, user_id=api_key, object_permission=permission)


class ProxyCallRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.events: queue.Queue[str] = queue.Queue()
        self.failures: queue.Queue[str] = queue.Queue()

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        payload = kwargs.get("standard_logging_object")
        if isinstance(payload, dict) and payload.get("call_type") == "call_mcp_tool":
            self.events.put(json.dumps(payload, default=str))

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        payload = kwargs.get("standard_logging_object")
        if isinstance(payload, dict) and payload.get("call_type") == "call_mcp_tool":
            self.failures.put(json.dumps(payload, default=str))


proxy_call_recorder = ProxyCallRecorder()


@asynccontextmanager
async def _scoped_session(url: str, key: str = "sk-1234", **headers: str) -> typing.AsyncIterator[ClientSession]:
    async with asyncio.timeout(30):
        async with _proxy_session(url, Authorization=f"Bearer {key}", **headers) as (read, write, _sid):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


async def _search(session: ClientSession, query: str) -> dict[str, str]:
    result = await session.call_tool("search_tools", arguments={"query": query})
    assert result.isError is False, result
    return {hit["name"]: hit["tool_id"] for hit in _payload(result)}


async def _call(session: ClientSession, tool_id: str, a: int = 3, b: int = 4) -> CallToolResult:
    return await session.call_tool("call_tool", arguments={"tool_id": tool_id, "arguments": {"a": a, "b": b}})


def _assert_unauthorized(result: CallToolResult) -> None:
    assert result.isError is True
    assert result.content[0].text == "Unknown or unauthorized tool_id"


class TestProxyMcpAuthorizationScope:
    @pytest.mark.asyncio
    async def test_server_grant_bounds_search_and_blocks_foreign_ids(self, proxy_server_url: str) -> None:
        async with _scoped_session(proxy_server_url, "sk-restricted") as granted:
            restricted_id = (await _search(granted, "add"))["math_restricted-add"]
            assert (await _call(granted, restricted_id)).content[0].text == "207"
        async with _scoped_session(proxy_server_url) as ungranted:
            assert set(await _search(ungranted, "add")) == {"math_stdio-add", "math_streamable_http-add"}
            _assert_unauthorized(await ungranted.call_tool("get_tool_schema", {"tool_id": restricted_id}))
            _assert_unauthorized(await _call(ungranted, restricted_id))

    @pytest.mark.asyncio
    async def test_no_mcp_servers_sentinel_hides_every_tool(self, proxy_server_url: str) -> None:
        async with _scoped_session(proxy_server_url) as granted:
            tool_id = (await _search(granted, "add"))["math_stdio-add"]
        async with _scoped_session(proxy_server_url, "sk-none") as session:
            assert await _search(session, "add") == {}
            _assert_unauthorized(await session.call_tool("get_tool_schema", {"tool_id": tool_id}))
            _assert_unauthorized(await _call(session, tool_id))

    @pytest.mark.asyncio
    async def test_tool_grant_hides_ungranted_tools_and_blocks_their_ids(self, proxy_server_url: str) -> None:
        async with _scoped_session(proxy_server_url) as granted:
            multiply_id = (await _search(granted, "multiply"))["math_stdio-multiply"]
        async with _scoped_session(proxy_server_url, "sk-add-only", **{"x-mcp-servers": "math_stdio"}) as session:
            ids = await _search(session, "add multiply request_headers")
            assert set(ids) == {"math_stdio-add"}
            assert (await _call(session, ids["math_stdio-add"])).content[0].text == "7"
            _assert_unauthorized(await session.call_tool("get_tool_schema", {"tool_id": multiply_id}))
            _assert_unauthorized(await _call(session, multiply_id))

    @pytest.mark.asyncio
    async def test_same_named_tools_keep_distinct_ids_and_reach_their_own_upstream(self, proxy_server_url: str) -> None:
        async with _scoped_session(proxy_server_url, "sk-restricted") as session:
            ids = await _search(session, "add")
            assert set(ids) == {"math_stdio-add", "math_streamable_http-add", "math_restricted-add"}
            assert len(set(ids.values())) == 3
            assert all(len(tool_id) == 32 for tool_id in ids.values())
            for name, expected in (
                ("math_stdio-add", "7"),
                ("math_streamable_http-add", "107"),
                ("math_restricted-add", "207"),
            ):
                schema = _payload(await session.call_tool("get_tool_schema", {"tool_id": ids[name]}))
                assert schema["name"] == name
                assert schema["tool_id"] == ids[name]
                result = await _call(session, ids[name])
                assert result.isError is False
                assert result.content[0].text == expected

    @pytest.mark.asyncio
    async def test_server_scope_header_narrows_grants_and_blocks_out_of_scope_ids(self, proxy_server_url: str) -> None:
        async with _scoped_session(proxy_server_url, "sk-restricted") as unscoped:
            other_id = (await _search(unscoped, "add"))["math_stdio-add"]
        async with _scoped_session(
            proxy_server_url, "sk-restricted", **{"x-mcp-servers": "math_restricted"}
        ) as session:
            ids = await _search(session, "add")
            assert set(ids) == {"math_restricted-add"}
            _assert_unauthorized(await session.call_tool("get_tool_schema", {"tool_id": other_id}))
            _assert_unauthorized(await _call(session, other_id))
            assert (await _call(session, ids["math_restricted-add"])).content[0].text == "207"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key", [None, "sk-invalid"])
    async def test_missing_or_invalid_key_cannot_initialize(self, proxy_server_url: str, key: str | None) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{proxy_server_url}/mcp/proxy",
                headers={
                    "Accept": "application/json, text/event-stream",
                    **({"Authorization": f"Bearer {key}"} if key else {}),
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "auth-test", "version": "1"},
                    },
                },
            )
        assert response.status_code == 401, response.text

    @pytest.mark.asyncio
    async def test_server_headers_are_forwarded_only_to_the_named_upstream(self, proxy_server_url: str) -> None:
        for tag in ("first-request", "second-request"):
            async with _scoped_session(
                proxy_server_url,
                "sk-restricted",
                **{
                    "x-mcp-math_restricted-authorization": f"Bearer {tag}",
                    "x-mcp-math_restricted-x-request-tag": tag,
                },
            ) as session:
                ids = await _search(session, "request_headers")
                for name, expected in (
                    ("math_restricted", {"authorization": f"Bearer {tag}", "x-request-tag": tag}),
                    ("math_streamable_http", {"authorization": "", "x-request-tag": ""}),
                ):
                    result = await session.call_tool(
                        "call_tool", {"tool_id": ids[f"{name}-request_headers"], "arguments": {}}
                    )
                    assert result.isError is False
                    assert _payload(result) == expected

    @pytest.mark.asyncio
    async def test_proxy_call_emits_spend_log(self, proxy_server_url: str) -> None:
        async with _scoped_session(proxy_server_url, "sk-restricted") as session:
            tool_id = (await _search(session, "add"))["math_restricted-add"]
            result = await _call(session, tool_id, 123, 456)
            assert result.isError is False and result.content[0].text == "779"
            async with asyncio.timeout(10):
                while True:
                    payload = json.loads(await asyncio.to_thread(proxy_call_recorder.events.get, True, 5))
                    if payload.get("metadata", {}).get("mcp_tool_call_metadata", {}).get("arguments") == {
                        "a": 123,
                        "b": 456,
                    }:
                        break
            assert payload["call_type"] == "call_mcp_tool"
            assert payload["response_cost"] == 0.25
            assert payload["status"] == "success"
            assert payload["metadata"]["mcp_tool_call_metadata"]["mcp_server_name"] == "math_restricted"
            assert payload["metadata"]["mcp_tool_call_metadata"]["name"] == "add"
            assert payload["metadata"]["mcp_tool_call_metadata"]["namespaced_tool_name"] == "math_restricted/add"

    @pytest.mark.asyncio
    async def test_proxy_scope_exception_returns_iserror_and_emits_failure_log(self, proxy_server_url: str) -> None:
        async with _scoped_session(
            proxy_server_url,
            "sk-none",
            **{"x-mcp-servers": "math_restricted", "x-litellm-call-id": "proxy-scope-denial"},
        ) as session:
            result = await session.call_tool("call_tool", {"tool_id": "denied-scope", "arguments": {}})
            assert result.isError is True
            assert result.content[0].text == (
                "Error: The key is not allowed to access the requested MCP servers: math_restricted"
            )
            async with asyncio.timeout(10):
                while True:
                    payload = json.loads(await asyncio.to_thread(proxy_call_recorder.failures.get, True, 5))
                    if payload["id"] == "proxy-scope-denial":
                        break
            assert payload["call_type"] == "call_mcp_tool"
            assert payload["status"] == "failure"
            assert payload["response_cost"] == 0
            assert "math_restricted" in payload["error_str"]

    @pytest.mark.parametrize("arguments", ["wrong", False, None, [], 0])
    def test_handler_rejects_non_object_arguments(
        self, proxy_server_url: str, _proxy_server: ProxyRig, arguments: object
    ) -> None:
        async def check() -> None:
            auth = UserAPIKeyAuth(
                object_permission=LiteLLM_ObjectPermissionTable(
                    object_permission_id="validation", mcp_servers=["math_stdio"]
                )
            )
            hits = _payload(await handle_mcp_proxy_tool("search_tools", {"query": "add"}, auth))
            tool_id = next(hit["tool_id"] for hit in hits if hit["name"] == "math_stdio-add")
            result = await handle_mcp_proxy_tool("call_tool", {"tool_id": tool_id, "arguments": arguments}, auth)
            assert result.isError is True
            assert result.content[0].text == "arguments must be an object"

        asyncio.run_coroutine_threadsafe(check(), _proxy_server.loop).result(timeout=30)
