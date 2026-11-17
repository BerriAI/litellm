import asyncio
import json
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import anyio
import httpx2
import pytest
from mcp import ClientSession, MCPError
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import Context
from mcp.server.mcpserver import MCPServer as UpstreamServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import (
    CONNECTION_CLOSED,
    CallToolRequestParams,
    CallToolResult,
    DiscoverResult,
    JSONRPCMessage,
    JSONRPCRequest,
    LoggingMessageNotificationParams,
    ServerCapabilities,
)
from pydantic import TypeAdapter, ValidationError
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Add the parent directory to the path so we can import litellm
import litellm.mcp_client.client as mcp_client_module
from litellm.mcp_client.client import (
    MCPClient,
    PersistentMCPSession,
    UpstreamSessionClosedError,
    _TransportContext,
    as_mcp_read_timeout,
)
from litellm.types.mcp import MCPTransport
from tests._support.clock_loops import AutojumpClockLoop, ManualClockLoop

_JSONRPC_MESSAGE_ADAPTER: Final = TypeAdapter(JSONRPCMessage)


class _MockTransportClient(MCPClient):
    """An MCPClient whose streamable-HTTP transport runs on an httpx2 MockTransport."""

    def __init__(
        self,
        respond,
        *,
        http_transport: httpx2.AsyncBaseTransport | None = None,
        transport_context: Callable[[str, httpx2.AsyncClient], _TransportContext] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._respond = respond
        self._http_transport = http_transport
        self._transport_context = transport_context

    def _create_transport_context(self) -> tuple[_TransportContext, httpx2.AsyncClient]:
        transport: Final = self._http_transport or httpx2.MockTransport(self._respond)
        http_client: Final = self._create_httpx_client_factory(transport=transport)(
            headers=self._get_auth_headers(), timeout=httpx2.Timeout(self.timeout)
        )
        if self._transport_context is not None:
            return self._transport_context(self.server_url, http_client), http_client
        return streamable_http_client(self.server_url, http_client=http_client), http_client


# ---------------------------------------------------------------------------
# _last_initialize_instructions capture
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Transport error surfacing
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__])


@pytest.mark.parametrize("module", ["mcp", "mcp_types", "httpx2", "httpcore2"])
def test_base_sdk_guard_rejects_mcp_dependencies(tmp_path: Path, module: str) -> None:
    import subprocess
    import sys

    (tmp_path / f"{module}.py").write_text("")
    checker = Path(__file__).parents[2] / "base_sdk_tests" / "check_base_sdk_install.py"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import runpy, sys; sys.path.insert(0, sys.argv[2]); "
            "runpy.run_path(sys.argv[1])['check_environment_is_base_only']()",
            str(checker),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, f"base-only guard accepted installed {module}"
    assert f"{module} installed" in result.stderr


class _DiagnosticSSEStream(httpx2.AsyncByteStream):
    def __init__(self, messages: asyncio.Queue[bytes | Exception | None]) -> None:
        self.messages = messages

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"event: endpoint\ndata: /messages\n\n"
        while True:
            message: Final = await self.messages.get()
            if message is None:
                return
            if isinstance(message, Exception):
                raise message
            yield b"event: message\ndata: " + message + b"\n\n"


_DIAGNOSTIC_STDIO_SERVER: Final = """
import json, sys
mode, failure_method = sys.argv[1:]
for line in sys.stdin:
    request = json.loads(line)
    if "method" not in request or "id" not in request:
        continue
    if request["method"] == failure_method:
        if mode == "bad-json":
            print("secret-invalid-json", flush=True)
            continue
        if mode == "closed":
            sys.exit(0)
        if mode == "silent":
            print(json.dumps({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "Waiting"}}), flush=True)
            continue
    if request["method"] == "initialize":
        result = {"protocolVersion": request["params"]["protocolVersion"], "capabilities": {"tools": {}, "logging": {}}, "serverInfo": {"name": "diagnostic", "version": "1"}}
    elif request["method"] == "tools/list":
        print(json.dumps({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "Listing tools"}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "id": "unmatched", "result": {}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "id": "server-ping", "method": "ping"}), flush=True)
        result = {"tools": [{"name": "ping", "inputSchema": {"type": "object"}}]}
    else:
        result = {"content": [{"type": "text", "text": "pong"}], "isError": False}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""


def _diagnostic_transport(transport: MCPTransport, mode: str, failure_method: str) -> _TransportContext:
    from mcp import StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client

    if transport == MCPTransport.stdio:
        return stdio_client(
            StdioServerParameters(
                command=sys.executable, args=["-u", "-c", _DIAGNOSTIC_STDIO_SERVER, mode, failure_method]
            )
        )
    messages: Final[asyncio.Queue[bytes | Exception | None]] = asyncio.Queue()

    async def respond(request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET":
            return httpx2.Response(
                200, headers={"Content-Type": "text/event-stream"}, stream=_DiagnosticSSEStream(messages)
            )
        payload: Final = json.loads(request.content)
        if "method" not in payload or "id" not in payload:
            return httpx2.Response(202)
        if payload["method"] == failure_method and mode != "ok":
            if mode == "bad-json":
                await messages.put(b"secret-invalid-json")
            elif mode == "io-error":
                await messages.put(httpx2.ReadError("secret-read-error"))
            elif mode == "closed":
                await messages.put(None)
            elif mode == "silent":
                await messages.put(
                    b'{"jsonrpc":"2.0","method":"notifications/message","params":{"level":"info","data":"Waiting"}}'
                )
            return httpx2.Response(202)
        if payload["method"] == "tools/list":
            for message in (
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/message",
                    "params": {"level": "info", "data": "Listing tools"},
                },
                {"jsonrpc": "2.0", "id": "unmatched", "result": {}},
                {"jsonrpc": "2.0", "id": "server-ping", "method": "ping"},
            ):
                await messages.put(json.dumps(message).encode())
        result: Final = (
            {
                "protocolVersion": payload["params"]["protocolVersion"],
                "capabilities": {"tools": {}, "logging": {}},
                "serverInfo": {"name": "diagnostic", "version": "1"},
            }
            if payload["method"] == "initialize"
            else {"tools": [{"name": "ping", "inputSchema": {"type": "object"}}]}
            if payload["method"] == "tools/list"
            else {"content": [{"type": "text", "text": "pong"}], "isError": False}
        )
        await messages.put(json.dumps({"jsonrpc": "2.0", "id": payload["id"], "result": result}).encode())
        return httpx2.Response(202)

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx2.Timeout | None = None,
        auth: httpx2.Auth | None = None,
    ) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(respond), headers=headers, timeout=timeout, auth=auth)

    return sse_client("https://example.com/sse", httpx_client_factory=factory)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", [MCPTransport.sse, MCPTransport.stdio])
@pytest.mark.parametrize("failure_method", ["initialize", "tools/list"])
async def test_transport_parsing_failure_is_preserved(transport: MCPTransport, failure_method: str) -> None:
    client: Final = MCPClient(server_url="https://example.com/sse", transport_type=transport, timeout=0.2)
    with pytest.raises(ValidationError):
        await asyncio.wait_for(
            client._execute_session_operation(
                _diagnostic_transport(transport, "bad-json", failure_method), lambda session: session.list_tools()
            ),
            timeout=3,
        )


def test_sse_read_failure_is_preserved() -> None:
    loop: Final = AutojumpClockLoop()

    async def run() -> None:
        client: Final = MCPClient(server_url="https://example.com/sse", transport_type=MCPTransport.sse, timeout=0.2)
        with pytest.raises(httpx2.ReadError, match="secret-read-error"):
            await asyncio.wait_for(
                client._execute_session_operation(
                    _diagnostic_transport(MCPTransport.sse, "io-error", "tools/list"),
                    lambda session: session.list_tools(),
                ),
                timeout=3,
            )

    try:
        loop.run_until_complete(run())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol_version", ["auto", "2025-06-18"])
@pytest.mark.parametrize("transport", [MCPTransport.sse, MCPTransport.stdio])
@pytest.mark.parametrize("mode", ["ok", "closed", "silent"])
async def test_transport_completion_and_normal_messages(
    transport: MCPTransport, mode: str, protocol_version: str
) -> None:
    from mcp import ClientSession

    from litellm.proxy.mcp_server.rest_endpoints import _connection_error_message

    logging_callback: Final = AsyncMock()
    read_timeout: Final = 0.2 if mode == "silent" else 30
    client: Final = MCPClient(
        server_url="https://example.com/sse",
        transport_type=transport,
        timeout=read_timeout,
        logging_callback=logging_callback,
        protocol_version=protocol_version,
    )

    async def operation(session: ClientSession) -> CallToolResult:
        tools: Final = await session.list_tools()
        assert [tool.name for tool in tools.tools] == ["ping"]
        return await session.call_tool("ping", {})

    pending: Final = client._execute_session_operation(_diagnostic_transport(transport, mode, "tools/list"), operation)
    if mode == "ok":
        result: Final = await asyncio.wait_for(pending, timeout=3)
        assert result.is_error is False
        assert result.content[0].text == "pong"
        logging_callback.assert_awaited_once_with(LoggingMessageNotificationParams(level="info", data="Listing tools"))
    else:
        with pytest.raises(MCPError) as caught:
            await asyncio.wait_for(pending, timeout=3)
        if mode == "closed":
            assert "connection was closed" in _connection_error_message(caught.value, client.server_url, read_timeout)
        else:
            assert isinstance(as_mcp_read_timeout(caught.value), TimeoutError)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", [MCPTransport.sse, MCPTransport.stdio])
async def test_transport_cancellation_cleans_up_a_pending_request(transport: MCPTransport) -> None:
    ready: Final = asyncio.Event()

    async def on_log(message: LoggingMessageNotificationParams) -> None:
        if message.data == "Waiting":
            ready.set()

    client: Final = MCPClient(
        server_url="https://example.com/sse", transport_type=transport, timeout=30, logging_callback=on_log
    )
    task: Final = asyncio.create_task(
        client._execute_session_operation(
            _diagnostic_transport(transport, "silent", "tools/list"), lambda session: session.list_tools()
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=3)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.parametrize("method", ("prompts/list", "resources/list", "resources/templates/list"))
@pytest.mark.parametrize(
    "failure", ("repeat", "cycle", "cap", "method_not_found", "internal_error", "unauthorized", "deadline")
)
@pytest.mark.parametrize("strict", (False, True))
def test_optional_discovery_rejects_incomplete_walks(
    method: str, failure: str, strict: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        mcp_client_module,
        "MCP_TOOL_LISTING_MAX_PAGES",
        3 if failure in ("cycle", "repeat") else 2,
        raising=False,
    )
    monkeypatch.setattr(mcp_client_module, "MCP_TOOL_LISTING_TIMEOUT", 0.05)
    field: Final = {
        "prompts/list": "prompts",
        "resources/list": "resources",
        "resources/templates/list": "resourceTemplates",
    }[method]
    entry: Final = {
        "prompts/list": {"name": "first"},
        "resources/list": {"name": "first", "uri": "test://first"},
        "resources/templates/list": {"name": "first", "uriTemplate": "test://{name}"},
    }[method]
    loop: Final = ManualClockLoop()

    async def run() -> None:
        cancelled: Final = asyncio.Event()

        async def respond(request: httpx2.Request) -> httpx2.Response:
            payload: Final = _JSONRPC_MESSAGE_ADAPTER.validate_json(request.content)
            if not isinstance(payload, JSONRPCRequest):
                return httpx2.Response(202)
            if payload.method == "initialize":
                return httpx2.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": payload.id,
                        "result": {
                            "protocolVersion": (payload.params or {})["protocolVersion"],
                            "capabilities": {"prompts": {}, "resources": {}},
                            "serverInfo": {"name": "interrupted", "version": "1"},
                        },
                    },
                )
            assert payload.method == method
            cursor: Final = (payload.params or {}).get("cursor")
            if cursor is not None:
                if failure == "deadline":
                    loop.advance(0.15)
                    try:
                        for _ in range(1_000):
                            await asyncio.sleep(0)
                    except asyncio.CancelledError:
                        cancelled.set()
                        raise
                    return httpx2.Response(500)
                if failure == "unauthorized":
                    return httpx2.Response(401)
                if failure in ("method_not_found", "internal_error"):
                    return httpx2.Response(
                        200,
                        json={
                            "jsonrpc": "2.0",
                            "id": payload.id,
                            "error": {
                                "code": -32601 if failure == "method_not_found" else -32603,
                                "message": "Later page unavailable",
                            },
                        },
                    )
            if failure == "deadline" and cursor is None:
                loop.advance(0.1)
            next_cursor: Final = (
                "private-cursor-2" if cursor == "private-cursor-1" and failure != "repeat" else "private-cursor-1"
            )
            return httpx2.Response(
                200, json={"jsonrpc": "2.0", "id": payload.id, "result": {field: [entry], "nextCursor": next_cursor}}
            )

        responder: Final = AsyncMock(side_effect=respond)
        client: Final = _MockTransportClient(responder, server_url="https://example.com/mcp", timeout=0.2)
        operation: Final = {
            "prompts/list": client.list_prompts,
            "resources/list": client.list_resources,
            "resources/templates/list": client.list_resource_templates,
        }[method]
        if strict:
            error_type: Final = {
                "internal_error": MCPError,
                "unauthorized": httpx2.HTTPStatusError,
                "deadline": TimeoutError,
            }.get(failure, RuntimeError)
            with pytest.raises(error_type):
                await operation(raise_on_error=True)
        else:
            assert await operation() == []
        assert len(
            tuple(
                payload
                for call in responder.call_args_list
                if isinstance(payload := _JSONRPC_MESSAGE_ADAPTER.validate_json(call.args[0].content), JSONRPCRequest)
                and payload.method == method
            )
        ) == (3 if failure == "cycle" else 2)
        assert "private-cursor" not in caplog.text
        if failure == "deadline":
            assert cancelled.is_set()

    try:
        loop.run_until_complete(run())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()


def test_client_import_before_proxy_credentials_succeeds_in_fresh_process():
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import litellm.mcp_client.client; from litellm.proxy.mcp_server.mcp_server_manager import MCPServerManager; print(MCPServerManager.__name__)",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "MCPServerManager"


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol_version", ("auto", "2025-06-18"))
@pytest.mark.parametrize("cancel_mode", ("scope", "task", "wait_for", "read_timeout"))
@pytest.mark.parametrize("concurrency", (1, 5))
@pytest.mark.parametrize("termination", ("ok", "hang", "hang_body"))
@pytest.mark.parametrize("raise_on_error", (False, True))
async def test_cancellation_delivers_termination_over_tcp(
    cancel_mode: str, concurrency: int, termination: str, raise_on_error: bool, protocol_version: str
) -> None:
    started: Final = asyncio.Event()
    scope_ready: Final[asyncio.Future[anyio.CancelScope]] = asyncio.get_running_loop().create_future()
    terminations: Final[list[bytes]] = []
    starts: Final[list[bytes]] = []
    stop: Final = asyncio.Event()
    connections: Final[list[asyncio.Task[None]]] = []

    async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connection: Final = asyncio.current_task()
        assert connection is not None
        connections.append(connection)
        try:
            request_line: Final = await reader.readline()
            if not request_line:
                return
            method: Final = request_line.split()[0]
            headers: Final = await reader.readuntil(b"\r\n\r\n")
            length: Final = next(
                (
                    int(line.split(b":", 1)[1])
                    for line in headers.splitlines()
                    if line.lower().startswith(b"content-length:")
                ),
                0,
            )
            try:
                body: Final = await reader.readexactly(length)
            except asyncio.IncompleteReadError:
                return
            if method == b"DELETE":
                terminations.append(body)
                if termination != "ok":
                    await stop.wait()
                    return
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            elif method == b"GET":
                writer.write(b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            else:
                payload: Final = json.loads(body)
                if payload["method"] == "tools/call":
                    if termination == "hang_body":
                        writer.write(
                            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                            b"Content-Length: 200\r\nConnection: close\r\n\r\n"
                        )
                        await writer.drain()
                    starts.append(body)
                    if len(starts) == concurrency:
                        started.set()
                    await stop.wait()
                    return
                if payload["method"] == "initialize":
                    if cancel_mode != "read_timeout":
                        await asyncio.sleep(0.75)
                    response: Final = json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": payload["id"],
                            "result": {
                                "protocolVersion": "2025-06-18",
                                "capabilities": {"tools": {}},
                                "serverInfo": {"name": "tcp-peer", "version": "1"},
                            },
                        }
                    ).encode()
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nMcp-Session-Id: tcp-session\r\n"
                        + f"Content-Length: {len(response)}\r\nConnection: close\r\n\r\n".encode()
                        + response
                    )
                else:
                    writer.write(b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    listener: Final = await asyncio.start_server(handle_connection, "127.0.0.1", 0)
    port: Final = listener.sockets[0].getsockname()[1]
    client_timeout: Final = 2 if cancel_mode == "read_timeout" else 30
    client: Final = MCPClient(
        server_url=f"http://127.0.0.1:{port}/mcp",
        protocol_version=protocol_version,
        timeout=client_timeout,
    )

    async def calls():
        results: Final = await asyncio.gather(
            *(
                client.call_tool(CallToolRequestParams(name="slow", arguments={}), raise_on_error=raise_on_error)
                for _ in range(concurrency)
            ),
            return_exceptions=cancel_mode == "read_timeout",
        )
        if cancel_mode == "read_timeout":
            if raise_on_error:
                assert all(isinstance(result, TimeoutError) for result in results)
            else:
                assert all(isinstance(result, CallToolResult) and result.is_error for result in results)
        return results

    async def invoke():
        if cancel_mode == "scope":
            with anyio.fail_after(None) as scope:
                scope_ready.set_result(scope)
                return await calls()
        return await calls()

    try:
        task: Final = asyncio.create_task(invoke())
        await asyncio.wait_for(started.wait(), 30)
        if cancel_mode == "scope":
            (await scope_ready).deadline = anyio.current_time() + 0.2
        if cancel_mode == "task":
            task.cancel()
        expected_error: Final = (
            TimeoutError
            if cancel_mode == "read_timeout"
            else asyncio.CancelledError
            if cancel_mode == "task"
            else TimeoutError
        )
        if cancel_mode == "read_timeout":
            done, _ = await asyncio.wait((task,), timeout=8)
            assert task in done, "Read timeout and bounded cleanup must complete without external cancellation"
            await task
        elif cancel_mode == "wait_for":
            with pytest.raises(expected_error):
                await asyncio.wait_for(task, 0.2)
        else:
            with pytest.raises(expected_error):
                await task
        assert len(starts) == concurrency
        assert len(terminations) == concurrency, "Each cancelled call must send DELETE over a fresh TCP connection"
    finally:
        stop.set()
        if not task.done():
            task.cancel()
            await asyncio.wait((task,), timeout=8)
        listener.close()
        for connection in connections:
            connection.cancel()
        closed: Final = await asyncio.wait_for(asyncio.gather(*connections, return_exceptions=True), 2)
        assert all(result is None or isinstance(result, asyncio.CancelledError) for result in closed), closed
        await asyncio.wait_for(listener.wait_closed(), 2)


def test_modern_call_uses_discovery_deadline_for_multiple_pages() -> None:
    from mcp.types import ToolsCapability

    loop: Final = AutojumpClockLoop()

    async def respond(request: httpx2.Request) -> httpx2.Response:
        payload: Final = _JSONRPC_MESSAGE_ADAPTER.validate_json(request.content)
        assert isinstance(payload, JSONRPCRequest)
        if payload.method == "server/discover":
            return httpx2.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload.id,
                    "result": DiscoverResult(
                        supported_versions=["2026-07-28"], capabilities=ServerCapabilities(tools=ToolsCapability())
                    ).model_dump(by_alias=True, exclude_none=True),
                },
            )
        if payload.method == "tools/list":
            await asyncio.sleep(0.06)
            assert payload.params is not None
            page: Final = (
                {"tools": [], "nextCursor": "second"}
                if "cursor" not in payload.params
                else {
                    "tools": [
                        {
                            "name": "quote",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"workspace": {"type": "string", "x-mcp-header": "Workspace"}},
                            },
                        }
                    ]
                }
            )
            return httpx2.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload.id,
                    "result": {"resultType": "complete", "cacheScope": "private", "ttlMs": 0, **page},
                },
            )
        assert payload.method == "tools/call"
        assert request.headers["mcp-param-workspace"] == "engineering"
        return httpx2.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": payload.id,
                "result": {"resultType": "complete", "content": [{"type": "text", "text": "quoted"}], "isError": False},
            },
        )

    async def run() -> None:
        client: Final = _MockTransportClient(
            respond, server_url="https://example.com/mcp", protocol_version="2026-07-28", timeout=0.1
        )
        assert [tool.name for tool in await client.list_tools(raise_on_error=True)] == ["quote"]
        result: Final = await client.call_tool(
            CallToolRequestParams(name="quote", arguments={"workspace": "engineering"}), raise_on_error=True
        )
        assert result.is_error is False
        assert result.content[0].text == "quoted"

    try:
        loop.run_until_complete(run())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


class _StatefulUpstreamClient(MCPClient):
    """An MCPClient whose streamable-HTTP transport talks to an in-process stateful MCP server."""

    def __init__(self, app: ASGIApp, *, server_url: str, transport_type: MCPTransport) -> None:
        super().__init__(server_url=server_url, transport_type=transport_type)
        self._app = app

    def _create_transport_context(self) -> tuple[_TransportContext, httpx2.AsyncClient]:
        http_client: Final = self._create_httpx_client_factory(transport=httpx2.ASGITransport(app=self._app))(
            headers=self._get_auth_headers(), timeout=httpx2.Timeout(self.timeout)
        )
        return streamable_http_client(self.server_url, http_client=http_client), http_client


class _CleanupHoldingASGITransport(httpx2.ASGITransport):
    def __init__(self, app: ASGIApp, cleanup_started: asyncio.Event, release_cleanup: asyncio.Event) -> None:
        super().__init__(app=app)
        self._cleanup_started: Final = cleanup_started
        self._release_cleanup: Final = release_cleanup

    async def aclose(self) -> None:
        self._cleanup_started.set()
        await self._release_cleanup.wait()
        await super().aclose()


class _CleanupHoldingStatefulUpstreamClient(_StatefulUpstreamClient):
    def __init__(
        self,
        app: ASGIApp,
        *,
        server_url: str,
        transport_type: MCPTransport,
        cleanup_started: asyncio.Event,
        release_cleanup: asyncio.Event,
    ) -> None:
        super().__init__(app, server_url=server_url, transport_type=transport_type)
        self._cleanup_started: Final = cleanup_started
        self._release_cleanup: Final = release_cleanup

    def _create_transport_context(self) -> tuple[_TransportContext, httpx2.AsyncClient]:
        transport: Final = _CleanupHoldingASGITransport(
            self._app,
            self._cleanup_started,
            self._release_cleanup,
        )
        http_client: Final = self._create_httpx_client_factory(transport=transport)(
            headers=self._get_auth_headers(),
            timeout=httpx2.Timeout(self.timeout),
        )
        return streamable_http_client(self.server_url, http_client=http_client), http_client


def _stateful_upstream() -> Starlette:
    service: Final = UpstreamServer("stateful")
    selected: Final[dict[str, str]] = {}

    @service.tool()
    def select_project(name: str, ctx: Context) -> str:
        selected[ctx.headers["mcp-session-id"]] = name
        return f"selected {name}"

    @service.tool()
    def create_feature(title: str, ctx: Context) -> str:
        return f"{selected[ctx.headers['mcp-session-id']]}/{title}"

    return service.streamable_http_app(
        stateless_http=False,
        json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )


def _upstream_with_status_for_method(method: str, status_code: int) -> Starlette:
    service: Final = UpstreamServer("status upstream")

    @service.tool()
    def protected_tool() -> str:
        return "protected"

    app: Final = service.streamable_http_app(
        stateless_http=False,
        json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    async def reject_method(request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == "POST":
            payload: Final = await request.json()
            if isinstance(payload, dict) and payload.get("method") == method:
                return Response(status_code=status_code, headers={"WWW-Authenticate": "Bearer"})
        return await call_next(request)

    app.add_middleware(BaseHTTPMiddleware, dispatch=reject_method)
    return app


def _upstream_with_malformed_response_for_method(method: str) -> Starlette:
    service: Final = UpstreamServer("malformed response upstream")

    @service.tool()
    def malformed_tool() -> str:
        return "unreachable"

    app: Final = service.streamable_http_app(
        stateless_http=False,
        json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    async def malformed_method(request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == "GET":
            return Response(content="data: {}\n\n", media_type="text/event-stream")
        if request.method == "POST":
            payload: Final = await request.json()
            if isinstance(payload, dict) and payload.get("method") == method:
                return Response(content="not a JSON-RPC message", media_type="text/event-stream")
        return await call_next(request)

    app.add_middleware(BaseHTTPMiddleware, dispatch=malformed_method)
    return app


def _client_with_session(
    app: Starlette,
    admission: Callable[[], AbstractAsyncContextManager[None]] | None = None,
) -> tuple[_StatefulUpstreamClient, PersistentMCPSession]:
    client: Final = _StatefulUpstreamClient(app, server_url="http://upstream/mcp", transport_type=MCPTransport.http)
    return client, client.open_persistent_session(admission)


@pytest.mark.asyncio
async def test_cancelling_one_waiter_for_session_readiness_does_not_cancel_other_waiters() -> None:
    app: Final = _stateful_upstream()
    async with app.router.lifespan_context(app):
        initialize_started: Final = asyncio.Event()
        release_initialize: Final = asyncio.Event()

        async def gated_server(scope: Scope, receive: Receive, send: Send) -> None:
            async def gated_send(message: Message) -> None:
                if message["type"] == "http.response.body" and not initialize_started.is_set():
                    initialize_started.set()
                    await release_initialize.wait()
                await send(message)

            await app(scope, receive, gated_send)

        gated_app: Final[ASGIApp] = gated_server
        client: Final = _StatefulUpstreamClient(
            gated_app, server_url="http://upstream/mcp", transport_type=MCPTransport.http
        )
        session: Final = client.open_persistent_session()

        async def operation(_: ClientSession) -> str:
            return "ready"

        cancelled_waiter: Final = asyncio.create_task(session.run(operation))
        remaining_waiter: Final = asyncio.create_task(session.run(operation))
        try:
            await asyncio.wait_for(initialize_started.wait(), 5)
            await asyncio.sleep(0)
            cancelled_waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled_waiter
            release_initialize.set()
            assert await asyncio.wait_for(remaining_waiter, 5) == "ready"
            assert not session.closed
        finally:
            release_initialize.set()
            session.close()
            await asyncio.wait_for(session.wait_closed(), 5)


@pytest.mark.asyncio
async def test_persistent_session_survives_recoverable_mcp_errors_and_closes_on_connection_closed() -> None:
    app: Final = _stateful_upstream()
    async with app.router.lifespan_context(app):
        client, session = _client_with_session(app)

        async def invalid_operation(_: ClientSession) -> str:
            raise MCPError(code=-32602, message="invalid params")

        try:
            with pytest.raises(MCPError):
                await session.run(invalid_operation)
            assert not session.closed
            selected: Final = await client.call_tool(
                CallToolRequestParams(name="select_project", arguments={"name": "recoverable"}),
                persistent_session=session,
            )
            created: Final = await client.call_tool(
                CallToolRequestParams(name="create_feature", arguments={"title": "still-open"}),
                persistent_session=session,
            )
        finally:
            session.close()
        assert selected.is_error is False
        assert created.content[0].text == "recoverable/still-open"
        await asyncio.wait_for(session.wait_closed(), 5)

    connection_app: Final = _stateful_upstream()
    async with connection_app.router.lifespan_context(connection_app):
        _, session = _client_with_session(connection_app)
        started: Final = asyncio.Event()
        release: Final = asyncio.Event()

        async def disconnected_operation(_: ClientSession) -> str:
            started.set()
            await release.wait()
            raise MCPError(code=CONNECTION_CLOSED, message="upstream connection closed")

        async def pending_operation(_: ClientSession) -> str:
            return "unused"

        active: Final = asyncio.create_task(session.run(disconnected_operation))
        await asyncio.wait_for(started.wait(), 5)
        pending: Final = asyncio.create_task(session.run(pending_operation))
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(MCPError) as caught:
            await asyncio.wait_for(active, 5)
        assert caught.value.error.code == CONNECTION_CLOSED
        with pytest.raises((MCPError, UpstreamSessionClosedError)):
            await asyncio.wait_for(pending, 5)
        await asyncio.wait_for(session.wait_closed(), 5)
        assert session.closed


@pytest.mark.asyncio
async def test_ended_persistent_session_is_not_reusable_during_transport_cleanup() -> None:
    app: Final = _stateful_upstream()
    async with app.router.lifespan_context(app):
        cleanup_started: Final = asyncio.Event()
        release_cleanup: Final = asyncio.Event()
        client: Final = _CleanupHoldingStatefulUpstreamClient(
            app,
            server_url="http://upstream/mcp",
            transport_type=MCPTransport.http,
            cleanup_started=cleanup_started,
            release_cleanup=release_cleanup,
        )
        session: Final = client.open_persistent_session()

        async def disconnected_operation(_: ClientSession) -> str:
            raise MCPError(code=CONNECTION_CLOSED, message="upstream connection closed")

        try:
            with pytest.raises(MCPError) as caught:
                await asyncio.wait_for(session.run(disconnected_operation), 5)
            assert caught.value.error.code == CONNECTION_CLOSED
            await asyncio.wait_for(cleanup_started.wait(), 5)

            assert not session.reusable
            assert not session.closed
            session.retire()

            release_cleanup.set()
            await asyncio.wait_for(session.wait_closed(), 5)
            assert session.closed
        finally:
            release_cleanup.set()
            if not session.closed:
                session.close()
            await asyncio.wait_for(session.wait_closed(), 5)


@pytest.mark.asyncio
async def test_persistent_session_remaps_stream_error_over_mcp_error() -> None:
    app: Final = _stateful_upstream()
    async with app.router.lifespan_context(app):
        client: Final = _StatefulUpstreamClient(
            app,
            server_url="http://upstream/mcp",
            transport_type=MCPTransport.http,
        )
        session: Final = client.open_persistent_session()
        request: Final = httpx2.Request("POST", "http://upstream/mcp")
        response: Final = httpx2.Response(401, request=request, headers={"WWW-Authenticate": "Bearer"})
        http_error: Final = httpx2.HTTPStatusError("401 Unauthorized", request=request, response=response)

        async def operation(_: ClientSession) -> str:
            stream_error: Final[asyncio.Future[Exception] | None] = session._stream_error
            assert stream_error is not None
            stream_error.set_result(http_error)
            raise MCPError(code=-32603, message="response stream closed")

        try:
            with pytest.raises(httpx2.HTTPStatusError) as caught:
                await session.run(operation)
            assert caught.value is http_error
            assert not session.reusable
        finally:
            session.close()
            await asyncio.wait_for(session.wait_closed(), 5)


@pytest.mark.asyncio
async def test_persistent_session_ends_when_a_malformed_response_is_remapped() -> None:
    app: Final = _upstream_with_malformed_response_for_method("tools/call")
    async with app.router.lifespan_context(app):
        client: Final = _StatefulUpstreamClient(
            app,
            server_url="http://upstream/mcp",
            transport_type=MCPTransport.http,
        )
        session: Final = client.open_persistent_session()
        try:

            async def ready(_: ClientSession) -> str:
                return "ready"

            assert await asyncio.wait_for(session.run(ready), 5) == "ready"
            stream_error: Final[asyncio.Future[Exception] | None] = session._stream_error
            assert stream_error is not None
            recorded_stream_error: Final = await asyncio.wait_for(asyncio.shield(stream_error), 5)
            assert isinstance(recorded_stream_error, ValidationError)

            with pytest.raises(ValidationError) as caught:
                await client.call_tool(
                    CallToolRequestParams(name="malformed_tool", arguments={}),
                    raise_on_error=True,
                    persistent_session=session,
                )
            assert caught.value is recorded_stream_error
            assert not session.reusable
            await asyncio.wait_for(session.wait_closed(), 5)
            assert session.closed
        finally:
            if not session.closed:
                session.close()
                await asyncio.wait_for(session.wait_closed(), 5)


@pytest.mark.asyncio
async def test_failed_persistent_initialize_is_not_reusable_during_transport_cleanup() -> None:
    app: Final = _upstream_with_status_for_method("initialize", 500)
    async with app.router.lifespan_context(app):
        cleanup_started: Final = asyncio.Event()
        release_cleanup: Final = asyncio.Event()
        client: Final = _CleanupHoldingStatefulUpstreamClient(
            app,
            server_url="http://upstream/mcp",
            transport_type=MCPTransport.http,
            cleanup_started=cleanup_started,
            release_cleanup=release_cleanup,
        )
        session: Final = client.open_persistent_session()
        pending: Final = asyncio.create_task(session.run(lambda _: "ready"))
        try:
            await asyncio.wait_for(cleanup_started.wait(), 5)
            assert not session.reusable
            assert not session.closed
            release_cleanup.set()
            with pytest.raises(httpx2.HTTPStatusError):
                await asyncio.wait_for(pending, 5)
            await asyncio.wait_for(session.wait_closed(), 5)
            assert session.closed
        finally:
            release_cleanup.set()
            if not session.closed:
                session.close()
            await asyncio.wait_for(session.wait_closed(), 5)


@pytest.mark.asyncio
async def test_cancelled_queued_operation_does_not_run_and_session_accepts_a_later_operation() -> None:
    app: Final = _stateful_upstream()
    async with app.router.lifespan_context(app):
        _, session = _client_with_session(app)
        first_started: Final = asyncio.Event()
        release_first: Final = asyncio.Event()
        cancelled_operation_ran: Final = asyncio.Event()

        async def first_operation(_: ClientSession) -> str:
            first_started.set()
            await release_first.wait()
            return "first"

        async def cancelled_operation(_: ClientSession) -> str:
            cancelled_operation_ran.set()
            return "cancelled"

        async def third_operation(_: ClientSession) -> str:
            return "third"

        first_task: Final = asyncio.create_task(session.run(first_operation))
        try:
            await asyncio.wait_for(first_started.wait(), 5)
            second_task: Final = asyncio.create_task(session.run(cancelled_operation))
            await asyncio.sleep(0)
            assert session._queue.qsize() == 1
            second_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await second_task
            third_task: Final = asyncio.create_task(session.run(third_operation))
            await asyncio.sleep(0)
            assert session._queue.qsize() == 2
            release_first.set()
            assert await asyncio.wait_for(first_task, 5) == "first"
            assert await asyncio.wait_for(third_task, 5) == "third"
            assert not cancelled_operation_ran.is_set()
            assert not session.closed
        finally:
            release_first.set()
            session.close()
            await asyncio.wait_for(session.wait_closed(), 5)


@pytest.mark.asyncio
async def test_retiring_persistent_session_drains_in_flight_and_queued_operations() -> None:
    app: Final = _stateful_upstream()
    async with app.router.lifespan_context(app):
        _, session = _client_with_session(app)
        first_started: Final = asyncio.Event()
        release_first: Final = asyncio.Event()

        async def first_operation(_: ClientSession) -> str:
            first_started.set()
            await release_first.wait()
            return "first"

        async def queued_operation(_: ClientSession) -> str:
            return "queued"

        first_waiter: Final = asyncio.create_task(session.run(first_operation))
        try:
            await asyncio.wait_for(first_started.wait(), 5)
            queued_waiter: Final = asyncio.create_task(session.run(queued_operation))
            await asyncio.sleep(0)
            assert session._queue.qsize() == 1
            session.retire()
            release_first.set()

            results: Final = await asyncio.wait_for(asyncio.gather(first_waiter, queued_waiter), 5)
            assert results == ["first", "queued"]
            await asyncio.wait_for(session.wait_closed(), 5)
            assert session.closed
        finally:
            release_first.set()
            if not session.closed:
                session.close()
            await asyncio.wait_for(session.wait_closed(), 5)


@pytest.mark.asyncio
async def test_timed_out_persistent_operation_waiting_for_admission_is_skipped() -> None:
    app: Final = _stateful_upstream()
    async with app.router.lifespan_context(app):
        admission_slots: Final = asyncio.Semaphore(1)
        admission_requested: Final = asyncio.Event()

        def admission() -> AbstractAsyncContextManager[None]:
            admission_requested.set()
            return admission_slots

        _, session = _client_with_session(app, admission)
        operation_ran: Final = asyncio.Event()

        async def warmup(_: ClientSession) -> str:
            return "ready"

        async def timed_out_operation(_: ClientSession) -> str:
            operation_ran.set()
            return "expired"

        async def later_operation(_: ClientSession) -> str:
            return "later"

        try:
            assert await session.run(warmup) == "ready"
            admission_requested.clear()
            async with admission_slots:
                timed_out_call: Final = asyncio.create_task(session.run(timed_out_operation))
                await asyncio.wait_for(admission_requested.wait(), 1)
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(timed_out_call, 0.05)
                assert not operation_ran.is_set()

            later_result: Final = await asyncio.wait_for(session.run(later_operation), 1)

            assert later_result == "later"
            assert session.reusable
            assert not operation_ran.is_set()
        finally:
            session.close()
            await asyncio.wait_for(session.wait_closed(), 5)


@pytest.mark.asyncio
async def test_persistent_session_holds_admission_until_initialize_completes() -> None:
    app: Final = _stateful_upstream()
    async with app.router.lifespan_context(app):
        initialize_seen: Final = asyncio.Event()
        admission_slots: Final = asyncio.Semaphore(1)

        async def observing_app(scope: Scope, receive: Receive, send: Send) -> None:
            async def observe_send(message: Message) -> None:
                if message["type"] == "http.response.body":
                    initialize_seen.set()
                await send(message)

            await app(scope, receive, observe_send)

        client: Final = _StatefulUpstreamClient(
            observing_app,
            server_url="http://upstream/mcp",
            transport_type=MCPTransport.http,
        )
        session: Final = client.open_persistent_session(lambda: admission_slots)

        async def operation(_: ClientSession) -> str:
            return "ready"

        run_task: Final = asyncio.create_task(session.run(operation))
        try:
            async with admission_slots:
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(initialize_seen.wait(), 0.05)
                assert not session._ready.done()
                assert not run_task.done()

            result: Final = await asyncio.wait_for(run_task, 5)
            assert result == "ready"
            assert initialize_seen.is_set()
        finally:
            session.close()
            await asyncio.wait_for(session.wait_closed(), 5)
            await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_persistent_session_releases_admission_when_connect_fails() -> None:
    async def failing_app(_: Scope, __: Receive, ___: Send) -> None:
        raise httpx2.ConnectError("connect failed")

    client: Final = _StatefulUpstreamClient(
        failing_app,
        server_url="http://upstream/mcp",
        transport_type=MCPTransport.http,
    )
    admission_slots: Final = asyncio.Semaphore(1)
    session: Final = client.open_persistent_session(lambda: admission_slots)

    async def operation(_: ClientSession) -> str:
        return "unreachable"

    try:
        with pytest.raises(httpx2.ConnectError, match="connect failed"):
            await asyncio.wait_for(session.run(operation), 1)
        await asyncio.wait_for(session.wait_closed(), 1)

        await asyncio.wait_for(admission_slots.acquire(), 0.1)
        admission_slots.release()
    finally:
        session.close()
        await asyncio.wait_for(session.wait_closed(), 5)


@pytest.mark.asyncio
async def test_failed_connect_is_not_reusable_during_admission_teardown() -> None:
    connect_failed: Final = asyncio.Event()
    admission_exit_started: Final = asyncio.Event()
    release_admission_exit: Final = asyncio.Event()

    async def failing_app(_: Scope, __: Receive, ___: Send) -> None:
        connect_failed.set()
        raise httpx2.ConnectError("connect failed")

    @asynccontextmanager
    async def blocking_admission() -> AsyncIterator[None]:
        try:
            yield
        finally:
            admission_exit_started.set()
            await release_admission_exit.wait()

    client: Final = _StatefulUpstreamClient(
        failing_app,
        server_url="http://upstream/mcp",
        transport_type=MCPTransport.http,
    )
    session: Final = client.open_persistent_session(blocking_admission)

    async def operation(_: ClientSession) -> str:
        return "unreachable"

    run_task: Final = asyncio.create_task(session.run(operation))
    try:
        await asyncio.wait_for(connect_failed.wait(), 1)
        await asyncio.wait_for(admission_exit_started.wait(), 1)

        assert not run_task.done()
        assert not session.reusable
        assert not session.closed

        release_admission_exit.set()
        with pytest.raises(httpx2.ConnectError, match="connect failed"):
            await asyncio.wait_for(run_task, 1)
        await asyncio.wait_for(session.wait_closed(), 1)
        assert session.closed
    finally:
        release_admission_exit.set()
        session.close()
        await asyncio.wait_for(session.wait_closed(), 5)
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_persistent_session_admission_is_acquired_only_when_operation_runs() -> None:
    app: Final = _stateful_upstream()
    async with app.router.lifespan_context(app):
        admission_slots: Final = asyncio.Semaphore(2)
        _, session = _client_with_session(app, lambda: admission_slots)
        first_started: Final = asyncio.Event()
        release_first: Final = asyncio.Event()
        operation_order: Final[list[str]] = []

        async def first_operation(_: ClientSession) -> str:
            operation_order.append("first")
            first_started.set()
            await release_first.wait()
            return "first"

        async def second_operation(_: ClientSession) -> str:
            operation_order.append("second")
            return "second"

        async def third_operation(_: ClientSession) -> str:
            operation_order.append("third")
            return "third"

        first_task: Final = asyncio.create_task(session.run(first_operation))
        second_task: Final = asyncio.create_task(session.run(second_operation))
        third_task: Final = asyncio.create_task(session.run(third_operation))
        try:
            await asyncio.wait_for(first_started.wait(), 5)
            await asyncio.sleep(0)
            assert session._queue.qsize() == 2
            await asyncio.wait_for(admission_slots.acquire(), 0.1)
            try:
                assert admission_slots.locked()
            finally:
                admission_slots.release()
            assert operation_order == ["first"]

            release_first.set()
            results: Final = await asyncio.wait_for(
                asyncio.gather(first_task, second_task, third_task),
                5,
            )

            assert results == ["first", "second", "third"]
            assert operation_order == ["first", "second", "third"]
        finally:
            release_first.set()
            session.close()
            await asyncio.wait_for(session.wait_closed(), 5)
            await asyncio.gather(first_task, second_task, third_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_retiring_idle_persistent_session_closes_and_rejects_new_operations() -> None:
    app: Final = _stateful_upstream()
    async with app.router.lifespan_context(app):
        _, session = _client_with_session(app)

        async def operation(_: ClientSession) -> str:
            return "ready"

        try:
            assert await session.run(operation) == "ready"
            session.retire()
            await asyncio.wait_for(session.wait_closed(), 1)

            assert session.closed
            with pytest.raises(UpstreamSessionClosedError):
                await session.run(operation)
        finally:
            if not session.closed:
                session.close()
            await asyncio.wait_for(session.wait_closed(), 5)


@pytest.mark.asyncio
async def test_closing_persistent_session_mid_operation_fails_the_waiter_instead_of_hanging() -> None:
    app: Final = _stateful_upstream()
    async with app.router.lifespan_context(app):
        _, session = _client_with_session(app)
        started: Final = asyncio.Event()

        async def slow_operation(_: object) -> str:
            started.set()
            await asyncio.sleep(30)
            return "never"

        waiter: Final = asyncio.ensure_future(session.run(slow_operation))
        await asyncio.wait_for(started.wait(), 5)
        session.close()
        with pytest.raises(RuntimeError, match="upstream MCP session closed"):
            await asyncio.wait_for(waiter, 5)
        await asyncio.wait_for(session.wait_closed(), 5)


@pytest.mark.asyncio
async def test_persistent_session_reports_an_upstream_cancellation_as_a_runtime_error_not_a_cancelled_caller() -> None:
    app: Final = _stateful_upstream()
    async with app.router.lifespan_context(app):
        client, session = _client_with_session(app)

        async def cancelled_upstream(_: object) -> str:
            raise asyncio.CancelledError()

        try:
            with pytest.raises(RuntimeError, match="cancelled"):
                await asyncio.wait_for(session.run(cancelled_upstream), 5)
            selected: Final = await client.call_tool(
                CallToolRequestParams(name="select_project", arguments={"name": "e"}), persistent_session=session
            )
        finally:
            session.close()
        assert selected.is_error is False, "an upstream cancellation must not be mistaken for a cancelled caller"
        await asyncio.wait_for(session.wait_closed(), 5)


@pytest.mark.asyncio
async def test_closing_persistent_session_fails_a_caller_blocked_on_a_full_queue_instead_of_hanging() -> None:
    from litellm.mcp_client.client import _MAX_PENDING_OPERATIONS

    app: Final = _stateful_upstream()
    async with app.router.lifespan_context(app):
        _, session = _client_with_session(app)
        started: Final = asyncio.Event()

        async def slow_operation(_: object) -> str:
            started.set()
            await asyncio.sleep(30)
            return "never"

        waiters: Final = tuple(
            asyncio.ensure_future(session.run(slow_operation)) for _ in range(_MAX_PENDING_OPERATIONS + 2)
        )
        await asyncio.wait_for(started.wait(), 5)
        await asyncio.sleep(0)
        session.close()
        outcomes: Final = await asyncio.wait_for(asyncio.gather(*waiters, return_exceptions=True), 5)
        assert all(isinstance(outcome, RuntimeError) for outcome in outcomes), outcomes
        await asyncio.wait_for(session.wait_closed(), 5)


@pytest.mark.asyncio
async def test_persistent_session_ends_after_a_broken_stream_so_the_next_call_gets_a_fresh_one() -> None:
    app: Final = _stateful_upstream()
    async with app.router.lifespan_context(app):
        _, session = _client_with_session(app)

        async def broken_stream(_: object) -> str:
            raise anyio.BrokenResourceError()

        with pytest.raises(anyio.BrokenResourceError):
            await asyncio.wait_for(session.run(broken_stream), 5)
        await asyncio.wait_for(session.wait_closed(), 5)
        assert session.closed, "a dead transport must end the session instead of being reused"

@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_callback", [False, True])
async def test_tool_progress_preserves_message_and_legacy_callback(legacy_callback: bool) -> None:
    received = AsyncMock()

    async def legacy(progress: float, total: float | None) -> None:
        await received(progress, total)

    async def modern(progress: float, total: float | None, message: str | None) -> None:
        await received(progress, total, message)

    async def call_tool(**kwargs):
        await kwargs["progress_callback"](1.0, 2.0, "halfway")
        return CallToolResult(content=[])

    session = AsyncMock(spec=ClientSession, call_tool=AsyncMock(side_effect=call_tool))

    async def run(operation, **kwargs):
        return await operation(session)

    persistent = MagicMock(spec=PersistentMCPSession, run=AsyncMock(side_effect=run))
    client = MCPClient(server_url="https://example.com/mcp")
    result = await client.call_tool(
        CallToolRequestParams(name="slow"),
        host_progress_callback=legacy if legacy_callback else modern,
        persistent_session=persistent,
    )
    assert not result.is_error
    if legacy_callback:
        received.assert_awaited_once_with(1.0, 2.0)
    else:
        received.assert_awaited_once_with(1.0, 2.0, "halfway")


@pytest.mark.asyncio
async def test_cancelling_persistent_operation_stops_work_and_keeps_session_usable() -> None:
    app: Final = _stateful_upstream()
    async with app.router.lifespan_context(app):
        client, session = _client_with_session(app)
        started: Final = asyncio.Event()
        stopped: Final = asyncio.Event()

        async def slow_operation(_: ClientSession) -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        waiter: Final = asyncio.create_task(session.run(slow_operation))
        try:
            await asyncio.wait_for(started.wait(), 5)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            await asyncio.sleep(0.1)
            assert stopped.is_set(), "Caller cancellation must stop the running upstream operation"
            result: Final = await asyncio.wait_for(
                client.call_tool(
                    CallToolRequestParams(name="select_project", arguments={"name": "after-cancellation"}),
                    persistent_session=session,
                ),
                5,
            )
            assert not result.is_error
        finally:
            session.close()
            await asyncio.wait_for(session.wait_closed(), 5)
            await asyncio.gather(waiter, return_exceptions=True)
