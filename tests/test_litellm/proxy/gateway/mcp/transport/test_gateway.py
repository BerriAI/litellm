import socket
import threading
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Message, Receive, Scope, Send

from litellm.proxy.gateway.mcp.app import build_gateway
from litellm.proxy.gateway.mcp.authn.context import current_subject
from litellm.proxy.gateway.mcp.authn.resolver import anonymous_subject
from litellm.proxy.gateway.mcp.foundation import Subject, build_test_deps
from litellm.proxy.gateway.mcp.transport.gateway import TransportGateway


class _RecordingManager:
    def __init__(self) -> None:
        self.subject_during_handle: Subject | None = None
        self.handled = False

    async def handle_request(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.handled = True
        self.subject_during_handle = current_subject()


class _SpyResolver:
    def __init__(self, subject: Subject) -> None:
        self.bearers: list[str | None] = []
        self._subject = subject

    def __call__(self, bearer: str | None) -> Subject:
        self.bearers.append(bearer)
        return self._subject


def _http_scope(path: str, headers: list[tuple[bytes, bytes]]) -> Scope:
    return {"type": "http", "path": path, "headers": headers}


async def _noop_receive() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


def _capturing_send() -> tuple[list[Message], Send]:
    captured: list[Message] = []

    async def send(message: Message) -> None:
        captured.append(message)

    return captured, send


async def test_subject_from_resolver_is_published_during_handle_and_reset_after():
    manager = _RecordingManager()
    distinctive = Subject(subject_id="alice", tenant="acme")
    resolver = _SpyResolver(distinctive)
    gateway = TransportGateway(manager, resolver)

    scope = _http_scope("/github/mcp", [(b"authorization", b"Bearer xyz")])
    _, send = _capturing_send()

    assert current_subject() is None
    await gateway(scope, _noop_receive, send)

    assert manager.handled is True
    assert resolver.bearers == ["xyz"]
    assert manager.subject_during_handle == distinctive
    assert current_subject() is None


async def test_only_a_bearer_token_is_extracted():
    cases: list[tuple[list[tuple[bytes, bytes]], str | None]] = [
        ([(b"authorization", b"Bearer xyz")], "xyz"),
        ([(b"authorization", b"bearer xyz")], "xyz"),
        ([(b"authorization", b"Basic dXNlcg==")], None),
        ([(b"authorization", b"Bearer ")], None),
        ([], None),
    ]
    for headers, expected in cases:
        resolver = _SpyResolver(anonymous_subject())
        gateway = TransportGateway(_RecordingManager(), resolver)
        _, send = _capturing_send()
        await gateway(_http_scope("/mcp", headers), _noop_receive, send)
        assert resolver.bearers == [expected]


async def test_non_mcp_path_returns_404():
    manager = _RecordingManager()
    gateway = TransportGateway(manager, _SpyResolver(anonymous_subject()))

    scope = _http_scope("/nope", [])
    captured, send = _capturing_send()
    await gateway(scope, _noop_receive, send)

    assert manager.handled is False
    starts = [m for m in captured if m["type"] == "http.response.start"]
    assert starts and starts[0]["status"] == 404


async def test_websocket_scope_is_closed_not_sent_http_frames():
    manager = _RecordingManager()
    gateway = TransportGateway(manager, _SpyResolver(anonymous_subject()))

    scope: Scope = {"type": "websocket", "path": "/mcp", "headers": []}
    captured, send = _capturing_send()
    await gateway(scope, _noop_receive, send)

    assert manager.handled is False
    assert captured == [{"type": "websocket.close", "code": 1000}]


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


async def _assert_handshake(url: str, call: bool) -> None:
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            assert init.serverInfo.name == "litellm-mcp-gateway"
            listed = await session.list_tools()
            assert listed.tools == []
            if call:
                result = await session.call_tool("anything", {})
                assert result.isError is True
                assert "not wired in S0" in result.content[0].text


async def test_both_paths_served_over_a_real_socket():
    app = build_gateway(build_test_deps())
    with _LiveServer(app) as base_url:
        await _assert_handshake(f"{base_url}/mcp", call=True)
        await _assert_handshake(f"{base_url}/github/mcp", call=False)


async def test_mcp_served_directly_without_redirect():
    app = build_gateway(build_test_deps())
    with _LiveServer(app) as base_url:
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


async def test_published_subject_reaches_the_sdk_handler_over_a_real_socket():
    seen: dict[str, Subject | None] = {"subject": None}
    distinctive = Subject(subject_id="alice", tenant="acme")

    server: Server[object, object] = Server(name="litellm-mcp-gateway", version="2.0.0")

    async def list_tools() -> list[object]:
        seen["subject"] = current_subject()
        return []

    _ = server.list_tools()(list_tools)
    manager = StreamableHTTPSessionManager(app=server, stateless=True, json_response=False)
    transport = TransportGateway(manager, lambda _bearer: distinctive)

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        async with manager.run():
            yield

    app = Starlette(lifespan=lifespan, routes=[Mount("", app=transport)])
    with _LiveServer(app) as base_url:
        async with streamablehttp_client(f"{base_url}/mcp") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.list_tools()

    assert seen["subject"] == distinctive
    assert current_subject() is None
