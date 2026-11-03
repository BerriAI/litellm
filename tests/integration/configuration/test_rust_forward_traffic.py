import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final

import anyio
import httpx
import openai
import pytest
from pydantic import BaseModel, JsonValue
from starlette.applications import Starlette
from starlette.requests import Request as StarletteRequest
from starlette.responses import StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect
from websockets.sync.client import connect

from tests.integration._support.asgi import asgi_server
from tests.integration._support.client import eventually
from tests.integration._support.mcp import mcp_peer
from tests.integration._support.ports import allocate_port
from tests.integration._support.process import stop_root_process
from tests.integration._support.wire import Reply, Request, wire_server

MASTER_KEY: Final = "sk-rust-forward-traffic-key-not-a-secret"
KEEPALIVE_SECONDS: Final = 1
FILES_MODEL: Final = "rust-forward-files"
FILE_ID: Final = "file-rust-forward"
CHAT_MODEL: Final = "rust-forward-chat"
REALTIME_MODEL: Final = "rust-forward-realtime"
MCP_HEADERS: Final = MappingProxyType(
    {
        "Authorization": f"Bearer {MASTER_KEY}",
        "Accept": "application/json, text/event-stream",
        "mcp-protocol-version": "2025-06-18",
    }
)


@dataclass(frozen=True, slots=True)
class ScriptedProvider:
    url: str
    released: threading.Event
    hung_up: SimpleQueue[str]


@dataclass(frozen=True, slots=True)
class RustProxy:
    url: str
    port: int
    log: Path
    provider: ScriptedProvider


class _Message(BaseModel):
    content: str


class _ChatRequest(BaseModel):
    messages: tuple[_Message, ...]


class _InputText(BaseModel):
    text: str


class _Item(BaseModel):
    content: tuple[_InputText, ...] = ()


class _RealtimeFrame(BaseModel):
    type: str
    item: _Item | None = None


def _sse(content: str | None, finish_reason: str | None = None) -> bytes:
    chunk: Final = {
        "id": "chatcmpl-rust-forward",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {"index": 0, "delta": {} if content is None else {"content": content}, "finish_reason": finish_reason}
        ],
    }
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


async def _gated_stream(released: threading.Event) -> AsyncIterator[bytes]:
    yield _sse("before-gate")
    await anyio.to_thread.run_sync(released.wait, 30)
    yield _sse("after-gate")
    yield _sse(None, "stop")
    yield b"data: [DONE]\n\n"


async def _endless_stream(marker: str, hung_up: SimpleQueue[str]) -> AsyncIterator[bytes]:
    try:
        for index in range(600):
            yield _sse(f"tick-{index} ")
            await asyncio.sleep(0.1)
    except (asyncio.CancelledError, GeneratorExit):
        hung_up.put(marker)
        raise


def _provider_app(released: threading.Event, hung_up: SimpleQueue[str]) -> Starlette:
    async def chat(request: StarletteRequest) -> StreamingResponse:
        prompt: Final = _ChatRequest.model_validate_json(await request.body()).messages[-1].content
        return StreamingResponse(
            _gated_stream(released) if prompt == "gated" else _endless_stream(prompt, hung_up),
            media_type="text/event-stream",
        )

    async def realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json({"type": "session.created", "session": {"id": "sess_rust_forward"}})
        try:
            while True:
                frame = _RealtimeFrame.model_validate_json(await websocket.receive_text())  # rebind-ok: per frame
                if frame.item is not None:
                    text = "".join(part.text for part in frame.item.content)  # rebind-ok: per frame
                    await websocket.send_json({"type": "response.output_text.delta", "delta": f"echo:{text}"})
        except WebSocketDisconnect:
            return

    return Starlette(
        routes=[
            Route("/v1/chat/completions", chat, methods=["POST"]),
            WebSocketRoute("/v1/realtime", realtime),
        ]
    )


def _routable_address() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 9))
        address: Final[str] = probe.getsockname()[0]
    assert not address.startswith("127."), address
    return address


def _ready(client: httpx.Client) -> bool:
    try:
        return client.get("/health/liveliness").status_code == 200
    except httpx.TransportError:
        return False


def _files_peer(content: bytes) -> Callable[[Request], Reply]:
    file_object: Final = json.dumps(
        {
            "id": FILE_ID,
            "object": "file",
            "purpose": "batch",
            "bytes": len(content),
            "created_at": 1,
            "filename": "upload.jsonl",
            "status": "processed",
        }
    ).encode()

    def respond(request: Request) -> Reply:
        if request.method == "POST" and request.target.startswith("/v1/files"):
            return Reply(body=file_object)
        if request.method == "GET" and request.target.startswith(f"/v1/files/{FILE_ID}/content"):
            return Reply(
                content_type="application/octet-stream", body=content, chunks=(content[:65536], content[65536:])
            )
        return Reply(status=404, body=b'{"error": {"message": "not found"}}')

    return respond


UPLOAD: Final = b"".join(
    json.dumps(
        {
            "custom_id": f"row-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": os.urandom(48).hex()}]},
        }
    ).encode()
    + b"\n"
    for index in range(30_000)
)


@pytest.fixture(scope="module")
def files_upstream() -> Iterator[str]:
    with wire_server(_files_peer(UPLOAD), keep_alive=True) as wire:
        yield wire.url


@pytest.fixture(scope="module")
def rust_proxy(tmp_path_factory: pytest.TempPathFactory, files_upstream: str) -> Iterator[RustProxy]:
    directory: Final = tmp_path_factory.mktemp("rust-forward-traffic")
    log: Final = directory / "proxy.log"
    port: Final = allocate_port()
    released: Final = threading.Event()
    hung_up: Final[SimpleQueue[str]] = SimpleQueue()
    with ExitStack() as stack:
        mcp: Final = stack.enter_context(mcp_peer())
        provider: Final = ScriptedProvider(
            stack.enter_context(asgi_server(_provider_app(released, hung_up))), released, hung_up
        )
        config: Final = directory / "config.yaml"
        config.write_text(
            json.dumps(
                {
                    "model_list": [
                        {
                            "model_name": CHAT_MODEL,
                            "litellm_params": {
                                "model": "openai/gpt-4o-mini",
                                "api_base": f"{provider.url}/v1",
                                "api_key": "integration-provider-key",
                            },
                        },
                        {
                            "model_name": REALTIME_MODEL,
                            "litellm_params": {
                                "model": "openai/gpt-realtime",
                                "api_base": provider.url,
                                "api_key": "integration-provider-key",
                            },
                            "model_info": {"mode": "realtime"},
                        },
                        {
                            "model_name": FILES_MODEL,
                            "litellm_params": {
                                "model": "openai/gpt-4o-mini",
                                "api_base": f"{files_upstream}/v1",
                                "api_key": "integration-provider-key",
                            },
                        },
                    ],
                    "mcp_servers": {"math": {**mcp.registration(), "available_on_public_internet": False}},
                    "general_settings": {
                        "master_key": "os.environ/LITELLM_MASTER_KEY",
                        "mcp_internal_ip_ranges": ["127.0.0.0/8"],
                    },
                }
            )
        )
        environment: Final = {
            **{
                key: value
                for key, value in os.environ.items()
                if not key.startswith("DATABASE_") and key != "DIRECT_URL"
            },
            "LITELLM_MASTER_KEY": MASTER_KEY,
            "LITELLM_RUST": "true",
            "STORE_MODEL_IN_DB": "False",
            "PYTHON_DOTENV_DISABLED": "1",
        }
        output: Final = stack.enter_context(log.open("wb"))
        supervisor: Final = stack.enter_context(
            subprocess.Popen(
                (
                    sys.executable,
                    "-I",
                    "-m",
                    "litellm.proxy.proxy_cli",
                    "--config",
                    str(config),
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(port),
                    "--num_workers",
                    "2",
                    "--keepalive_timeout",
                    str(KEEPALIVE_SECONDS),
                    "--timeout_worker_healthcheck",
                    "120",
                ),
                env=environment,
                stdout=output,
                stderr=output,
            )
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=5) as probe:
                assert eventually(lambda: _ready(probe), bool, seconds=180), log.read_text()
            yield RustProxy(f"http://127.0.0.1:{port}", port, log, provider)
        finally:
            released.set()
            assert stop_root_process(supervisor), log.read_text()
        assert supervisor.returncode == 0, log.read_text()


def _still_open(stream: httpx.Response) -> bool:
    try:
        next(stream.iter_raw())
    except httpx.ReadTimeout:
        return True
    except (StopIteration, httpx.TransportError):
        return False
    return True


def _initialize(client: httpx.Client, headers: Mapping[str, str]) -> httpx.Response:
    return client.post(
        "/mcp",
        headers={**MCP_HEADERS, **headers},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "rust-forward", "version": "1"},
            },
        },
    )


def _mcp_post(client: httpx.Client, session: str, body: Mapping[str, JsonValue]) -> httpx.Response:
    return client.post("/mcp", headers={**MCP_HEADERS, "mcp-session-id": session}, json=body)


def test_mcp_session_keeps_working_while_its_get_stream_sits_idle(rust_proxy: RustProxy) -> None:
    idle_seconds: Final = 8 * KEEPALIVE_SECONDS
    with httpx.Client(base_url=rust_proxy.url, trust_env=False, timeout=30) as client:
        initialized: Final = _initialize(client, {})
        assert initialized.status_code == 200, initialized.text
        session: Final = initialized.headers["mcp-session-id"]
        assert _mcp_post(client, session, {"jsonrpc": "2.0", "method": "notifications/initialized"}).status_code == 202
        with httpx.Client(base_url=rust_proxy.url, trust_env=False, timeout=httpx.Timeout(5, read=2)) as listener:
            with listener.stream(
                "GET", "/mcp", headers={**MCP_HEADERS, "Accept": "text/event-stream", "mcp-session-id": session}
            ) as notifications:
                assert notifications.status_code == 200
                assert notifications.headers["content-type"].startswith("text/event-stream")
                time.sleep(idle_seconds)
                called: Final = _mcp_post(
                    client,
                    session,
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": "math-add", "arguments": {"a": 2, "b": 3}},
                    },
                )
                assert called.status_code == 200, called.text
                assert '"text":"5"' in called.text.replace(" ", ""), called.text
                assert _still_open(notifications)


def test_uploaded_file_and_its_download_cross_byte_for_byte(rust_proxy: RustProxy) -> None:
    with openai.OpenAI(api_key=MASTER_KEY, base_url=rust_proxy.url, max_retries=0, timeout=60) as client:
        created: Final = client.files.create(
            file=("upload.jsonl", UPLOAD), purpose="batch", extra_body={"model": FILES_MODEL}
        )
        downloaded: Final = client.files.content(created.id, extra_query={"model": FILES_MODEL}).content
    assert len(UPLOAD) > 4 * 1024 * 1024
    assert downloaded == UPLOAD, (len(downloaded), len(UPLOAD))


def test_internal_only_mcp_server_judges_the_connecting_peer_not_its_forwarded_for_claim(
    rust_proxy: RustProxy,
) -> None:
    outsider: Final = _routable_address()
    with httpx.Client(
        base_url=f"http://{outsider}:{rust_proxy.port}",
        transport=httpx.HTTPTransport(local_address=outsider),
        trust_env=False,
        timeout=10,
    ) as spoofing:
        spoofed: Final = _initialize(spoofing, {"X-Forwarded-For": "127.0.0.1"})
    with httpx.Client(base_url=rust_proxy.url, trust_env=False, timeout=10) as internal:
        admitted: Final = _initialize(internal, {})
    assert admitted.status_code == 200, admitted.text
    assert spoofed.status_code == 403, spoofed.text
    assert "client IP" in spoofed.text, spoofed.text


def test_oauth_metadata_names_the_origin_the_client_used(rust_proxy: RustProxy) -> None:
    with httpx.Client(base_url=rust_proxy.url, trust_env=False, timeout=10) as client:
        direct: Final = client.get("/.well-known/oauth-protected-resource")
        balanced: Final = client.get(
            "/.well-known/oauth-protected-resource",
            headers={"Host": "gateway.example.test", "X-Forwarded-Proto": "https"},
        )
    assert direct.status_code == 200, direct.text
    assert direct.json()["resource"] == rust_proxy.url
    assert direct.json()["authorization_servers"] == [f"{rust_proxy.url}/mcp"]
    assert balanced.status_code == 200, balanced.text
    assert balanced.json()["resource"] == "https://gateway.example.test"


def _chat_stream(client: httpx.Client, prompt: str) -> AbstractContextManager[httpx.Response]:
    return client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {MASTER_KEY}"},
        json={"model": CHAT_MODEL, "stream": True, "messages": [{"role": "user", "content": prompt}]},
    )


def _content(lines: Iterator[str]) -> str:
    return next(line for line in lines if '"content"' in line)


def test_stream_chunk_reaches_the_client_before_the_provider_sends_the_next(rust_proxy: RustProxy) -> None:
    with httpx.Client(base_url=rust_proxy.url, trust_env=False, timeout=httpx.Timeout(10, read=5)) as client:
        with _chat_stream(client, "gated") as response:
            assert response.status_code == 200, response.read()
            lines: Final = response.iter_lines()
            first: Final = _content(lines)
            rust_proxy.provider.released.set()
            rest: Final = tuple(line for line in lines if line)
    assert "before-gate" in first, first
    assert any("after-gate" in line for line in rest), rest
    assert rest[-1] == "data: [DONE]", rest


def test_client_hang_up_mid_stream_closes_the_provider_stream(rust_proxy: RustProxy) -> None:
    marker: Final = f"endless-{time.monotonic_ns()}"
    with httpx.Client(base_url=rust_proxy.url, trust_env=False, timeout=httpx.Timeout(10, read=5)) as client:
        with _chat_stream(client, marker) as response:
            assert response.status_code == 200, response.read()
            assert "tick-0" in _content(response.iter_lines())
    assert rust_proxy.provider.hung_up.get(timeout=15) == marker


def test_realtime_websocket_frames_cross_both_ways(rust_proxy: RustProxy) -> None:
    marker: Final = f"realtime-{time.monotonic_ns()}"
    with connect(
        f"ws://127.0.0.1:{rust_proxy.port}/v1/realtime?model={REALTIME_MODEL}",
        additional_headers={"Authorization": f"Bearer {MASTER_KEY}"},
        open_timeout=10,
    ) as session:
        session.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": marker}]},
                }
            )
        )
        replies: Final = tuple(session.recv(timeout=10) for _ in range(2))
    assert any(f"echo:{marker}" in reply for reply in replies), replies
