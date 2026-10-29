import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import httpx
import pytest
import pytest_asyncio
import yaml
from aiohttp import web

from tests.integration._support.anthropic_sse import LIFECYCLE, delta_text, event_types, parse_sse, user_prompt
from tests.integration._support.client import JSON_OBJECT, Gateway
from tests.integration._support.openai_wire import chat_stream
from tests.integration._support.ports import allocate_port
from tests.integration._support.process import OwnedProxy, owned_proxy_process, paused

_MODEL: Final = "disconnect-repro"
_KEY: Final = "sk-integration-master-key"
_DEADLINE: Final = 10
_RACE_ATTEMPTS: Final = 20


class _Mode(Enum):
    OCCUPY = "occupy"
    QUEUED = "queued"
    CONTENT = "content"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class _Request:
    mode: _Mode
    headers_allowed: asyncio.Event = field(default_factory=asyncio.Event)
    headers_sent: asyncio.Event = field(default_factory=asyncio.Event)
    running: asyncio.Event = field(default_factory=asyncio.Event)
    released: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(frozen=True, slots=True)
class _Backend:
    url: str
    requests: asyncio.Queue[_Request]
    runner: web.AppRunner


@pytest_asyncio.fixture(loop_scope="function")
async def backend() -> AsyncIterator[_Backend]:
    requests: Final[asyncio.Queue[_Request]] = asyncio.Queue()
    worker: Final = asyncio.Semaphore(1)

    async def complete(request: web.Request) -> web.StreamResponse:
        body: Final = JSON_OBJECT.validate_json(await request.read())
        assert body["model"] == "test-model" and body["stream"] is True, body
        assert request.headers["Authorization"] == "Bearer integration-provider-key"
        pending: Final = _Request(_Mode(user_prompt(body)))
        await requests.put(pending)
        response: Final = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        try:
            await pending.headers_allowed.wait()
            await response.prepare(request)
            pending.headers_sent.set()
            async with worker:
                pending.running.set()
                chunks: Final = chat_stream("chatcmpl-disconnect", "test-model", "hello")
                if pending.mode in (_Mode.CONTENT, _Mode.COMPLETE):
                    for chunk in chunks[:2]:
                        await response.write(chunk)
                if pending.mode is _Mode.COMPLETE:
                    await response.write(chunks[2])
                    await response.write_eof()
                else:
                    await asyncio.Event().wait()
        finally:
            pending.released.set()
        return response

    app: Final = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    runner: Final = web.AppRunner(app, handler_cancellation=True, shutdown_timeout=0.1)
    await runner.setup()
    port: Final = allocate_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    try:
        yield _Backend(f"http://127.0.0.1:{port}", requests, runner)
    finally:
        await runner.cleanup()


@pytest_asyncio.fixture(loop_scope="function")
async def proxy(backend: _Backend, tmp_path: Path) -> AsyncIterator[OwnedProxy]:
    config: Final = tmp_path / "disconnect.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_list": [
                    {
                        "model_name": _MODEL,
                        "litellm_params": {
                            "model": "custom_openai/test-model",
                            "api_base": backend.url + "/v1",
                            "api_key": "integration-provider-key",
                        },
                    }
                ],
                "litellm_settings": {"anthropic_sse_ping_interval_seconds": 0},
                "router_settings": {"num_retries": 0},
                "general_settings": {"dangerously_permit_weak_or_unset_master_key": True},
            }
        )
    )
    with httpx.Client() as client:
        template: Final = Gateway(client, _KEY, backend.url)
        with owned_proxy_process(
            template,
            tmp_path,
            {"STORE_MODEL_IN_DB": "False"},
            config=config,
            database_setup=(),
            remove_environment=("DATABASE_URL", "DATABASE_URL_READ_REPLICA", "REDIS_HOST", "REDIS_PORT"),
        ) as owned:
            try:
                yield owned
            finally:
                await backend.runner.cleanup()


def _body(mode: _Mode) -> bytes:
    return json.dumps(
        {"model": _MODEL, "stream": True, "max_tokens": 16, "messages": [{"role": "user", "content": mode.value}]}
    ).encode()


@asynccontextmanager
async def _request(
    proxy: OwnedProxy, mode: _Mode, endpoint: str = "/v1/messages"
) -> AsyncGenerator[asyncio.StreamReader, None]:
    address: Final = urlsplit(str(proxy.gateway.client.base_url))
    reader, writer = await asyncio.open_connection(address.hostname, address.port)
    body: Final = _body(mode)
    writer.write(
        f"POST {endpoint} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {_KEY}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    await writer.drain()
    try:
        yield reader
    finally:
        writer.close()
        await writer.wait_closed()


async def _next(backend: _Backend, mode: _Mode) -> _Request:
    pending: Final = await asyncio.wait_for(backend.requests.get(), _DEADLINE)
    assert pending.mode is mode
    return pending


async def _headers(pending: _Request) -> None:
    pending.headers_allowed.set()
    await asyncio.wait_for(pending.headers_sent.wait(), _DEADLINE)


async def _released(pending: _Request) -> None:
    try:
        await asyncio.wait_for(pending.released.wait(), _DEADLINE)
    except TimeoutError:
        pytest.fail(f"Backend {pending.mode.value} request still active after client disconnect", pytrace=False)


async def _disconnect_queued_request(backend: _Backend, proxy: OwnedProxy) -> None:
    with ExitStack() as scheduling:
        async with _request(proxy, _Mode.QUEUED):
            queued: Final = await _next(backend, _Mode.QUEUED)
            scheduling.enter_context(paused(proxy.process))
            await _headers(queued)
            assert not queued.running.is_set()
    await _released(queued)
    assert backend.requests.empty(), "Disconnect must not trigger another upstream request"


async def test_disconnect_with_upstream_headers_pending_releases_queued_request(
    backend: _Backend, proxy: OwnedProxy
) -> None:
    """Hold headers and the client FIN ready together without patching proxy code."""
    async with _request(proxy, _Mode.OCCUPY):
        occupied: Final = await _next(backend, _Mode.OCCUPY)
        await _headers(occupied)
        await asyncio.wait_for(occupied.running.wait(), _DEADLINE)
        for attempt in range(_RACE_ATTEMPTS):
            await _disconnect_queued_request(backend, proxy)
            assert not occupied.released.is_set(), attempt
    await _released(occupied)


@pytest.mark.parametrize("endpoint", ["/v1/messages", "/v1/chat/completions"])
async def test_disconnect_after_content_releases_upstream(backend: _Backend, proxy: OwnedProxy, endpoint: str) -> None:
    async with _request(proxy, _Mode.CONTENT, endpoint) as response:
        pending: Final = await _next(backend, _Mode.CONTENT)
        await _headers(pending)
        received: Final = await asyncio.wait_for(response.readuntil(b'"hello"'), _DEADLINE)
        assert b"200 OK" in received
        if endpoint == "/v1/messages":
            assert b"content_block_delta" in received
    await _released(pending)
    assert backend.requests.empty(), "Disconnect must not trigger another upstream request"


async def test_messages_stream_completes_with_unchanged_events(backend: _Backend, proxy: OwnedProxy) -> None:
    async with httpx.AsyncClient(base_url=str(proxy.gateway.client.base_url), trust_env=False) as client:
        response_task: Final = asyncio.create_task(
            client.post(
                "/v1/messages",
                headers={"Authorization": f"Bearer {_KEY}", "Content-Type": "application/json"},
                content=_body(_Mode.COMPLETE),
            )
        )
        pending: Final = await _next(backend, _Mode.COMPLETE)
        await _headers(pending)
        response: Final = await asyncio.wait_for(response_task, _DEADLINE)
    assert response.status_code == 200, response.text
    events: Final = parse_sse(response.text)
    assert event_types(events) == LIFECYCLE
    assert delta_text(events) == "hello"
    assert pending.released.is_set()
    assert backend.requests.empty(), "Successful streaming must not trigger another upstream request"
