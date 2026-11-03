from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Final

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from litellm.proxy.telemetry.middleware import TelemetryMiddleware
from litellm.proxy.telemetry.request_context import AttemptObservation, current_request
from litellm.telemetry.records import (
    AttemptRecord,
    InstanceInfo,
    RequestRecord,
    StatusClass,
    TokenCounts,
    UIEvent,
)


@dataclass
class _RecordingSink:
    requests: tuple[RequestRecord, ...] = ()

    def set_instance(self, info: InstanceInfo) -> None: ...

    def record_request(self, record: RequestRecord) -> None:
        self.requests = (*self.requests, record)

    def record_attempt(self, record: AttemptRecord) -> None: ...

    def record_ui_event(self, event: UIEvent) -> None: ...

    async def flush(self) -> None: ...


@dataclass
class _Spawned:
    coroutines: tuple[Coroutine[None, None, None], ...] = ()

    def __call__(self, coroutine: Coroutine[None, None, None]) -> None:
        self.coroutines = (*self.coroutines, coroutine)

    async def drain(self) -> None:
        for coroutine in self.coroutines:
            await coroutine


@dataclass
class _Clock:
    now: float = 0.0

    def __call__(self) -> float:
        self.now += 0.1
        return self.now


def _attempt(provider: str, status: StatusClass, *, succeeded: bool, tokens: TokenCounts) -> AttemptObservation:
    return AttemptObservation(
        attempt=AttemptRecord(provider=provider, provider_status=status, stream=False),
        succeeded=succeeded,
        tokens=tokens,
    )


async def _retried_then_served(request: Request) -> Response:
    accumulator: Final = current_request.get()
    assert accumulator is not None
    accumulator.add(_attempt("azure", StatusClass.SERVER_ERROR, succeeded=False, tokens=TokenCounts()))
    accumulator.add(_attempt("openai", StatusClass.SUCCESS, succeeded=True, tokens=TokenCounts(input=12, output=3)))
    return JSONResponse({"ok": True})


async def _streamed(request: Request) -> Response:
    async def chunks():
        yield b"data: 1\n\n"
        yield b"data: [DONE]\n\n"

    return StreamingResponse(chunks(), media_type="text/event-stream")


async def _served_by_rust(request: Request) -> Response:
    return JSONResponse({"ok": True}, headers={"x-litellm-rust": "true"})


async def _rejected(request: Request) -> Response:
    return JSONResponse({"error": "bad key"}, status_code=401)


def _client(
    sink: _RecordingSink | Callable[[], _RecordingSink | None], spawned: _Spawned, *, settle_timeout_s: float = 0.0
) -> AsyncClient:
    app: Final = Starlette(
        routes=[
            Route("/v1/chat/completions", _retried_then_served, methods=["POST"]),
            Route("/v1/messages", _streamed, methods=["POST"]),
            Route("/v1/embeddings", _rejected, methods=["POST"]),
            Route("/v1/responses", _served_by_rust, methods=["POST"]),
            Route("/health", _rejected, methods=["GET"]),
        ]
    )
    wrapped: Final = TelemetryMiddleware(
        app,
        sink_provider=sink if callable(sink) else lambda: sink,
        spawn=spawned,
        settle_timeout_s=lambda: settle_timeout_s,
        clock=_Clock(),
    )
    return AsyncClient(transport=ASGITransport(app=wrapped), base_url="http://proxy")


@pytest.mark.asyncio
async def test_a_retried_request_reports_the_serving_attempt_and_the_attempt_count() -> None:
    sink: Final = _RecordingSink()
    spawned: Final = _Spawned()
    async with _client(sink, spawned) as client:
        response: Final = await client.post(
            "/v1/chat/completions", json={}, headers={"Anthropic-Beta": "x", "Authorization": "Bearer sk-1"}
        )
    assert response.status_code == 200, response.text
    await spawned.drain()

    (record,) = sink.requests
    assert record.endpoint == "/chat/completions"
    assert (record.provider, record.provider_status, record.provider_attempts) == ("openai", StatusClass.SUCCESS, 2)
    assert record.litellm_status is StatusClass.SUCCESS
    assert record.tokens == TokenCounts(input=12, output=3)
    assert record.stream is False
    assert record.latency_to_first_byte_ms is not None
    assert {"anthropic-beta", "authorization"} <= record.header_keys


@pytest.mark.asyncio
async def test_a_stream_reports_time_to_headers_before_the_first_byte() -> None:
    sink: Final = _RecordingSink()
    spawned: Final = _Spawned()
    async with _client(sink, spawned) as client:
        response: Final = await client.post("/v1/messages", json={})
    assert response.status_code == 200, response.text
    await spawned.drain()

    (record,) = sink.requests
    assert record.stream is True
    assert record.latency_to_headers_ms is not None
    assert record.latency_to_first_byte_ms is not None
    assert record.latency_to_headers_ms < record.latency_to_first_byte_ms
    assert (record.provider, record.provider_status, record.provider_attempts) == (None, StatusClass.NONE, 0)


@pytest.mark.asyncio
async def test_a_request_rejected_before_any_provider_call_has_no_provider_status() -> None:
    sink: Final = _RecordingSink()
    spawned: Final = _Spawned()
    async with _client(sink, spawned, settle_timeout_s=0.01) as client:
        response: Final = await client.post("/v1/embeddings", json={})
    assert response.status_code == 401, response.text
    await spawned.drain()

    (record,) = sink.requests
    assert (record.litellm_status, record.provider_status) == (StatusClass.CLIENT_ERROR, StatusClass.NONE)


@pytest.mark.asyncio
async def test_routes_outside_llm_mcp_and_a2a_are_not_recorded() -> None:
    sink: Final = _RecordingSink()
    spawned: Final = _Spawned()
    async with _client(sink, spawned) as client:
        await client.get("/health")
    assert spawned.coroutines == ()
    assert sink.requests == ()


@pytest.mark.asyncio
async def test_only_a_response_carrying_the_rust_header_counts_as_handled_by_rust() -> None:
    sink: Final = _RecordingSink()
    spawned: Final = _Spawned()
    async with _client(sink, spawned) as client:
        rust_response: Final = await client.post("/v1/responses", json={})
        python_response: Final = await client.post("/v1/messages", json={})
    assert (rust_response.status_code, python_response.status_code) == (200, 200)
    await spawned.drain()
    assert [(record.endpoint, record.handled_by_rust) for record in sink.requests] == [
        ("/responses", True),
        ("/v1/messages", False),
    ]


@pytest.mark.asyncio
async def test_a_request_finishing_after_the_groups_change_is_recorded_into_the_current_sink() -> None:
    before: Final = _RecordingSink()
    after: Final = _RecordingSink()
    sinks: Final = [before]  # mutable-ok: the runtime swaps its sink between windows
    spawned: Final = _Spawned()
    async with _client(lambda: sinks[-1], spawned) as client:
        response: Final = await client.post("/v1/chat/completions", json={})
    assert response.status_code == 200, response.text
    sinks.append(after)
    await spawned.drain()

    assert (before.requests, len(after.requests)) == ((), 1)
