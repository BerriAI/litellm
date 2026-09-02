import asyncio
import json
from typing import Final

import pytest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm.proxy.middleware.admission_control_middleware import (
    _get_admission_control_settings,
    AdmissionControlMiddleware,
    AdmissionControlSettings,
    AdmissionControlStats,
    get_admission_control_stats,
)


@pytest.fixture(autouse=True)
def reset_state() -> None:
    AdmissionControlMiddleware._admitted = 0
    AdmissionControlMiddleware._queued = 0
    AdmissionControlMiddleware._rejected_total = 0
    AdmissionControlMiddleware._semaphore = None
    AdmissionControlMiddleware._semaphore_loop = None
    yield
    AdmissionControlMiddleware._admitted = 0
    AdmissionControlMiddleware._queued = 0
    AdmissionControlMiddleware._rejected_total = 0
    AdmissionControlMiddleware._semaphore = None
    AdmissionControlMiddleware._semaphore_loop = None


async def _call(
    middleware: AdmissionControlMiddleware,
    path: str = "/",
) -> tuple[Message, ...]:
    messages: Final[list[Message]] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    scope: Final[Scope] = {"type": "http", "path": path, "method": "GET", "headers": []}
    await middleware(scope, receive, send)
    return tuple(messages)


def _handler_with_release(
    started: asyncio.Event,
    release: asyncio.Event,
) -> ASGIApp:
    async def handler(scope: Scope, receive: Receive, send: Send) -> None:
        started.set()
        await release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    return handler


def test_is_not_base_http_middleware() -> None:
    assert not issubclass(AdmissionControlMiddleware, BaseHTTPMiddleware)


@pytest.mark.asyncio
async def test_capacity_rejects_excess_and_releases_queued_request() -> None:
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()
    middleware: Final = AdmissionControlMiddleware(
        _handler_with_release(started, release),
        lambda: AdmissionControlSettings(1, 1, 1.0),
    )

    first: Final = asyncio.create_task(_call(middleware))
    await started.wait()
    second: Final = asyncio.create_task(_call(middleware))
    await asyncio.sleep(0)
    assert get_admission_control_stats().queued == 1

    third: Final = await _call(middleware)
    assert third[0]["status"] == 503
    assert (b"retry-after", b"1") in third[0]["headers"]
    assert json.loads(third[1]["body"])["error"]["code"] == "503"
    assert get_admission_control_stats().rejected_total == 1

    release.set()
    assert (await first)[0]["status"] == 200
    assert (await second)[0]["status"] == 200
    assert get_admission_control_stats() == AdmissionControlStats(0, 0, 1)


@pytest.mark.asyncio
async def test_queue_timeout_rejects_and_decrements_queue() -> None:
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()
    middleware: Final = AdmissionControlMiddleware(
        _handler_with_release(started, release),
        lambda: AdmissionControlSettings(1, 1, 0.05),
    )

    first: Final = asyncio.create_task(_call(middleware))
    await started.wait()
    start_time: Final = asyncio.get_running_loop().time()
    second: Final = await _call(middleware)
    elapsed: Final = asyncio.get_running_loop().time() - start_time

    assert second[0]["status"] == 503
    assert elapsed < 0.5
    assert get_admission_control_stats().queued == 0
    assert get_admission_control_stats().rejected_total == 1
    release.set()
    await first


@pytest.mark.asyncio
async def test_exempt_path_passes_through_when_saturated() -> None:
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()

    async def handler(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["path"] == "/":
            started.set()
            await release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    middleware: Final = AdmissionControlMiddleware(handler, lambda: AdmissionControlSettings(1, 0, 1.0))

    first: Final = asyncio.create_task(_call(middleware))
    await started.wait()
    health: Final = await _call(middleware, "/health/liveliness")
    assert health[0]["status"] == 200
    release.set()
    await first


@pytest.mark.asyncio
async def test_none_settings_does_not_limit_concurrency() -> None:
    active: Final = [0]
    peak: Final = [0]
    all_started: Final = asyncio.Event()
    release: Final = asyncio.Event()

    async def handler(scope: Scope, receive: Receive, send: Send) -> None:
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        if active[0] == 3:
            all_started.set()
        await release.wait()
        active[0] -= 1
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    middleware: Final = AdmissionControlMiddleware(handler, lambda: None)
    requests: Final = tuple(asyncio.create_task(_call(middleware)) for _ in range(3))
    await all_started.wait()
    assert peak[0] == 3
    release.set()
    results: Final = await asyncio.gather(*requests)
    assert tuple(result[0]["status"] for result in results) == (200, 200, 200)


@pytest.mark.asyncio
async def test_cancelling_queued_request_does_not_leak_counter() -> None:
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()
    middleware: Final = AdmissionControlMiddleware(
        _handler_with_release(started, release),
        lambda: AdmissionControlSettings(1, 1, 1.0),
    )

    first: Final = asyncio.create_task(_call(middleware))
    await started.wait()
    queued: Final = asyncio.create_task(_call(middleware))
    await asyncio.sleep(0)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert get_admission_control_stats().queued == 0
    release.set()
    await first


@pytest.mark.asyncio
async def test_streaming_response_holds_admission_until_final_body() -> None:
    first_chunk_sent: Final = asyncio.Event()
    finish_stream: Final = asyncio.Event()

    async def handler(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"first", "more_body": True})
        first_chunk_sent.set()
        await finish_stream.wait()
        await send({"type": "http.response.body", "body": b"last", "more_body": False})

    middleware: Final = AdmissionControlMiddleware(
        handler,
        lambda: AdmissionControlSettings(1, 1, 1.0),
    )
    first: Final = asyncio.create_task(_call(middleware))
    await first_chunk_sent.wait()
    second: Final = asyncio.create_task(_call(middleware))
    await asyncio.sleep(0)
    assert not second.done()
    assert get_admission_control_stats().queued == 1
    finish_stream.set()
    assert (await first)[0]["status"] == 200
    assert (await second)[0]["status"] == 200
    assert get_admission_control_stats().admitted == 0
    assert get_admission_control_stats().queued == 0


@pytest.mark.parametrize(
    ("settings", "expected"),
    (
        ({}, None),
        ({"max_in_flight_requests_per_worker": None}, None),
        ({"max_in_flight_requests_per_worker": 0}, None),
        (
            {"max_in_flight_requests_per_worker": 3},
            AdmissionControlSettings(3, 3, 1.0),
        ),
        (
            {
                "max_in_flight_requests_per_worker": 3,
                "max_queued_requests_per_worker": 5,
                "admission_queue_timeout_seconds": 0.25,
            },
            AdmissionControlSettings(3, 5, 0.25),
        ),
    ),
)
def test_get_admission_control_settings(
    settings: dict[str, object],
    expected: AdmissionControlSettings | None,
) -> None:
    assert _get_admission_control_settings(settings) == expected
