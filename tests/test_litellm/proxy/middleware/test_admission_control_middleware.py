import asyncio
import json
from typing import Final

import pytest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm.proxy.middleware.admission_control_middleware import (
    AdmissionControlMetrics,
    AdmissionControlMiddleware,
    AdmissionControlSettings,
    AdmissionControlState,
    AdmissionControlStats,
    create_prometheus_admission_metrics,
    get_admission_control_settings,
)


@pytest.fixture
def state() -> AdmissionControlState:
    return AdmissionControlState(lambda: None)


async def _call(
    middleware: AdmissionControlMiddleware,
    path: str = "/",
    root_path: str = "",
) -> tuple[Message, ...]:
    messages: Final[list[Message]] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    scope: Final[Scope] = {
        "type": "http",
        "path": path,
        "root_path": root_path,
        "method": "GET",
        "headers": [],
    }
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
async def test_capacity_rejects_excess_and_releases_queued_request(state: AdmissionControlState) -> None:
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()
    middleware: Final = AdmissionControlMiddleware(
        _handler_with_release(started, release),
        lambda: AdmissionControlSettings(1, 1, 1.0),
        state,
    )

    first: Final = asyncio.create_task(_call(middleware))
    await started.wait()
    second: Final = asyncio.create_task(_call(middleware))
    await asyncio.sleep(0)
    assert state.get_stats().queued == 1

    third: Final = await _call(middleware)
    assert third[0]["status"] == 503
    headers: Final = dict(third[0]["headers"])
    assert headers[b"retry-after"] == b"1"
    assert headers[b"content-type"] == b"application/json"
    assert json.loads(third[1]["body"])["error"] == {
        "message": "Worker at capacity: 1 in-flight, 1 queued requests. Retry later.",
        "type": "overloaded_error",
        "code": "503",
    }
    assert state.get_stats().rejected_total == 1

    release.set()
    assert (await first)[0]["status"] == 200
    assert (await second)[0]["status"] == 200
    assert state.get_stats() == AdmissionControlStats(0, 0, 1)


@pytest.mark.asyncio
async def test_queue_timeout_rejects_and_decrements_queue(state: AdmissionControlState) -> None:
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()
    middleware: Final = AdmissionControlMiddleware(
        _handler_with_release(started, release),
        lambda: AdmissionControlSettings(1, 1, 0.05),
        state,
    )

    first: Final = asyncio.create_task(_call(middleware))
    await started.wait()
    start_time: Final = asyncio.get_running_loop().time()
    second: Final = await _call(middleware)
    elapsed: Final = asyncio.get_running_loop().time() - start_time

    assert second[0]["status"] == 503
    assert elapsed < 0.5
    assert state.get_stats().queued == 0
    assert state.get_stats().rejected_total == 1
    release.set()
    await first


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("root_path", "probe_path"),
    (
        ("", "/health/liveliness"),
        ("/proxy", "/proxy/health/liveliness"),
        ("/proxy", "/proxy/metrics"),
    ),
)
async def test_exempt_path_passes_through_when_saturated(
    state: AdmissionControlState,
    root_path: str,
    probe_path: str,
) -> None:
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()

    async def handler(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["path"] == "/":
            started.set()
            await release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    middleware: Final = AdmissionControlMiddleware(handler, lambda: AdmissionControlSettings(1, 0, 1.0), state)

    first: Final = asyncio.create_task(_call(middleware))
    await started.wait()
    health: Final = await _call(middleware, probe_path, root_path)
    assert health[0]["status"] == 200
    blocked: Final = await _call(middleware, "/proxy/v1/chat/completions", root_path)
    assert blocked[0]["status"] == 503
    lookalike: Final = await _call(middleware, "/proxyhealth/liveliness", "/proxy")
    assert lookalike[0]["status"] == 503
    release.set()
    await first


@pytest.mark.asyncio
async def test_non_http_scope_passes_through_when_saturated(state: AdmissionControlState) -> None:
    seen: Final[list[str]] = []

    async def handler(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope["type"])

    middleware: Final = AdmissionControlMiddleware(handler, lambda: AdmissionControlSettings(1, 0, 1.0), state)
    state.record_admission()

    async def receive() -> Message:
        return {"type": "lifespan.startup"}

    async def send(message: Message) -> None:
        return None

    await middleware({"type": "lifespan"}, receive, send)
    assert seen == ["lifespan"]


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

    middleware: Final = AdmissionControlMiddleware(handler, lambda: None, AdmissionControlState(lambda: None))
    requests: Final = tuple(asyncio.create_task(_call(middleware)) for _ in range(3))
    await all_started.wait()
    assert peak[0] == 3
    release.set()
    results: Final = await asyncio.gather(*requests)
    assert tuple(result[0]["status"] for result in results) == (200, 200, 200)


@pytest.mark.asyncio
async def test_cancelling_queued_request_does_not_leak_counter(state: AdmissionControlState) -> None:
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()
    middleware: Final = AdmissionControlMiddleware(
        _handler_with_release(started, release),
        lambda: AdmissionControlSettings(1, 1, 1.0),
        state,
    )

    first: Final = asyncio.create_task(_call(middleware))
    await started.wait()
    queued: Final = asyncio.create_task(_call(middleware))
    await asyncio.sleep(0)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert state.get_stats().queued == 0
    release.set()
    await first


@pytest.mark.asyncio
async def test_streaming_response_holds_admission_until_final_body(state: AdmissionControlState) -> None:
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
        state,
    )
    first: Final = asyncio.create_task(_call(middleware))
    await first_chunk_sent.wait()
    second: Final = asyncio.create_task(_call(middleware))
    await asyncio.sleep(0)
    assert not second.done()
    assert state.get_stats().queued == 1
    finish_stream.set()
    assert (await first)[0]["status"] == 200
    assert (await second)[0]["status"] == 200
    assert state.get_stats().admitted == 0
    assert state.get_stats().queued == 0


class _FakeGauge:
    def __init__(self) -> None:
        self.value = 0.0

    def inc(self, amount: float = 1) -> None:
        self.value += amount

    def dec(self, amount: float = 1) -> None:
        self.value -= amount


class _FakeCounter:
    def __init__(self) -> None:
        self.by_reason: Final[dict[str, _FakeGauge]] = {}

    def labels(self, reason: str) -> _FakeGauge:
        return self.by_reason.setdefault(reason, _FakeGauge())


@pytest.mark.asyncio
async def test_metrics_track_admitted_queued_and_rejected() -> None:
    admitted: Final = _FakeGauge()
    queued: Final = _FakeGauge()
    rejected: Final = _FakeCounter()
    state: Final = AdmissionControlState(
        lambda: AdmissionControlMetrics(admitted_gauge=admitted, queued_gauge=queued, rejected_counter=rejected)
    )
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()
    middleware: Final = AdmissionControlMiddleware(
        _handler_with_release(started, release),
        lambda: AdmissionControlSettings(1, 1, 0.05),
        state,
    )

    first: Final = asyncio.create_task(_call(middleware))
    await started.wait()
    second: Final = asyncio.create_task(_call(middleware))
    await asyncio.sleep(0)
    assert (admitted.value, queued.value) == (1.0, 1.0)
    await _call(middleware)
    assert rejected.by_reason["queue_full"].value == 1.0
    await second
    assert rejected.by_reason["queue_timeout"].value == 1.0
    release.set()
    await first
    assert (admitted.value, queued.value) == (0.0, 0.0)


def test_create_prometheus_admission_metrics_registers_named_metrics() -> None:
    from prometheus_client import REGISTRY

    metrics: Final = create_prometheus_admission_metrics()
    if metrics is not None:
        metrics.admitted_gauge.inc()
        metrics.queued_gauge.inc()
        metrics.rejected_counter.labels(reason="queue_full").inc()
        assert REGISTRY.get_sample_value("litellm_admission_admitted_requests") == 1.0
        assert REGISTRY.get_sample_value("litellm_admission_queued_requests") == 1.0
    assert (
        REGISTRY.get_sample_value("litellm_admission_rejected_requests_total", {"reason": "queue_full"}) is not None
    )
    assert create_prometheus_admission_metrics() is None


@pytest.mark.parametrize(
    ("settings", "expected"),
    (
        ({}, None),
        ({"max_in_flight_requests_per_worker": None}, None),
        ({"max_in_flight_requests_per_worker": 0}, None),
        ({"max_in_flight_requests_per_worker": "many"}, None),
        ({"max_in_flight_requests_per_worker": 3, "max_queued_requests_per_worker": -1}, None),
        ({"max_in_flight_requests_per_worker": 3, "admission_queue_timeout_seconds": 0}, None),
        ({"max_in_flight_requests_per_worker": 3, "admission_queue_timeout_seconds": -0.5}, None),
        ({"max_in_flight_requests_per_worker": 3, "max_queued_requests_per_worker": 0}, AdmissionControlSettings(3, 0, 1.0)),
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
    assert get_admission_control_settings(settings) == expected
