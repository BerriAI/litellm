import asyncio
import json
from typing import Final

import pytest
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm.proxy.middleware.admission_control_middleware import (
    ADMISSION_LEASE_SCOPE_KEY,
    AdmissionControlMetrics,
    AdmissionControlMiddleware,
    AdmissionControlSettings,
    AdmissionControlState,
    AdmissionControlStats,
    _parse_admission_control_settings,
    create_prometheus_admission_metrics,
    get_admission_control_settings,
)


@pytest.fixture
def state() -> AdmissionControlState:
    return AdmissionControlState(lambda: None)


async def _call(
    middleware: ASGIApp,
    path: str = "/",
    root_path: str = "",
    parent_scope: Scope | None = None,
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
        "query_string": b"",
        "headers": [],
        ADMISSION_LEASE_SCOPE_KEY: parent_scope.get(ADMISSION_LEASE_SCOPE_KEY) if parent_scope else None,
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
async def test_pending_waiter_is_not_skipped_after_admission_is_released(state: AdmissionControlState) -> None:
    started: Final = asyncio.Event()
    release: Final = asyncio.Event()
    third_trigger: Final = asyncio.Event()
    middleware: Final = AdmissionControlMiddleware(
        _handler_with_release(started, release),
        lambda: AdmissionControlSettings(1, 2, 1.0),
        state,
    )

    first: Final = asyncio.create_task(_call(middleware))
    await started.wait()
    second: Final = asyncio.create_task(_call(middleware))
    await asyncio.sleep(0)

    async def call_third() -> tuple[Message, ...]:
        await third_trigger.wait()
        return await _call(middleware)

    third: Final = asyncio.create_task(call_third())
    await asyncio.sleep(0)
    release.set()
    third_trigger.set()
    await asyncio.sleep(0)

    assert state.get_stats().queued == 2
    await asyncio.gather(first, second, third)


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
    assert REGISTRY.get_sample_value("litellm_admission_rejected_requests_total", {"reason": "queue_full"}) is not None
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
        (
            {"max_in_flight_requests_per_worker": 3, "max_queued_requests_per_worker": 0},
            AdmissionControlSettings(3, 0, 1.0),
        ),
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


def test_invalid_admission_control_settings_logs_once(caplog: pytest.LogCaptureFixture) -> None:
    _parse_admission_control_settings.cache_clear()
    caplog.set_level("ERROR")
    settings: Final = {"max_in_flight_requests_per_worker": [1]}

    assert get_admission_control_settings(settings) is None
    assert get_admission_control_settings(settings) is None

    messages: Final = tuple(
        record.message
        for record in caplog.records
        if record.message.startswith("Ignoring invalid admission control settings")
    )
    assert len(messages) == 1


def _single_slot() -> AdmissionControlSettings:
    return AdmissionControlSettings(1, 0, 1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (None, RuntimeError, asyncio.CancelledError))
async def test_outer_wrapper_retains_route_metadata_after_admission(
    state: AdmissionControlState,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException] | None,
) -> None:
    monkeypatch.delenv("LITELLM_ENABLE_ADMIN_MCP", raising=False)
    app: Final = FastAPI()

    @app.get("/items/{item_id}")
    async def item(item_id: str) -> dict[str, str]:
        assert state.get_stats().admitted == 1
        if failure is not None:
            raise failure("request interrupted")
        return {"item_id": item_id}

    middleware: Final = AdmissionControlMiddleware(app, _single_slot, state)

    async def outer_probe(scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await middleware(scope, receive, send)
        finally:
            assert scope["route"].path == "/items/{item_id}"
            assert scope["endpoint"] is item
            assert scope["path_params"] == {"item_id": "sample"}
            assert ADMISSION_LEASE_SCOPE_KEY not in scope
            assert state.get_stats() == AdmissionControlStats(0, 0, 0)

    if failure is not None:
        with pytest.raises(failure, match="request interrupted"):
            await _call(outer_probe, "/items/sample")
    else:
        response: Final = await _call(outer_probe, "/items/sample")
        assert response[0]["status"] == 200
        assert json.loads(response[1]["body"]) == {"item_id": "sample"}


@pytest.mark.asyncio
async def test_background_request_acquires_a_new_slot_after_parent_finishes(state: AdmissionControlState) -> None:
    release: Final = asyncio.Event()
    background: Final[asyncio.Future[asyncio.Task[tuple[Message, ...]]]] = asyncio.get_running_loop().create_future()

    async def later_request(parent_scope: Scope) -> tuple[Message, ...]:
        await release.wait()
        return await _call(middleware, path="/child", parent_scope=parent_scope)

    async def handler(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["path"] == "/":
            background.set_result(asyncio.create_task(later_request(scope.copy())))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": str(state.get_stats().admitted).encode()})

    middleware: Final = AdmissionControlMiddleware(handler, _single_slot, state)
    parent: Final = await _call(middleware)
    assert parent[1]["body"] == b"1"
    assert state.get_stats() == AdmissionControlStats(0, 0, 0)
    release.set()
    child: Final = await (await background)
    assert child[1]["body"] == b"1"
    assert state.get_stats() == AdmissionControlStats(0, 0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("linked", [False, True])
async def test_only_explicitly_linked_requests_share_admission(state: AdmissionControlState, linked: bool) -> None:
    async def handler(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["path"] == "/":
            nested: Final = await _call(middleware, path="/child", parent_scope=scope if linked else None)
            await send(nested[0])
            await send(nested[1])
            return
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": str(state.get_stats().admitted).encode()})

    middleware: Final = AdmissionControlMiddleware(handler, _single_slot, state)
    response: Final = await _call(middleware)
    assert response[0]["status"] == (200 if linked else 503)
    assert state.get_stats() == AdmissionControlStats(0, 0, 0 if linked else 1)
    if linked:
        assert response[1]["body"] == b"1"


@pytest.mark.asyncio
async def test_admission_lease_cannot_be_reused_by_another_worker(state: AdmissionControlState) -> None:
    def no_metrics() -> None:
        return None

    other_state: Final = AdmissionControlState(no_metrics)

    async def child_handler(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": str(other_state.get_stats().admitted).encode()})

    child: Final = AdmissionControlMiddleware(child_handler, _single_slot, other_state)
    parent: Final = AdmissionControlMiddleware(child, _single_slot, state)
    response: Final = await _call(parent)
    assert response[1]["body"] == b"1"
    assert state.get_stats() == other_state.get_stats() == AdmissionControlStats(0, 0, 0)
