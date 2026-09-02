import asyncio
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, TypeAlias, runtime_checkable

from pydantic import TypeAdapter, ValidationError
from starlette.types import ASGIApp, Receive, Scope, Send

AdmissionControlSettingsGetter: TypeAlias = Callable[
    [], "AdmissionControlSettings | None"
]  # mutable-ok: Callable typing syntax is parsed as a list literal


@dataclass(frozen=True, slots=True)
class AdmissionControlSettings:
    max_in_flight_requests: int
    max_queued_requests: int
    queue_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class AdmissionControlStats:
    admitted: int
    queued: int
    rejected_total: int


@runtime_checkable
class _Gauge(Protocol):
    def inc(self, amount: float = 1) -> None: ...

    def dec(self, amount: float = 1) -> None: ...


@runtime_checkable
class _CounterChild(Protocol):
    def inc(self, amount: float = 1) -> None: ...


@runtime_checkable
class _Counter(Protocol):
    def labels(self, reason: str) -> _CounterChild: ...


class AdmissionControlMiddleware:
    _EXEMPT_PATHS: Final[frozenset[str]] = frozenset(
        {
            "/health/liveliness",
            "/health/liveness",
            "/health/readiness",
            "/health/readiness/details",
            "/health/backlog",
            "/health/drain",
            "/metrics",
        }
    )
    _admitted: int = 0
    _queued: int = 0
    _rejected_total: int = 0
    _semaphore: asyncio.Semaphore | None = None
    _semaphore_loop: asyncio.AbstractEventLoop | None = None
    _metrics_init_attempted: bool = False
    _admitted_gauge: _Gauge | None = None
    _queued_gauge: _Gauge | None = None
    _rejected_counter: _Counter | None = None

    def __init__(self, app: ASGIApp, get_settings: AdmissionControlSettingsGetter) -> None:
        self.app = app
        self.get_settings = get_settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        settings: Final = self.get_settings()
        if settings is None or scope["path"] in self._EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        semaphore: Final = self._get_semaphore(settings.max_in_flight_requests)
        if self._admitted < settings.max_in_flight_requests:
            await semaphore.acquire()
            self._record_admission()
        elif self._queued >= settings.max_queued_requests:
            self._record_rejection("queue_full")
            await _send_overloaded_response(send=send)
            return
        else:
            self._record_queue()
            try:
                await asyncio.wait_for(
                    semaphore.acquire(),
                    timeout=settings.queue_timeout_seconds,
                )
            except asyncio.TimeoutError:
                self._record_dequeue()
                self._record_rejection("queue_timeout")
                await _send_overloaded_response(send=send)
                return
            except asyncio.CancelledError:
                self._record_dequeue()
                raise
            self._record_dequeue()
            self._record_admission()

        try:
            await self.app(scope, receive, send)
        finally:
            semaphore.release()
            self._record_release()

    @classmethod
    def _get_semaphore(cls, max_in_flight_requests: int) -> asyncio.Semaphore:
        loop: Final = asyncio.get_running_loop()
        if cls._semaphore_loop is not loop:
            cls._semaphore = asyncio.Semaphore(max_in_flight_requests)
            cls._semaphore_loop = loop
        semaphore: Final = cls._semaphore
        if semaphore is None:
            raise RuntimeError("Admission control semaphore was not initialized")
        return semaphore

    @classmethod
    def _record_admission(cls) -> None:
        cls._admitted += 1
        gauge: Final = cls._get_metrics()[0]
        if gauge is not None:
            gauge.inc()

    @classmethod
    def _record_release(cls) -> None:
        cls._admitted -= 1
        gauge: Final = cls._get_metrics()[0]
        if gauge is not None:
            gauge.dec()

    @classmethod
    def _record_queue(cls) -> None:
        cls._queued += 1
        gauge: Final = cls._get_metrics()[1]
        if gauge is not None:
            gauge.inc()

    @classmethod
    def _record_dequeue(cls) -> None:
        cls._queued -= 1
        gauge: Final = cls._get_metrics()[1]
        if gauge is not None:
            gauge.dec()

    @classmethod
    def _record_rejection(cls, reason: str) -> None:
        cls._rejected_total += 1
        counter: Final = cls._get_metrics()[2]
        if counter is not None:
            counter.labels(reason=reason).inc()

    @classmethod
    def get_stats(cls) -> AdmissionControlStats:
        return AdmissionControlStats(
            admitted=cls._admitted,
            queued=cls._queued,
            rejected_total=cls._rejected_total,
        )

    @classmethod
    def _get_metrics(cls) -> tuple[_Gauge | None, _Gauge | None, _Counter | None]:
        if cls._metrics_init_attempted:
            return cls._admitted_gauge, cls._queued_gauge, cls._rejected_counter
        cls._metrics_init_attempted = True
        try:
            from prometheus_client import Counter, Gauge

            admitted_gauge: Final = _create_gauge(
                Gauge,
                "litellm_admission_admitted_requests",
                "Number of requests admitted by this worker",
            )
            queued_gauge: Final = _create_gauge(
                Gauge,
                "litellm_admission_queued_requests",
                "Number of requests queued by this worker",
            )
            rejected_counter: Final = Counter(  # mutable-ok: Prometheus requires runtime Counter construction
                "litellm_admission_rejected_requests_total",
                "Number of requests rejected by this worker",
                labelnames=("reason",),
            )
            cls._admitted_gauge = admitted_gauge
            cls._queued_gauge = queued_gauge
            cls._rejected_counter = rejected_counter
        except Exception:
            cls._admitted_gauge = None
            cls._queued_gauge = None
            cls._rejected_counter = None
        return cls._admitted_gauge, cls._queued_gauge, cls._rejected_counter


def _create_gauge(gauge_type: Callable[..., object], name: str, description: str) -> _Gauge:
    metric: Final = (
        gauge_type(name, description, multiprocess_mode="livesum")
        if "PROMETHEUS_MULTIPROC_DIR" in os.environ
        else gauge_type(name, description)
    )
    if not isinstance(metric, _Gauge):
        raise TypeError("Admission gauge has an unexpected type")
    return metric


def get_admission_control_stats() -> AdmissionControlStats:
    return AdmissionControlMiddleware.get_stats()


def get_admission_control_settings(settings: Mapping[str, object]) -> AdmissionControlSettings | None:
    max_in_flight_raw: Final = settings.get("max_in_flight_requests_per_worker")
    if max_in_flight_raw is None:
        return None
    try:
        max_in_flight: Final = TypeAdapter(int).validate_python(max_in_flight_raw)
        if max_in_flight <= 0:
            return None
        max_queued_raw: Final = settings.get("max_queued_requests_per_worker")
        max_queued: Final = (
            max_in_flight if max_queued_raw is None else TypeAdapter(int).validate_python(max_queued_raw)
        )
        queue_timeout: Final = TypeAdapter(float).validate_python(settings.get("admission_queue_timeout_seconds", 1.0))
    except ValidationError:
        return None
    return AdmissionControlSettings(
        max_in_flight_requests=max_in_flight,
        max_queued_requests=max_queued,
        queue_timeout_seconds=queue_timeout,
    )


async def _send_overloaded_response(send: Send) -> None:
    stats: Final = get_admission_control_stats()
    body: Final = json.dumps(
        {  # mutable-ok: JSON serialization requires a plain response mapping
            "error": {  # mutable-ok: JSON serialization requires a nested response mapping
                "message": (
                    f"Worker at capacity: {stats.admitted} in-flight, {stats.queued} queued requests. Retry later."
                ),
                "type": "overloaded_error",
                "code": "503",
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    await send(
        {  # mutable-ok: ASGI requires a response message mapping
            "type": "http.response.start",
            "status": 503,
            "headers": (
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("latin-1")),
                (b"retry-after", b"1"),
            ),
        }
    )
    await send(
        {  # mutable-ok: ASGI requires a response message mapping
            "type": "http.response.body",
            "body": body,
            "more_body": False,
        }
    )
