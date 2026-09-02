import asyncio
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Final, Protocol, TypeAlias, runtime_checkable

from pydantic import Field, TypeAdapter, ValidationError
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

_EXEMPT_PATHS: Final[frozenset[str]] = frozenset(
    {
        "/health/liveliness",
        "/health/liveness",
        "/health/readiness",
        "/health/readiness/details",
        "/health/backlog",
        "/health/drain",
        "/metrics",
        "/metrics/",
    }
)


@dataclass(frozen=True, slots=True)
class AdmissionControlSettings:
    max_in_flight_requests: int
    max_queued_requests: int
    queue_timeout_seconds: float


AdmissionControlSettingsGetter: TypeAlias = Callable[[], AdmissionControlSettings | None]  # mutable-ok: Callable params


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


@dataclass(frozen=True, slots=True)
class AdmissionControlMetrics:
    admitted_gauge: _Gauge
    queued_gauge: _Gauge
    rejected_counter: _Counter


AdmissionControlMetricsFactory: TypeAlias = Callable[[], AdmissionControlMetrics | None]  # mutable-ok: Callable params


class AdmissionControlState:
    """Per-process admission counters and the in-flight semaphore shared by one worker's requests."""

    def __init__(self, metrics_factory: AdmissionControlMetricsFactory) -> None:
        self._metrics_factory = metrics_factory
        self._metrics: AdmissionControlMetrics | None = None
        self._metrics_init_attempted = False
        self._admitted = 0
        self._queued = 0
        self._rejected_total = 0
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None

    def get_stats(self) -> AdmissionControlStats:
        return AdmissionControlStats(
            admitted=self._admitted,
            queued=self._queued,
            rejected_total=self._rejected_total,
        )

    def get_semaphore(self, max_in_flight_requests: int) -> asyncio.Semaphore:
        loop: Final = asyncio.get_running_loop()
        if self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(max_in_flight_requests)
            self._semaphore_loop = loop
        semaphore: Final = self._semaphore
        if semaphore is None:
            raise RuntimeError("Admission control semaphore was not initialized")
        return semaphore

    def record_admission(self) -> None:
        self._admitted += 1
        metrics: Final = self._get_metrics()
        if metrics is not None:
            metrics.admitted_gauge.inc()

    def record_release(self) -> None:
        self._admitted -= 1
        metrics: Final = self._get_metrics()
        if metrics is not None:
            metrics.admitted_gauge.dec()

    def record_queue(self) -> None:
        self._queued += 1
        metrics: Final = self._get_metrics()
        if metrics is not None:
            metrics.queued_gauge.inc()

    def record_dequeue(self) -> None:
        self._queued -= 1
        metrics: Final = self._get_metrics()
        if metrics is not None:
            metrics.queued_gauge.dec()

    def record_rejection(self, reason: str) -> None:
        self._rejected_total += 1
        metrics: Final = self._get_metrics()
        if metrics is not None:
            metrics.rejected_counter.labels(reason=reason).inc()

    def _get_metrics(self) -> AdmissionControlMetrics | None:
        if not self._metrics_init_attempted:
            self._metrics_init_attempted = True
            self._metrics = self._metrics_factory()
        return self._metrics


class AdmissionControlMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        get_settings: AdmissionControlSettingsGetter,
        state: AdmissionControlState,
    ) -> None:
        self.app = app
        self.get_settings = get_settings
        self.state = state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        settings: Final = self.get_settings()
        if settings is None or _get_route_path(scope) in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        state: Final = self.state
        semaphore: Final = state.get_semaphore(settings.max_in_flight_requests)
        if state.get_stats().admitted < settings.max_in_flight_requests:
            await semaphore.acquire()
            state.record_admission()
        elif state.get_stats().queued >= settings.max_queued_requests:
            state.record_rejection("queue_full")
            await _overloaded_response(state)(scope, receive, send)
            return
        else:
            state.record_queue()
            try:
                await asyncio.wait_for(
                    semaphore.acquire(),
                    timeout=settings.queue_timeout_seconds,
                )
            except asyncio.TimeoutError:
                state.record_dequeue()
                state.record_rejection("queue_timeout")
                await _overloaded_response(state)(scope, receive, send)
                return
            except asyncio.CancelledError:
                state.record_dequeue()
                raise
            state.record_dequeue()
            state.record_admission()

        try:
            await self.app(scope, receive, send)
        finally:
            semaphore.release()
            state.record_release()


def _get_route_path(scope: Scope) -> str:
    """Strip the ASGI root_path (SERVER_ROOT_PATH) the same way Starlette does before route matching."""
    path: Final[str] = scope["path"]
    root_path: Final[str] = scope.get("root_path", "")
    if not root_path or not path.startswith(root_path):
        return path
    if path == root_path:
        return ""
    if path[len(root_path)] == "/":
        return path[len(root_path) :]
    return path


def _create_gauge(gauge_type: Callable[..., object], name: str, description: str) -> _Gauge:
    metric: Final = (
        gauge_type(name, description, multiprocess_mode="livesum")
        if "PROMETHEUS_MULTIPROC_DIR" in os.environ
        else gauge_type(name, description)
    )
    if not isinstance(metric, _Gauge):
        raise TypeError("Admission gauge has an unexpected type")
    return metric


def create_prometheus_admission_metrics() -> AdmissionControlMetrics | None:
    try:
        from prometheus_client import Counter, Gauge

        return AdmissionControlMetrics(
            admitted_gauge=_create_gauge(
                Gauge,
                "litellm_admission_admitted_requests",
                "Number of requests admitted by this worker",
            ),
            queued_gauge=_create_gauge(
                Gauge,
                "litellm_admission_queued_requests",
                "Number of requests queued by this worker",
            ),
            rejected_counter=Counter(  # mutable-ok: Prometheus requires runtime Counter construction
                "litellm_admission_rejected_requests_total",
                "Number of requests rejected by this worker",
                labelnames=("reason",),
            ),
        )
    except Exception:
        return None


admission_control_state: Final = AdmissionControlState(create_prometheus_admission_metrics)


def get_admission_control_stats() -> AdmissionControlStats:
    return admission_control_state.get_stats()


_PositiveInt: TypeAlias = Annotated[int, Field(gt=0)]
_NonNegativeInt: TypeAlias = Annotated[int, Field(ge=0)]
_PositiveFloat: TypeAlias = Annotated[float, Field(gt=0)]


def get_admission_control_settings(settings: Mapping[str, object]) -> AdmissionControlSettings | None:
    max_in_flight_raw: Final = settings.get("max_in_flight_requests_per_worker")
    if max_in_flight_raw is None:
        return None
    try:
        max_in_flight: Final = TypeAdapter(_PositiveInt).validate_python(max_in_flight_raw)
        max_queued_raw: Final = settings.get("max_queued_requests_per_worker")
        max_queued: Final = (
            max_in_flight if max_queued_raw is None else TypeAdapter(_NonNegativeInt).validate_python(max_queued_raw)
        )
        queue_timeout: Final = TypeAdapter(_PositiveFloat).validate_python(
            settings.get("admission_queue_timeout_seconds", 1.0)
        )
    except ValidationError:
        return None
    return AdmissionControlSettings(
        max_in_flight_requests=max_in_flight,
        max_queued_requests=max_queued,
        queue_timeout_seconds=queue_timeout,
    )


def _overloaded_response(state: AdmissionControlState) -> JSONResponse:
    stats: Final = state.get_stats()
    return JSONResponse(
        status_code=503,
        headers={"retry-after": "1"},  # mutable-ok: Starlette expects a plain headers mapping
        content={  # mutable-ok: Starlette serializes a plain response mapping
            "error": {  # mutable-ok: nested response mapping
                "message": (
                    f"Worker at capacity: {stats.admitted} in-flight, {stats.queued} queued requests. Retry later."
                ),
                "type": "overloaded_error",
                "code": "503",
            }
        },
    )
