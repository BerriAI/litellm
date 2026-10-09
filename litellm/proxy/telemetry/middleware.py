import time
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from typing import Final, TypeAlias

from pydantic import TypeAdapter, ValidationError
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm._logging import verbose_proxy_logger
from litellm.proxy.middleware.billable_request_metrics_middleware import classify_billable_request
from litellm.proxy.telemetry.request_context import AttemptObservation, RequestAccumulator, current_request
from litellm.telemetry.records import RequestRecord, StatusClass, TokenCounts
from litellm.telemetry.sink import TelemetrySink

RUST_RESPONSE_HEADER: Final = b"x-litellm-rust"

Spawn: TypeAlias = Callable[[Coroutine[None, None, None]], None]


@dataclass(frozen=True, slots=True)
class ResponseTiming:
    status_code: int
    stream: bool
    to_headers_ms: float | None
    to_first_byte_ms: float | None
    handled_by_rust: bool = False


def _final_observation(observations: tuple[AttemptObservation, ...]) -> AttemptObservation | None:
    succeeded: Final = tuple(observation for observation in observations if observation.succeeded)
    return (succeeded or observations or (None,))[-1]


def build_request_record(
    *, endpoint: str, timing: ResponseTiming, header_keys: frozenset[str], observations: tuple[AttemptObservation, ...]
) -> RequestRecord:
    final: Final = _final_observation(observations)
    return RequestRecord(
        endpoint=endpoint,
        stream=timing.stream,
        litellm_status=StatusClass.from_status_code(timing.status_code),
        provider=final.attempt.provider if final is not None else None,
        deployment_hash=final.attempt.deployment_hash if final is not None else None,
        provider_status=final.attempt.provider_status if final is not None else StatusClass.NONE,
        litellm_cache_hit=final is not None and final.litellm_cache_hit,
        handled_by_rust=timing.handled_by_rust,
        provider_cache_hit=final is not None and final.tokens.cache_read > 0,
        provider_attempts=sum(not observation.litellm_cache_hit for observation in observations),
        tokens=final.tokens if final is not None else TokenCounts(),
        latency_to_headers_ms=timing.to_headers_ms,
        latency_to_first_byte_ms=timing.to_first_byte_ms,
        blocks=final.blocks if final is not None else None,
        header_keys=header_keys,
    )


_HEADERS: Final = TypeAdapter(tuple[tuple[bytes, bytes], ...])


def _headers(asgi_mapping: Mapping[str, object]) -> tuple[tuple[bytes, bytes], ...]:
    try:
        return _HEADERS.validate_python(asgi_mapping.get("headers", ()))
    except ValidationError:
        return ()


def _str_field(asgi_mapping: Mapping[str, object], key: str, default: str) -> str:
    value: Final = asgi_mapping.get(key)
    return value if isinstance(value, str) else default


def _handled_by_rust(headers: tuple[tuple[bytes, bytes], ...]) -> bool:
    return any(name.lower() == RUST_RESPONSE_HEADER and value == b"true" for name, value in headers)


def _is_event_stream(headers: tuple[tuple[bytes, bytes], ...]) -> bool:
    return any(name.lower() == b"content-type" and value.startswith(b"text/event-stream") for name, value in headers)


class _ResponseObserver:
    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock: Final = clock
        self.started: Final = clock()
        self.status_code: int = 500
        self.stream: bool = False
        self.handled_by_rust: bool = False
        self.headers_at: float | None = None
        self.first_body_at: float | None = None

    def observe(self, message: Mapping[str, object]) -> None:
        match message.get("type"):
            case "http.response.start":
                status: Final = message.get("status")
                self.status_code = status if isinstance(status, int) else self.status_code
                self.headers_at = self._clock()
                headers: Final = _headers(message)
                self.stream = _is_event_stream(headers)
                self.handled_by_rust = _handled_by_rust(headers)
            case "http.response.body" if self.first_body_at is None and message.get("body"):
                self.first_body_at = self._clock()
            case _:
                pass

    def timing(self) -> ResponseTiming:
        return ResponseTiming(
            status_code=self.status_code,
            stream=self.stream,
            to_headers_ms=_elapsed_ms(self.started, self.headers_at),
            to_first_byte_ms=_elapsed_ms(self.started, self.first_body_at),
            handled_by_rust=self.handled_by_rust,
        )


def _elapsed_ms(started: float, at: float | None) -> float | None:
    return (at - started) * 1000 if at is not None else None


class TelemetryMiddleware:
    """Times each LLM, MCP and A2A request at the ASGI layer and joins it with the provider attempts it made"""

    def __init__(
        self,
        app: ASGIApp,
        sink_provider: Callable[[], TelemetrySink | None],
        spawn: Spawn,
        *,
        settle_timeout_s: Callable[[], float] = lambda: 2.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.app: Final = app
        self._sink_provider: Final = sink_provider
        self._spawn: Final = spawn
        self._settle_timeout_s: Final = settle_timeout_s
        self._clock: Final = clock

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        sink: Final = self._sink_provider() if scope["type"] == "http" else None
        classification: Final = (
            classify_billable_request(_str_field(scope, "path", ""), _str_field(scope, "method", "POST"))
            if sink is not None
            else None
        )
        if sink is None or classification is None:
            await self.app(scope, receive, send)
            return

        observer: Final = _ResponseObserver(self._clock)
        accumulator: Final = RequestAccumulator()

        async def send_wrapper(message: Message) -> None:
            observer.observe(message)
            await send(message)

        token: Final = current_request.set(accumulator)
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            current_request.reset(token)
            header_keys: Final = frozenset(name.decode("latin-1").lower() for name, _ in _headers(scope))
            self._spawn(self._finish(sink, classification[1], observer.timing(), header_keys, accumulator))

    async def _finish(
        self,
        sink: TelemetrySink,
        endpoint: str,
        timing: ResponseTiming,
        header_keys: frozenset[str],
        accumulator: RequestAccumulator,
    ) -> None:
        await accumulator.wait_for_success(self._settle_timeout_s())
        try:
            sink.record_request(
                build_request_record(
                    endpoint=endpoint, timing=timing, header_keys=header_keys, observations=accumulator.observations
                )
            )
        except Exception:  # noqa: BLE001 -- telemetry must never surface into request handling
            verbose_proxy_logger.debug("telemetry: failed to record request for %s", endpoint, exc_info=True)
