import hmac
import os
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import unquote

from asgiref.typing import (
    ASGI3Application,
    ASGIReceiveCallable,
    ASGIReceiveEvent,
    ASGISendCallable,
    ASGISendEvent,
    HTTPDisconnectEvent,
    HTTPRequestEvent,
    HTTPResponseBodyEvent,
    HTTPResponseStartEvent,
    HTTPScope,
    Scope,
    WebSocketCloseEvent,
)
from httpx import AsyncClient, HTTPError
from pydantic import TypeAdapter

from litellm.proxy.experimental_codex_gateway.capture import TraceRecorder, TraceStore
from litellm.proxy.experimental_codex_gateway.metrics import GatewayMetrics
from litellm.proxy.experimental_codex_gateway.pipeline import InspectionStage, RequestPipeline, StageOutcome
from litellm.proxy.experimental_codex_gateway.settings import GatewaySettings

_PUBLIC_RESPONSES_PATH = "/v1/responses"
_INTERNAL_RESPONSES_PATH = "/chatgpt/responses"
_ASGI_APP_ADAPTER: TypeAdapter[ASGI3Application] = TypeAdapter(ASGI3Application)


def map_responses_path(path: str, raw_path: bytes | None = None) -> str | None:
    if path != _PUBLIC_RESPONSES_PATH and not path.startswith(f"{_PUBLIC_RESPONSES_PATH}/"):
        return None
    decoded_path = unquote((raw_path or path.encode()).decode("latin-1"))
    suffix = path[len(_PUBLIC_RESPONSES_PATH) :]
    decoded_suffix = decoded_path[len(_PUBLIC_RESPONSES_PATH) :]
    segments = tuple(segment for segment in decoded_suffix.split("/") if segment)
    if "\\" in decoded_suffix or any(segment in {".", ".."} for segment in segments):
        return None
    return f"{_INTERNAL_RESPONSES_PATH}{suffix}"


def _headers(scope: HTTPScope) -> tuple[tuple[bytes, bytes], ...]:
    return tuple(scope["headers"])


def _header(scope: HTTPScope, name: bytes) -> str | None:
    return next((value.decode("latin-1") for key, value in _headers(scope) if key.lower() == name), None)


async def _response(send: ASGISendCallable, status: int, body: bytes = b"", content_type: bytes | None = None) -> None:
    headers = ((b"content-length", str(len(body)).encode()),)
    resolved_headers = headers if content_type is None else (*headers, (b"content-type", content_type))
    response_start = HTTPResponseStartEvent(
        type="http.response.start", status=status, headers=resolved_headers, trailers=False
    )
    response_body = HTTPResponseBodyEvent(type="http.response.body", body=body, more_body=False)
    await send(response_start)
    await send(response_body)


async def _read_request_body(receive: ASGIReceiveCallable) -> tuple[bytes, bool]:
    chunks: tuple[bytes, ...] = ()
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return b"".join(chunks), True
        if message["type"] != "http.request":
            continue
        chunks = (*chunks, message.get("body", b""))
        if not message.get("more_body", False):
            return b"".join(chunks), False


class ReplayedReceive:
    def __init__(self, body: bytes, disconnected: bool, receive: ASGIReceiveCallable) -> None:
        self._body = body
        self._disconnected = disconnected
        self._receive = receive
        self._sent = False

    async def __call__(self) -> ASGIReceiveEvent:
        if not self._sent:
            self._sent = True
            return HTTPRequestEvent(type="http.request", body=self._body, more_body=False)
        if self._disconnected:
            return HTTPDisconnectEvent(type="http.disconnect")
        return await self._receive()


class ReadinessProbe(Protocol):
    async def reachable(self, url: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class HttpReadinessProbe:
    timeout_seconds: float = 2.0

    async def reachable(self, url: str) -> bool:
        try:
            async with AsyncClient(timeout=self.timeout_seconds) as client:
                await client.get(url, headers={"accept": "application/json"})
        except HTTPError:
            return False
        return True


class ResponseMonitor:
    def __init__(self, started: float, recorder: TraceRecorder | None, send: ASGISendCallable) -> None:
        self.status = 500
        self.first_byte_seconds: float | None = None
        self._started = started
        self._recorder = recorder
        self._send = send

    async def __call__(self, message: ASGISendEvent) -> None:
        if message["type"] == "http.response.start":
            self.status = message["status"]
            if self._recorder is not None:
                self._recorder.response_start(self.status, tuple(message["headers"]))
        if message["type"] == "http.response.body":
            body = message["body"]
            if body and self.first_byte_seconds is None:
                self.first_byte_seconds = time.monotonic() - self._started
            if self._recorder is not None:
                self._recorder.response_body(body)
        await self._send(message)


class CodexGateway:
    def __init__(
        self,
        inner_app: ASGI3Application,
        settings: GatewaySettings,
        pipeline: RequestPipeline,
        metrics: GatewayMetrics,
        trace_store: TraceStore,
        readiness_probe: ReadinessProbe,
    ) -> None:
        self._inner_app = inner_app
        self._settings = settings
        self._pipeline = pipeline
        self._metrics = metrics
        self._trace_store = trace_store
        self._readiness_probe = readiness_probe

    def _authorized(self, scope: HTTPScope) -> bool:
        supplied = _header(scope, b"x-litellm-api-key")
        return supplied is not None and hmac.compare_digest(supplied, self._settings.local_key)

    async def _operational_route(self, scope: HTTPScope, send: ASGISendCallable) -> bool:
        path = scope["path"]
        if path == "/healthz":
            await _response(send, 204)
            return True
        if path == "/readyz":
            ready = await self._readiness_probe.reachable(self._settings.readiness_url)
            await _response(send, 204 if ready else 503)
            return True
        if path == "/metrics":
            if not self._authorized(scope):
                await _response(send, 401)
                return True
            await _response(send, 200, await self._metrics.render(), b"text/plain; version=0.0.4")
            return True
        prefix = "/debug/traces/"
        suffix = "/export"
        if path.startswith(prefix) and path.endswith(suffix):
            if not self._authorized(scope):
                await _response(send, 401)
                return True
            trace_id = path[len(prefix) : -len(suffix)].strip("/")
            trace = self._trace_store.read(trace_id)
            await _response(send, 404 if trace is None else 200, trace or b"", b"application/json")
            return True
        return False

    async def __call__(self, scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
        if scope["type"] == "websocket":
            await send(WebSocketCloseEvent(type="websocket.close", code=4404, reason=""))
            return
        if scope["type"] != "http":
            await self._inner_app(scope, receive, send)
            return
        if await self._operational_route(scope, send):
            return
        mapped_path = map_responses_path(scope["path"], scope["raw_path"])
        if mapped_path is None:
            await _response(send, 404)
            return
        original_body, disconnected = await _read_request_body(receive)
        content_type = _header(scope, b"content-type") or ""
        try:
            pipeline_result = self._pipeline.process(original_body, content_type)
        except Exception:  # noqa: BLE001  # extension stages must fail open on any implementation error
            pipeline_result = RequestPipeline(()).process(original_body, content_type)
            await self._metrics.record_fail_open()
        if pipeline_result.outcome is StageOutcome.FAILED:
            await self._metrics.record_fail_open()
        rewritten_scope: HTTPScope = {**scope, "path": mapped_path, "raw_path": mapped_path.encode()}
        recorder = (
            TraceRecorder(
                method=scope["method"],
                path=scope["path"],
                query_string=scope["query_string"],
                request_headers=_headers(scope),
                pipeline_result=pipeline_result,
                local_key=self._settings.local_key,
                max_trace_bytes=self._settings.max_trace_bytes,
            )
            if self._settings.capture_enabled
            else None
        )
        started = time.monotonic()
        monitored_send = ResponseMonitor(started, recorder, send)
        try:
            await self._inner_app(
                rewritten_scope,
                ReplayedReceive(pipeline_result.body, disconnected, receive),
                monitored_send,
            )
        finally:
            elapsed = time.monotonic() - started
            await self._metrics.record_request(monitored_send.status, elapsed, monitored_send.first_byte_seconds)
            if recorder is not None:
                try:
                    stored = self._trace_store.write(recorder.export())
                    if not stored:
                        await self._metrics.record_capture_drop()
                except Exception:  # noqa: BLE001  # capture failures must never alter the forwarded response
                    await self._metrics.record_capture_drop()


def create_gateway_app(
    settings: GatewaySettings | None = None,
    inner_app: ASGI3Application | None = None,
    pipeline: RequestPipeline | None = None,
    metrics: GatewayMetrics | None = None,
    trace_store: TraceStore | None = None,
    readiness_probe: ReadinessProbe | None = None,
) -> CodexGateway:
    resolved_settings = settings or GatewaySettings.from_environment()
    os.environ["CHATGPT_API_BASE"] = resolved_settings.headroom_base_url
    os.environ.setdefault("LITELLM_MASTER_KEY", resolved_settings.local_key)
    if inner_app is None:
        from litellm.proxy.proxy_server import app as litellm_app

        resolved_inner_app = _ASGI_APP_ADAPTER.validate_python(litellm_app)
    else:
        resolved_inner_app = inner_app
    resolved_trace_store = trace_store or TraceStore(
        directory=resolved_settings.trace_directory,
        max_trace_bytes=resolved_settings.max_trace_bytes,
        max_storage_bytes=resolved_settings.max_trace_storage_bytes,
        retention_seconds=resolved_settings.trace_retention_seconds,
    )
    return CodexGateway(
        inner_app=resolved_inner_app,
        settings=resolved_settings,
        pipeline=pipeline or RequestPipeline((InspectionStage(),)),
        metrics=metrics or GatewayMetrics(),
        trace_store=resolved_trace_store,
        readiness_probe=readiness_probe or HttpReadinessProbe(),
    )
