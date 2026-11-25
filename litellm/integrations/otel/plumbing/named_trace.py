"""The trace a caller named for a proxy request or an SDK call, so Langfuse exporters export the call in it.

A caller names a trace with a ``langfuse_trace_id`` header, else with ``trace_id`` in the request body's metadata,
the route's own metadata field first, the way the ``langfuse`` callback reads them. A request that asks to join a
trace with ``existing_trace_id`` names none: joining a trace without becoming its root is not something a root span
can do. Every span keeps its own random trace id, so every other exporter, the ``traceresponse`` header and the spend
log see the trace they always saw. The first root span a call starts carries the name as an attribute, every span
under it inherits it, and a Langfuse exporter moves the spans that carry it into the named trace. The server span
starts before any route reads the body, so the name is read here, at the outermost ASGI layer. A trace the request
continues from its propagation headers wins: the name is not read when they carry a valid one.
"""

from __future__ import annotations

import functools
import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

import anyio
import anyio.to_thread
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import Link, Span, SpanKind, Tracer, TracerProvider, get_current_span

from litellm._logging import verbose_logger
from litellm.integrations.langfuse.trace_id import resolve_trace_id
from litellm.integrations.otel.model.semconv import LiteLLM
from litellm.integrations.otel.model.utils import as_str_mapping

if TYPE_CHECKING:
    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import Span as SDKSpan
    from opentelemetry.util._decorator import (
        _AgnosticContextManager,  # pyright: ignore[reportPrivateUsage]  # the return type the Tracer ABC declares
    )
    from opentelemetry.util.types import Attributes
    from starlette.types import Message, Receive, Scope

    from litellm.proxy.common_utils.http_parsing_utils import RequestBodyKind

SkipReason: TypeAlias = Literal[
    "existing_trace_id",
    "body_too_large",
    "budget_spent",
    "timed_out",
    "unknown_encoding",
    "undecodable",
    "decoded_too_large",
]

_SKIP_MESSAGES: Final[Mapping[SkipReason, str]] = MappingProxyType(
    {
        "existing_trace_id": "a request asked to join a trace with existing_trace_id, which OTel V2 does not support",
        "body_too_large": "a request body over the trace lookahead limit was not read",
        "budget_spent": "the worker's trace lookahead budget was spent",
        "timed_out": "a request body that took longer than the trace lookahead deadline to arrive was not read",
        "unknown_encoding": "a request body in a Content-Encoding the proxy does not decode was not read",
        "undecodable": "a request body that did not decode was not read",
        "decoded_too_large": "a request body that decodes past the trace lookahead limit was not read",
    }
)


@dataclass(frozen=True, slots=True)
class TraceNaming:
    """The trace a call's caller named, else why the proxy read no name for it."""

    trace_id: int | None = None
    skipped: SkipReason | None = None


_NOT_NAMED: Final = TraceNaming()

_pending_naming: Final[ContextVar[TraceNaming | None]] = ContextVar("litellm_otel_pending_naming", default=None)
_request_started_ns: Final[ContextVar[int | None]] = ContextVar("litellm_otel_request_started_ns", default=None)

_TRACE_HEADER_PREFIX: Final = "langfuse_"
_TRACE_KEY: Final = "trace_id"
_JOIN_KEY: Final = "existing_trace_id"
_TRACE_KEY_MARKER: Final = _TRACE_KEY.encode()
# The body is read ahead of the proxy's request size limit, admission control and auth, so what is held here is
# capped per request, in bytes, by the proxy's own size limit too, and in time, and per worker: past a cap a body is
# handed to the app with what was read so far and gets its own trace.
_MAX_LOOKAHEAD_BYTES: Final = 16 * 1024 * 1024
_MAX_LOOKAHEAD_BYTES_IN_FLIGHT: Final = 4 * _MAX_LOOKAHEAD_BYTES
_LOOKAHEAD_DEADLINE_SECONDS: Final = 10.0
_NO_BODY: Final[Mapping[str, object]] = MappingProxyType({})


class LookaheadBudget:
    """The bytes all of a worker's lookaheads hold at once, bodies read and bodies decoded alike. A request that
    finds it spent names no trace instead of waiting, so no request queues behind another before auth."""

    def __init__(self, max_bytes: int = _MAX_LOOKAHEAD_BYTES_IN_FLIGHT) -> None:
        self.max_bytes: Final = max_bytes
        self.held_bytes = 0

    def try_hold(self, size: int) -> bool:
        if self.held_bytes + size > self.max_bytes:
            return False
        self.held_bytes += size
        return True

    def release(self, size: int) -> None:
        self.held_bytes -= size


class _Hold:
    """One request's share of the budget, given back once the app takes the body or the request ends."""

    def __init__(self, budget: LookaheadBudget) -> None:
        self.budget: Final = budget
        self.size = 0

    def grow(self, size: int) -> bool:
        if not self.budget.try_hold(size):
            return False
        self.size += size
        return True

    def release(self) -> None:
        self.budget.release(self.size)
        self.size = 0


class NamedTraceSpanProcessor(SpanProcessor):
    """Marks every span of a trace its caller named with the named trace id. The first root span a call starts takes
    the name and every span under it inherits it from its parent, so work the call detaches into its own trace (spend
    tracking, the response cache write) keeps its own. A root span the proxy read no name for says why."""

    def on_start(self, span: SDKSpan, parent_context: Context | None = None) -> None:
        naming: Final = _take_pending() if span.parent is None else TraceNaming(named_trace_of(parent_context))
        if naming.trace_id is not None:
            span.set_attribute(LiteLLM.NAMED_TRACE_ID, format(naming.trace_id, "032x"))
        if naming.skipped is not None:
            span.set_attribute(LiteLLM.TRACE_ID_SKIPPED, naming.skipped)


def _take_pending() -> TraceNaming:
    pending: Final = _pending_naming.get()
    if pending is None:
        return _NOT_NAMED
    _pending_naming.set(None)
    return pending


def named_trace_of(context: Context | None) -> int | None:
    """The trace its caller named for the span current in ``context``, ``None`` when they named none."""
    span: Final = get_current_span(context)
    return named_trace_id_of(span) if isinstance(span, ReadableSpan) else None


def named_trace_id_of(span: ReadableSpan) -> int | None:
    named: Final = (span.attributes or _NO_BODY).get(LiteLLM.NAMED_TRACE_ID)
    if not isinstance(named, str):
        return None
    try:
        return int(named, 16) or None
    except ValueError:
        return None


@contextmanager
def named_trace_scope(trace_id: int | None) -> Generator[None]:
    """Names the trace of the next root span started inside it, ``None`` naming none."""
    token: Final = _pending_naming.set(None if trace_id is None else TraceNaming(trace_id))
    try:
        yield
    finally:
        _pending_naming.reset(token)


@contextmanager
def request_trace_naming(naming: TraceNaming, started_ns: int) -> Generator[None]:
    """One proxy request's trace name and the moment the request reached the proxy, for as long as it runs."""
    naming_token: Final = _pending_naming.set(naming)
    started_token: Final = _request_started_ns.set(started_ns)
    try:
        yield
    finally:
        _request_started_ns.reset(started_token)
        _pending_naming.reset(naming_token)


def trace_id_named(name: str | None) -> int | None:
    return None if name is None else int(resolve_trace_id(name), 16) or None


class ArrivalTimedTracerProvider(TracerProvider):
    """Hands out tracers that start a proxy request's server span when the request reached the proxy, so the
    time spent reading its body for a trace name stays inside the span."""

    def __init__(self, provider: TracerProvider) -> None:
        self.provider: Final = provider

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: str | None = None,
        schema_url: str | None = None,
        attributes: Attributes = None,
    ) -> Tracer:
        return _ArrivalTimedTracer(
            self.provider.get_tracer(instrumenting_module_name, instrumenting_library_version, schema_url, attributes)
        )


class _ArrivalTimedTracer(Tracer):
    def __init__(self, tracer: Tracer) -> None:
        self.tracer: Final = tracer

    def start_span(
        self,
        name: str,
        context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Span:
        return self.tracer.start_span(
            name,
            context,
            kind,
            attributes,
            links,
            _arrival(kind, start_time),
            record_exception,
            set_status_on_exception,
        )

    def start_as_current_span(
        self,
        name: str,
        context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> _AgnosticContextManager[Span]:
        return self.tracer.start_as_current_span(
            name,
            context,
            kind,
            attributes,
            links,
            _arrival(kind, start_time),
            record_exception,
            set_status_on_exception,
            end_on_exit,
        )


def _arrival(kind: SpanKind, start_time: int | None) -> int | None:
    return _request_started_ns.get() if kind is SpanKind.SERVER and start_time is None else start_time


def forget_named_trace() -> None:
    """Drop a name no span took, once the request's server span exists: it either took the name or has a parent."""
    if _pending_naming.get() is not None:
        _pending_naming.set(None)


def sdk_named_trace_id(litellm_params: Mapping[str, object]) -> int | None:
    """The trace an SDK call's caller named in its metadata. A proxy request's name is read at its server span."""
    if litellm_params.get("proxy_server_request"):
        return None
    return _named_in(_metadata_layers(litellm_params, ("metadata", "litellm_metadata"))).trace_id


def _continues_a_trace(scope: Scope) -> bool:
    """Whether the request's propagation headers carry a valid trace, read the way its server span reads them."""
    from opentelemetry import propagate, trace
    from opentelemetry.instrumentation.asgi import (  # pyright: ignore[reportMissingTypeStubs]  # upstream ships no py.typed marker
        asgi_getter,
    )

    carried: Final = propagate.extract(scope, getter=asgi_getter)  # pyright: ignore[reportArgumentType]  # asgi_getter reads the ASGI scope, as the server span does
    return trace.get_current_span(carried).get_span_context().is_valid


def _named(headers: Mapping[str, str], body: Mapping[str, object], route: str) -> TraceNaming:
    return _named_in((_header_trace_fields(headers), *_metadata_layers(body, _metadata_keys(route, body))))


def _named_in(layers: tuple[Mapping[str, object], ...]) -> TraceNaming:
    if _first(_JOIN_KEY, layers) is not None:
        return _skip("existing_trace_id")
    named: Final = _first(_TRACE_KEY, layers)
    trace_id: Final = trace_id_named(named) if isinstance(named, str) and named else None
    return _NOT_NAMED if trace_id is None else TraceNaming(trace_id)


def _unread(headers: Mapping[str, str], route: str, reason: SkipReason) -> TraceNaming:
    """The trace the request's headers name, since its body was not read, else why it names none."""
    named: Final = _named(headers, _NO_BODY, route)
    return named if named != _NOT_NAMED else _skip(reason)


def _first(key: str, layers: tuple[Mapping[str, object], ...]) -> object:
    return next((layer[key] for layer in layers if layer.get(key) is not None), None)


def _header_trace_fields(headers: Mapping[str, str]) -> Mapping[str, object]:
    fields: Final = ((key, _TRACE_HEADER_PREFIX + key) for key in (_TRACE_KEY, _JOIN_KEY))
    return MappingProxyType({key: headers[header] for key, header in fields if header in headers})


def _metadata_keys(route: str, body: Mapping[str, object]) -> tuple[str, str]:
    """The body's metadata fields, the one the request's callbacks read first."""
    from litellm.proxy.litellm_pre_call_utils import metadata_variable_name_for_request

    own: Final = metadata_variable_name_for_request(route, body)
    return own, "metadata" if own == "litellm_metadata" else "litellm_metadata"


def _metadata_layers(body: Mapping[str, object], keys: tuple[str, str]) -> tuple[Mapping[str, object], ...]:
    return tuple(as_str_mapping(body.get(key)) or _NO_BODY for key in keys)


def _skip(reason: SkipReason) -> TraceNaming:
    verbose_logger.debug("OTel V2 named trace: %s", _SKIP_MESSAGES[reason])
    _warn_once(reason)
    return TraceNaming(skipped=reason)


@functools.cache
def _warn_once(reason: SkipReason) -> None:
    verbose_logger.warning(
        "OTel V2: %s, so a trace_id the request names is not applied and the request gets its own trace, whose root "
        "span carries %s=%s. Later requests that skip naming for this reason are logged at debug level",
        _SKIP_MESSAGES[reason],
        LiteLLM.TRACE_ID_SKIPPED,
        reason,
    )


async def _body_named(kind: RequestBodyKind, scope: Scope, headers: Mapping[str, str], body: bytes) -> TraceNaming:
    """The trace a request's body names, parsed the way the proxy parses it and only when the body spells a trace
    key, so a request naming no trace pays one substring scan."""
    route: Final = str(scope.get("path", ""))
    if _TRACE_KEY_MARKER not in body:
        return _named(headers, _NO_BODY, route)
    parsed: Final = await _form_object(scope, body) if kind == "form" else _json_object(body)
    return _named(headers, parsed, route)


def _json_object(body: bytes) -> Mapping[str, object]:
    """A body that does not parse names no trace and is left for the route to reject."""
    try:
        parsed: Final[object] = _loads(body)
    except (ValueError, RecursionError):
        return _NO_BODY
    return as_str_mapping(parsed) or _NO_BODY


def _loads(body: bytes) -> object:
    try:
        import orjson
    except ImportError:
        return json.loads(body)
    return orjson.loads(body)


async def _form_object(scope: Scope, body: bytes) -> Mapping[str, object]:
    """The form's ``metadata`` field. A form that does not parse names no trace and is left for the route to
    reject."""
    from starlette.datastructures import Headers
    from starlette.formparsers import MultiPartException
    from starlette.requests import Request

    from litellm.proxy._types import ProxyException
    from litellm.proxy.common_utils.http_parsing_utils import parse_form_metadata

    async def receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    request: Final = Request({"type": "http", "method": "POST", "headers": Headers(scope=scope).raw}, receive)
    try:
        form: Final = await request.form()
    except (MultiPartException, ValueError):
        return _NO_BODY
    try:
        metadata: Final = form.get("metadata")
        return MappingProxyType({"metadata": parse_form_metadata(metadata)}) if isinstance(metadata, str) else _NO_BODY
    except (ProxyException, ValueError, RecursionError):
        return _NO_BODY
    finally:
        await form.close()


class _Lookahead:
    """The body read ahead of the app, held as one buffer, so a body sent in many tiny pieces costs its bytes and
    not a message per piece."""

    def __init__(self, receive: Receive, hold: _Hold, cap: int) -> None:
        self.receive: Final = receive
        self.hold: Final = hold
        self.cap: Final = cap
        self.held: Final = bytearray()
        self.more_body = True
        self.within_budget = True
        self.after: Message | None = None

    async def read(self) -> None:
        while self.more_body and self.within_budget and len(self.held) <= self.cap:
            message = await self.receive()
            if message["type"] != "http.request":
                self.after = message
                return
            piece = message.get("body", b"")
            self.held.extend(piece)
            self.more_body = message.get("more_body", False)
            self.within_budget = self.hold.grow(len(piece))

    def taken(self) -> tuple[bytes, Receive]:
        """What was read, and the ``receive`` that hands the app those bytes back as one message, then the rest of
        the stream, so it reads the same body."""
        sent: Final = bytes(self.held)
        self.held.clear()
        whole: Final[Message] = {"type": "http.request", "body": sent, "more_body": self.more_body}
        replayed: Final = iter((whole,) if self.after is None else (whole, self.after))
        hold: Final = self.hold
        receive: Final = self.receive

        async def replay() -> Message:
            hold.release()
            message: Final = next(replayed, None)
            return message if message is not None else await receive()

        return sent, _as_receive(replay)


@asynccontextmanager
async def read_named_trace(
    scope: Scope,
    receive: Receive,
    budget: LookaheadBudget,
    max_body_bytes: int | None = None,
    deadline_seconds: float = _LOOKAHEAD_DEADLINE_SECONDS,
) -> AsyncGenerator[tuple[TraceNaming, Receive]]:
    """The trace an HTTP request's caller named and the ``receive`` the app must read the request from, for as long
    as the request runs.

    Only a ``POST`` that continues no trace and is no binary upload has its body read before the app runs, and only
    up to ``_MAX_LOOKAHEAD_BYTES``, the ``max_body_bytes`` the app accepts, what ``budget`` has left and
    ``deadline_seconds``.
    """
    hold: Final = _Hold(budget)
    try:
        yield await _read_named_trace(scope, receive, hold, _lookahead_cap(max_body_bytes), deadline_seconds)
    finally:
        hold.release()


def _lookahead_cap(max_body_bytes: int | None) -> int:
    return _MAX_LOOKAHEAD_BYTES if max_body_bytes is None else min(max_body_bytes, _MAX_LOOKAHEAD_BYTES)


async def _read_named_trace(
    scope: Scope, receive: Receive, hold: _Hold, cap: int, deadline_seconds: float
) -> tuple[TraceNaming, Receive]:
    from starlette.datastructures import Headers

    from litellm.proxy.common_utils.http_parsing_utils import request_body_kind

    if _continues_a_trace(scope):
        return _NOT_NAMED, receive
    headers: Final = dict(Headers(scope=scope))
    route: Final = str(scope.get("path", ""))
    kind: Final = request_body_kind(headers.get("content-type", ""))
    if scope.get("method") != "POST" or kind == "binary":
        return _named(headers, _NO_BODY, route), receive
    if _declared_length(headers) > cap:
        return _unread(headers, route, "body_too_large"), receive
    lookahead: Final = _Lookahead(receive, hold, cap)
    with anyio.move_on_after(deadline_seconds) as reading:
        await lookahead.read()
    sent, replay = lookahead.taken()
    if reading.cancelled_caught:
        return _unread(headers, route, "timed_out"), replay
    if not lookahead.within_budget:
        return _unread(headers, route, "budget_spent"), replay
    if len(sent) > cap:
        return _unread(headers, route, "body_too_large"), replay
    if lookahead.more_body:
        return _named(headers, _NO_BODY, route), replay
    coding: Final = headers.get("content-encoding", "").strip().lower()
    if not coding:
        return await _body_named(kind, scope, headers, sent), replay
    limit: Final = _decoded_limit(sent, cap)
    if not hold.budget.try_hold(limit):
        return _unread(headers, route, "budget_spent"), replay
    try:
        decoded: Final = await anyio.to_thread.run_sync(_decoded, coding, sent, limit)
        if isinstance(decoded, str):
            return _unread(headers, route, decoded), replay
        return await _body_named(kind, scope, headers, decoded), replay
    finally:
        hold.budget.release(limit)


def _decoded_limit(sent: bytes, cap: int) -> int:
    from litellm.constants import MAX_REQUEST_BODY_DECOMPRESSION_RATIO

    return min(cap, len(sent) * MAX_REQUEST_BODY_DECOMPRESSION_RATIO)


def _decoded(coding: str, sent: bytes, limit: int) -> bytes | SkipReason:
    """A body sent with a ``Content-Encoding``, which the proxy only decodes further in, else why it was not read.

    This runs ahead of the proxy's own decoding and the memory it budgets for it, so a body is decoded to at most
    ``limit``, which the caller holds in the lookahead budget. One that is larger, damaged or in a coding the proxy
    does not decode is not read, and one that is not actually encoded is read as it was sent, as the proxy reads it.
    """
    from litellm.proxy.middleware.request_body_decompression_middleware import (
        AVAILABLE_CODECS,
        CODING_ALIASES,
        CODING_SIGNATURES,
    )

    name: Final = CODING_ALIASES.get(coding, coding)
    starts_encoded: Final = CODING_SIGNATURES.get(name)
    if starts_encoded is not None and not starts_encoded(sent):
        return sent
    codec: Final = AVAILABLE_CODECS.get(name)
    if codec is None:
        return "unknown_encoding"
    decompressor: Final = codec.new_decompressor()
    try:
        decoded: Final = decompressor.decompress(sent, max_length=limit + 1)
    except codec.error:
        return "undecodable"
    if not decompressor.eof or len(decoded) > limit:
        return "decoded_too_large"
    return decoded


def _declared_length(headers: Mapping[str, str]) -> int:
    declared: Final = headers.get("content-length", "")
    return int(declared) if declared.isascii() and declared.isdigit() else 0


def _as_receive(replay: Callable[[], Awaitable[Message]]) -> Receive:
    return replay
