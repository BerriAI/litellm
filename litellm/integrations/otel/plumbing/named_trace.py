"""The trace a proxy request's caller named, so the request's server span starts in it.

A caller names a trace with a ``langfuse_trace_id`` header, else with ``trace_id`` in the request body's metadata,
the route's own metadata field first, the way the ``langfuse`` callback reads them. A request that asks to join a
trace with ``existing_trace_id`` names none: joining a trace without becoming its root is not something a root
server span can do. The server span starts before any route reads the body, so the name is read here, at the
outermost ASGI layer, and handed to the tracer provider's id generator through a context variable. A trace the
request continues from its propagation headers wins: the name is not read when they carry a valid one.
"""

from __future__ import annotations

import functools
import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import anyio.to_thread
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace import Link, Span, SpanKind, Tracer, TracerProvider
from typing_extensions import LiteralString

from litellm._logging import verbose_logger
from litellm.integrations.langfuse.trace_id import resolve_trace_id
from litellm.integrations.otel.model.utils import as_str_mapping

if TYPE_CHECKING:
    from opentelemetry.context import Context
    from opentelemetry.util._decorator import (
        _AgnosticContextManager,  # pyright: ignore[reportPrivateUsage]  # the return type the Tracer ABC declares
    )
    from opentelemetry.util.types import Attributes
    from starlette.types import Message, Receive, Scope

_named_trace_id: Final[ContextVar[int | None]] = ContextVar("litellm_otel_named_trace_id", default=None)
_request_named_trace_id: Final[ContextVar[int | None]] = ContextVar("litellm_otel_request_named_trace_id", default=None)
_request_trace_name: Final[ContextVar[str | None]] = ContextVar("litellm_otel_request_trace_name", default=None)
_request_started_ns: Final[ContextVar[int | None]] = ContextVar("litellm_otel_request_started_ns", default=None)
_last_trace_id_named: Final[ContextVar[bool]] = ContextVar("litellm_otel_last_trace_id_named", default=False)

_TRACE_HEADER_PREFIX: Final = "langfuse_"
_TRACE_KEY: Final = "trace_id"
_JOIN_KEY: Final = "existing_trace_id"
_TRACE_KEY_MARKER: Final = _TRACE_KEY.encode()
# The body is read ahead of the proxy's request size limit, admission control and auth, so what is held here is
# capped per request, by the proxy's own size limit too, and per worker: past a cap a body is handed to the app
# unread and gets its own trace.
_MAX_LOOKAHEAD_BYTES: Final = 16 * 1024 * 1024
_MAX_LOOKAHEAD_BYTES_IN_FLIGHT: Final = 4 * _MAX_LOOKAHEAD_BYTES
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


class NamedTraceIdGenerator(RandomIdGenerator):
    """Starts the first root span of a request in the trace its caller named, and every other root span in a
    random one. The name is spent on that first span, so work the request detaches into its own trace (spend
    tracking, the response cache write) keeps its own trace id. A named trace id is the caller's choice, so its
    span does not carry the W3C random trace id flag."""

    def generate_trace_id(self) -> int:
        named: Final = _named_trace_id.get()
        _last_trace_id_named.set(named is not None)
        if named is None:
            return super().generate_trace_id()
        _named_trace_id.set(None)
        return named

    def is_trace_id_random(self) -> bool:
        return not _last_trace_id_named.get()


@contextmanager
def named_trace_scope(trace_id: int | None) -> Generator[None]:
    token: Final = _named_trace_id.set(trace_id)
    try:
        yield
    finally:
        _named_trace_id.reset(token)


@contextmanager
def request_trace_naming(name: str | None, started_ns: int) -> Generator[None]:
    """One proxy request's trace name, its trace and the moment the request reached the proxy, for as long as it
    runs."""
    trace_id: Final = trace_id_named(name)
    name_token: Final = _request_trace_name.set(name)
    request_token: Final = _request_named_trace_id.set(trace_id)
    started_token: Final = _request_started_ns.set(started_ns)
    try:
        with named_trace_scope(trace_id):
            yield
    finally:
        _request_started_ns.reset(started_token)
        _request_named_trace_id.reset(request_token)
        _request_trace_name.reset(name_token)


def request_named_trace_id() -> int | None:
    """The trace the running proxy request's caller named, ``None`` when it named none."""
    return _request_named_trace_id.get()


def request_trace_name() -> str | None:
    """The trace id the running proxy request's caller sent, as they sent it, ``None`` when they sent none."""
    return _request_trace_name.get()


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
    if _named_trace_id.get() is not None:
        _named_trace_id.set(None)


def sdk_named_trace_id(litellm_params: Mapping[str, object]) -> int | None:
    """The trace an SDK call's caller named in its metadata. A proxy request's name is read at its server span."""
    if litellm_params.get("proxy_server_request"):
        return None
    return trace_id_named(_named_in(_metadata_layers(litellm_params, ("metadata", "litellm_metadata"))))


def _continues_a_trace(scope: Scope) -> bool:
    """Whether the request's propagation headers carry a valid trace, read the way its server span reads them."""
    from opentelemetry import propagate, trace
    from opentelemetry.instrumentation.asgi import (  # pyright: ignore[reportMissingTypeStubs]  # upstream ships no py.typed marker
        asgi_getter,
    )

    carried: Final = propagate.extract(scope, getter=asgi_getter)  # pyright: ignore[reportArgumentType]  # asgi_getter reads the ASGI scope, as the server span does
    return trace.get_current_span(carried).get_span_context().is_valid


def _named(headers: Mapping[str, str], body: Mapping[str, object], route: str) -> str | None:
    return _named_in((_header_trace_fields(headers), *_metadata_layers(body, _metadata_keys(route, body))))


def _named_in(layers: tuple[Mapping[str, object], ...]) -> str | None:
    if _first(_JOIN_KEY, layers) is not None:
        _skip("a request asked to join a trace with existing_trace_id, which OTel V2 does not support")
        return None
    named: Final = _first(_TRACE_KEY, layers)
    return named if isinstance(named, str) and named else None


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


def _skip(reason: LiteralString) -> None:
    verbose_logger.debug("OTel V2 named trace: %s", reason)
    _warn_once(reason)


@functools.cache
def _warn_once(reason: LiteralString) -> None:
    verbose_logger.warning(
        "OTel V2: %s, so a trace_id the request names is not applied and the request gets its own trace. "
        "Later requests that skip naming for this reason are logged at debug level",
        reason,
    )


def _json_object(body: bytes) -> Mapping[str, object]:
    """The request's JSON object, parsed only when the body spells a trace key, so a request naming no trace
    pays one substring scan. A body that does not parse names no trace and is left for the route to reject."""
    if _TRACE_KEY_MARKER not in body:
        return _NO_BODY
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


@asynccontextmanager
async def read_named_trace(
    scope: Scope, receive: Receive, budget: LookaheadBudget, max_body_bytes: int | None = None
) -> AsyncGenerator[tuple[str | None, Receive]]:
    """The trace id an HTTP request's caller named, as they sent it, and the ``receive`` the app must read the request from,
    for as long as the request runs.

    Only a JSON ``POST`` that continues no trace has its body read before the app runs, and only up to
    ``_MAX_LOOKAHEAD_BYTES``, the ``max_body_bytes`` the app accepts, and what ``budget`` has left. What was read
    is held as one buffer, so a body sent in
    many tiny pieces costs its bytes and not a message per piece, and the app gets those bytes back as one
    message, then the rest of the stream, so it reads the same body.
    """
    hold: Final = _Hold(budget)
    try:
        yield await _read_named_trace(scope, receive, hold, _lookahead_cap(max_body_bytes))
    finally:
        hold.release()


def _lookahead_cap(max_body_bytes: int | None) -> int:
    return _MAX_LOOKAHEAD_BYTES if max_body_bytes is None else min(max_body_bytes, _MAX_LOOKAHEAD_BYTES)


async def _read_named_trace(scope: Scope, receive: Receive, hold: _Hold, cap: int) -> tuple[str | None, Receive]:
    from starlette.datastructures import Headers

    if _continues_a_trace(scope):
        return None, receive
    headers: Final = dict(Headers(scope=scope))
    route: Final = str(scope.get("path", ""))
    if scope.get("method") != "POST" or not headers.get("content-type", "").lower().startswith("application/json"):
        return _named(headers, _NO_BODY, route), receive
    if _declared_length(headers) > cap:
        _skip("a request body over the trace lookahead limit was not read")
        return _named(headers, _NO_BODY, route), receive
    held: Final = bytearray()
    more_body = True  # rebind-ok: whether the body goes on past what was read, as its last piece said
    within_budget = True  # rebind-ok: whether the worker's budget had room for every piece read so far
    after: Message | None = None  # rebind-ok: a message that is not body, read while looking for the end of it
    while more_body and within_budget and len(held) <= cap:
        message = await receive()
        if message["type"] != "http.request":
            after = message
            break
        piece = message.get("body", b"")
        held.extend(piece)
        more_body = message.get("more_body", False)
        within_budget = hold.grow(len(piece))
    sent: Final = bytes(held)
    held.clear()
    whole: Final[Message] = {"type": "http.request", "body": sent, "more_body": more_body}
    replayed: Final = iter((whole,) if after is None else (whole, after))

    async def replay() -> Message:
        hold.release()
        message: Final = next(replayed, None)
        return message if message is not None else await receive()

    if not within_budget:
        _skip("the worker's trace lookahead budget was spent")
        return _named(headers, _NO_BODY, route), _as_receive(replay)
    if len(sent) > cap:
        _skip("a request body over the trace lookahead limit was not read")
        return _named(headers, _NO_BODY, route), _as_receive(replay)
    if more_body:
        return _named(headers, _NO_BODY, route), _as_receive(replay)
    coding: Final = headers.get("content-encoding", "").strip().lower()
    if not coding:
        return _named(headers, _json_object(sent), route), _as_receive(replay)
    limit: Final = _decoded_limit(sent, cap)
    if not hold.budget.try_hold(limit):
        _skip("the worker's trace lookahead budget was spent")
        return _named(headers, _NO_BODY, route), _as_receive(replay)
    try:
        named: Final = await anyio.to_thread.run_sync(_named_in_encoded, headers, coding, sent, limit, route)
    finally:
        hold.budget.release(limit)
    return named, _as_receive(replay)


def _decoded_limit(sent: bytes, cap: int) -> int:
    from litellm.constants import MAX_REQUEST_BODY_DECOMPRESSION_RATIO

    return min(cap, len(sent) * MAX_REQUEST_BODY_DECOMPRESSION_RATIO)


def _named_in_encoded(headers: Mapping[str, str], coding: str, sent: bytes, limit: int, route: str) -> str | None:
    """The trace id named by a body sent with a ``Content-Encoding``, which the proxy only decodes further in.

    This runs ahead of the proxy's own decoding and the memory it budgets for it, so a body is decoded to at
    most ``limit``, which the caller holds in the lookahead budget, and dropped once read. One that is larger,
    damaged or in a coding the proxy does not decode names no trace, and one that is not actually encoded is
    read as it was sent, as the proxy reads it.
    """
    from litellm.proxy.middleware.request_body_decompression_middleware import (
        AVAILABLE_CODECS,
        CODING_ALIASES,
        CODING_SIGNATURES,
    )

    name: Final = CODING_ALIASES.get(coding, coding)
    starts_encoded: Final = CODING_SIGNATURES.get(name)
    if starts_encoded is not None and not starts_encoded(sent):
        return _named(headers, _json_object(sent), route)
    codec: Final = AVAILABLE_CODECS.get(name)
    if codec is None:
        _skip("a request body in a Content-Encoding the proxy does not decode was not read")
        return _named(headers, _NO_BODY, route)
    decompressor: Final = codec.new_decompressor()
    try:
        decoded: Final = decompressor.decompress(sent, max_length=limit + 1)
    except codec.error:
        _skip("a request body that did not decode was not read")
        return _named(headers, _NO_BODY, route)
    if not decompressor.eof or len(decoded) > limit:
        _skip("a request body that decodes past the trace lookahead limit was not read")
        return _named(headers, _NO_BODY, route)
    return _named(headers, _json_object(decoded), route)


def _declared_length(headers: Mapping[str, str]) -> int:
    declared: Final = headers.get("content-length", "")
    return int(declared) if declared.isascii() and declared.isdigit() else 0


def _as_receive(replay: Callable[[], Awaitable[Message]]) -> Receive:
    return replay
