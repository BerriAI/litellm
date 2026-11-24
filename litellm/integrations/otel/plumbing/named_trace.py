"""The trace a proxy request's caller named, so the request's server span starts in it.

A caller names a trace with ``trace_id`` or ``existing_trace_id`` in the request body's ``metadata`` or
``litellm_metadata``, or with a ``langfuse_trace_id`` / ``langfuse_existing_trace_id`` header, the same
places and the same order the ``langfuse`` callback reads. The server span starts before any route reads
the body, so the name is read here, at the outermost ASGI layer, and handed to the tracer provider's id
generator through a context variable. A W3C ``traceparent`` header wins: the name is not read when the
request carries one.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import anyio.to_thread
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

from litellm.integrations.langfuse.trace_id import resolve_trace_id
from litellm.integrations.otel.model.utils import as_str_mapping

if TYPE_CHECKING:
    from starlette.types import Message, Receive, Scope

_named_trace_id: Final[ContextVar[int | None]] = ContextVar("litellm_otel_named_trace_id", default=None)

_METADATA_KEYS: Final = ("metadata", "litellm_metadata")
_TRACE_KEYS: Final = ("existing_trace_id", "trace_id")
_LOOKUPS: Final = tuple(product(_METADATA_KEYS, _TRACE_KEYS))
_TRACE_KEY_MARKER: Final = b"trace_id"  # also inside ``existing_trace_id``
_HEADER_PREFIX: Final = "langfuse_"
# The body is read ahead of the proxy's request size limit, admission control and auth, so what is held here is
# capped: a larger body is handed to the app unread past the cap and the request gets a trace of its own.
_MAX_LOOKAHEAD_BYTES: Final = 16 * 1024 * 1024
_DECODING: Final = anyio.CapacityLimiter(1)
_NO_BODY: Final[Mapping[str, object]] = MappingProxyType({})


class NamedTraceIdGenerator(RandomIdGenerator):
    """Starts the first root span of a request in the trace its caller named, and every other root span in a
    random one. The name is spent on that first span, so work the request detaches into its own trace (spend
    tracking, the response cache write) keeps its own trace id."""

    def generate_trace_id(self) -> int:
        named: Final = _named_trace_id.get()
        if named is None:
            return super().generate_trace_id()
        _named_trace_id.set(None)
        return named


@contextmanager
def named_trace_scope(trace_id: int | None) -> Generator[None]:
    token: Final = _named_trace_id.set(trace_id)
    try:
        yield
    finally:
        _named_trace_id.reset(token)


def forget_named_trace() -> None:
    """Drop a name no span took, once the request's server span exists: it either took the name or has a parent."""
    if _named_trace_id.get() is not None:
        _named_trace_id.set(None)


def named_trace_id(headers: Mapping[str, str], body: Mapping[str, object]) -> int | None:
    """The OTel trace id for the trace the caller named, ``None`` when it named none. A UUID or 32 hex
    characters are used as they are and any other string is hashed, as the ``langfuse`` callback does."""
    if "traceparent" in headers:
        return None
    named: Final = next(
        (
            value
            for metadata_key, trace_key in _LOOKUPS
            if isinstance(value := _named(headers, body, metadata_key, trace_key), str) and value
        ),
        None,
    )
    if named is None:
        return None
    return int(resolve_trace_id(named), 16) or None


def _named(headers: Mapping[str, str], body: Mapping[str, object], metadata_key: str, trace_key: str) -> object:
    from_header: Final = headers.get(f"{_HEADER_PREFIX}{trace_key}")
    if from_header is not None:
        return from_header
    metadata: Final = as_str_mapping(body.get(metadata_key))
    return metadata.get(trace_key) if metadata is not None else None


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


async def read_named_trace(scope: Scope, receive: Receive) -> tuple[int | None, Receive]:
    """The trace id an HTTP request's caller named, and the ``receive`` the app must read the request from.

    Only a JSON ``POST`` without a ``traceparent`` has its body read before the app runs, and only up to
    ``_MAX_LOOKAHEAD_BYTES``. What was read is held as one buffer, so a body sent in many tiny pieces costs
    its bytes and not a message per piece, and the app gets those bytes back as one message, then the rest of
    the stream, so it reads the same body.
    """
    headers: Final = {
        name.decode("latin-1").lower(): value.decode("latin-1") for name, value in scope.get("headers", ())
    }
    if "traceparent" in headers:
        return None, receive
    if scope.get("method") != "POST" or not headers.get("content-type", "").lower().startswith("application/json"):
        return named_trace_id(headers, _NO_BODY), receive
    if _declared_length(headers) > _MAX_LOOKAHEAD_BYTES:
        return named_trace_id(headers, _NO_BODY), receive
    # The body bytes read so far: one buffer however many pieces carried them.
    held: Final = bytearray()
    more_body = True  # rebind-ok: whether the body goes on past what was read, as its last piece said
    after: Message | None = None  # rebind-ok: a message that is not body, read while looking for the end of it
    while more_body and len(held) <= _MAX_LOOKAHEAD_BYTES:
        message = await receive()
        if message["type"] != "http.request":
            after = message
            break
        held.extend(message.get("body", b""))
        more_body = message.get("more_body", False)
    sent: Final = bytes(held)
    held.clear()
    whole: Final[Message] = {"type": "http.request", "body": sent, "more_body": more_body}
    replayed: Final = iter((whole,) if after is None else (whole, after))

    async def replay() -> Message:
        message: Final = next(replayed, None)
        return message if message is not None else await receive()

    if more_body or len(sent) > _MAX_LOOKAHEAD_BYTES:
        return named_trace_id(headers, _NO_BODY), _as_receive(replay)
    coding: Final = headers.get("content-encoding", "").strip().lower()
    if not coding:
        return named_trace_id(headers, _json_object(sent)), _as_receive(replay)
    named: Final = await anyio.to_thread.run_sync(_named_in_encoded, headers, coding, sent, limiter=_DECODING)
    return named, _as_receive(replay)


def _named_in_encoded(headers: Mapping[str, str], coding: str, sent: bytes) -> int | None:
    """The trace id named by a body sent with a ``Content-Encoding``, which the proxy only decodes further in.

    This runs ahead of the proxy's own decoding and the memory it budgets for it, so one body is decoded at a
    time and its decoded bytes are dropped before the next: a body is decoded to at most ``_MAX_LOOKAHEAD_BYTES``
    and the proxy's expansion ratio. One that is larger, damaged or in a coding the proxy does not decode names
    no trace, and one that is not actually encoded is read as it was sent, as the proxy reads it.
    """
    from litellm.constants import MAX_REQUEST_BODY_DECOMPRESSION_RATIO
    from litellm.proxy.middleware.request_body_decompression_middleware import (
        AVAILABLE_CODECS,
        CODING_ALIASES,
        CODING_SIGNATURES,
    )

    name: Final = CODING_ALIASES.get(coding, coding)
    starts_encoded: Final = CODING_SIGNATURES.get(name)
    if starts_encoded is not None and not starts_encoded(sent):
        return named_trace_id(headers, _json_object(sent))
    codec: Final = AVAILABLE_CODECS.get(name)
    if codec is None:
        return named_trace_id(headers, _NO_BODY)
    limit: Final = min(_MAX_LOOKAHEAD_BYTES, len(sent) * MAX_REQUEST_BODY_DECOMPRESSION_RATIO)
    decompressor: Final = codec.new_decompressor()
    try:
        decoded: Final = decompressor.decompress(sent, max_length=limit + 1)
    except codec.error:
        return named_trace_id(headers, _NO_BODY)
    if not decompressor.eof or len(decoded) > limit:
        return named_trace_id(headers, _NO_BODY)
    return named_trace_id(headers, _json_object(decoded))


def _declared_length(headers: Mapping[str, str]) -> int:
    declared: Final = headers.get("content-length", "")
    return int(declared) if declared.isascii() and declared.isdigit() else 0


def _as_receive(replay: Callable[[], Awaitable[Message]]) -> Receive:
    return replay
