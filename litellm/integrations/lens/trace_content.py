import asyncio
import hashlib
import json
import sys
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from io import StringIO
from itertools import chain
from types import MappingProxyType
from typing import Final, TypeAlias

from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace import Span as RecordingSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import format_span_id, format_trace_id
from typing_extensions import TypeIs, override

from litellm.integrations.lens.content import (
    MAX_CONTENT_JOBS,
    MAX_PENDING_CONTENT_BYTES,
    MAX_SERIALIZATIONS,
    REFERENCE_ATTRIBUTE,
    UPLOAD_TIMEOUT_SECONDS,
    ContentLimits,
    ContentState,
    ContentStore,
    Owner,
    Stored,
    Unavailable,
    object_name,
    signed_reference,
)
from litellm.integrations.lens.exporter import ExportFailure
from litellm.integrations.lens.exporter import (
    _check_size as check_size,  # pyright: ignore[reportPrivateUsage]  # one bounded size walk for every Lens payload
)
from litellm.integrations.otel.model.metadata import RequestIdentity
from litellm.integrations.otel.plumbing.providers import (
    _CountedSpanView as CountedSpanView,  # pyright: ignore[reportPrivateUsage]  # the plumbing's span view keeps dropped counts
)
from litellm.integrations.otel.plumbing.providers import span_limits

MAX_EXPORT_BATCH_BYTES: Final = 16 * 1024 * 1024
_MAX_DEPTH: Final = 32
ContentAttributes: TypeAlias = Mapping[str, str]
ContentWork: TypeAlias = Callable[[int], ContentAttributes | Unavailable]
_SpanKey: TypeAlias = tuple[int, int]


def _bounded_json(
    value: object, limit: int, *, default: Callable[[object], object] | None, ascii_only: bool
) -> str | None:
    """``json.dumps(value)`` with the given options, or None as soon as the text would pass ``limit`` characters."""
    encoder: Final = json.JSONEncoder(default=default, ensure_ascii=ascii_only)
    with StringIO() as text:
        for chunk in encoder.iterencode(value):
            if text.tell() + len(chunk) > limit:
                return None
            _ = text.write(chunk)
        return text.getvalue()


def _is_dict(value: object) -> TypeIs[dict[object, object]]:  # guard-ok: stringify_message keeps dict messages only
    return isinstance(value, dict)


def _is_items(value: object) -> TypeIs[list[object] | tuple[object, ...]]:  # guard-ok: walks stored containers only
    return isinstance(value, (list, tuple))


def _stringified(message: object, limit: int) -> object | None | Unavailable:
    """One message as ``stringify_message`` round-trips it, without ever holding more than ``limit`` of its text."""
    if not _is_dict(message):
        return None
    try:
        text: Final = _bounded_json(message, limit, default=str, ascii_only=True)
    except Exception:  # noqa: BLE001  # mirrors stringify_message, which drops a message it cannot encode
        return None
    if text is None:
        return Unavailable.TOO_LARGE
    loaded: object = json.loads(text)  # pyright: ignore[reportAny]  # untyped JSON is re-encoded unchanged
    return loaded


def serialized_messages(messages: Sequence[object], limit: int) -> str | None | Unavailable:
    """Exactly ``serialize_messages(messages)``, bounded to ``limit`` characters while it is built."""
    loaded: Final = tuple(_stringified(message, limit) for message in messages)
    if any(item is Unavailable.TOO_LARGE for item in loaded):
        return Unavailable.TOO_LARGE
    kept: Final = [item for item in loaded if item is not None]
    if not kept:
        return None
    text: Final = _bounded_json(kept, limit, default=None, ascii_only=True)
    return Unavailable.TOO_LARGE if text is None else text


def _is_mapping(value: object) -> TypeIs[Mapping[object, object]]:  # guard-ok: walks stored containers only
    return isinstance(value, Mapping)


def _children(value: object) -> Iterable[object]:
    if _is_mapping(value):
        return chain.from_iterable(value.items())
    if _is_items(value):
        return value
    return ()


def _retained_bytes(value: object, budget: int, depth: int = 0) -> int | None:
    """Memory ``value`` keeps alive as stored, or None once it passes ``budget``."""
    if depth > _MAX_DEPTH:
        return None
    total = sys.getsizeof(value)  # rebind-ok: consumes a finite budget across children
    for child in _children(value):
        if total > budget:
            return None
        match _retained_bytes(child, budget - total, depth + 1):
            case None:
                return None
            case int() as counted:
                total += counted
    return None if total > budget else total


@dataclass(frozen=True, slots=True)
class _Encoded:
    body: str
    size: int
    sha256: str


def _encoded(work: ContentWork, limit: int) -> _Encoded | Unavailable:
    attributes: Final = work(limit)
    if isinstance(attributes, Unavailable):
        return attributes
    try:
        body: Final = _bounded_json(dict(attributes), limit, default=None, ascii_only=False)
        data: Final = body.encode() if body is not None else None
    except (ValueError, TypeError, RecursionError, RuntimeError, UnicodeEncodeError):
        return Unavailable.UNSERIALIZABLE
    if data is None or len(data) > limit:
        return Unavailable.TOO_LARGE
    return _Encoded(body=body or "", size=len(data), sha256=hashlib.sha256(data).hexdigest())


@dataclass(slots=True)
class _Capture:
    owner: Owner
    raw: int
    held: int = 0
    submitted: bool = False
    state: ContentState | None = None
    ended: ReadableSpan | None = None
    forward: Callable[[ReadableSpan], None] | None = None


class TraceContentSpool:
    """Content too large to ride inline on a Lens span, stored before the span is exported.

    Content is taken from the span while it is still open, before any attribute limit applies. A
    span whose content is being stored is held at its end and exported once the upload is
    acknowledged or fails, carrying a signed reference where the content was. Spans without a
    capture pass straight through.
    """

    def __init__(
        self,
        store: ContentStore,
        signing_key: bytes,
        loop: asyncio.AbstractEventLoop,
        limits: ContentLimits = ContentLimits(),
        upload_timeout: float = UPLOAD_TIMEOUT_SECONDS,
    ) -> None:
        self.store: Final = store
        self.signing_key: Final = signing_key
        self.loop: Final = loop
        self.limits: Final = limits
        self.upload_timeout: Final = upload_timeout
        self.lock: Final = threading.Lock()
        self.captures: Final[dict[_SpanKey, _Capture]] = {}  # mutable-ok: at most MAX_CONTENT_JOBS spans
        self.tasks: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: at most MAX_CONTENT_JOBS uploads
        self.executor: Final = ThreadPoolExecutor(MAX_SERIALIZATIONS, thread_name_prefix="lens-content")
        self.pending_bytes = 0
        self.closing = False

    def _inline_limit(self, span: RecordingSpan) -> int | None:
        return span_limits(span).max_span_attribute_length

    def model_admission(
        self, span: RecordingSpan, messages_in: Sequence[object], outputs: Sequence[object]
    ) -> int | Unavailable | None:
        """None when the model call's content stays inline, else the bytes to reserve or why it cannot be stored."""
        cap: Final = self._inline_limit(span)
        inline: Final = isinstance(check_size((messages_in, outputs), self.limits.inline_bytes), int) and (
            cap is None
            or all(
                not isinstance(serialized_messages(messages, cap), Unavailable) for messages in (messages_in, outputs)
            )
        )
        return None if inline else self._admission((messages_in, outputs))

    def text_admission(self, span: RecordingSpan, texts: Sequence[str]) -> int | Unavailable | None:
        """None when these attribute values stay inline, else the bytes to reserve or why they cannot be stored."""
        cap: Final = self._inline_limit(span)
        inline: Final = isinstance(check_size(tuple(texts), self.limits.inline_bytes), int) and (
            cap is None or all(len(text) <= cap for text in texts)
        )
        return None if inline else self._admission(tuple(texts))

    def _admission(self, content: object) -> int | Unavailable:
        measured: Final = check_size(content, self.limits.object_bytes)
        if measured is ExportFailure.TOO_LARGE:
            return Unavailable.TOO_LARGE
        if isinstance(measured, ExportFailure):
            return Unavailable.UNSERIALIZABLE
        retained: Final = _retained_bytes(content, MAX_PENDING_CONTENT_BYTES)
        return Unavailable.OVERLOADED if retained is None else retained

    def capture(
        self, span: RecordingSpan, identity: RequestIdentity, admission: int | Unavailable, work: ContentWork
    ) -> bool:
        """Takes the open span's content; False leaves the content on the span, as when nothing will export it."""
        if not span.is_recording():
            return False
        context: Final = span.get_span_context()
        owner: Final = Owner(
            team_id=identity.team_id or "",
            api_key=identity.key_hash or "",
            trace_id=format_trace_id(context.trace_id),
            span_id=format_span_id(context.span_id),
        )
        key: Final = (context.trace_id, context.span_id)
        admitted: Final = self._admit(key, owner, admission)
        if isinstance(admitted, Unavailable):
            span.set_attribute(REFERENCE_ATTRIBUTE, self._reference(owner, admitted))
            return True
        start_ms: Final = (span.start_time or 0) // 1_000_000
        try:
            self.loop.call_soon_threadsafe(self._start, key, admitted, start_ms, work)
        except RuntimeError:
            self._unstarted(key, admitted)
        return True

    def _admit(self, key: _SpanKey, owner: Owner, admission: int | Unavailable) -> _Capture | Unavailable:
        if isinstance(admission, Unavailable):
            return admission
        with self.lock:
            if (
                self.closing
                or len(self.captures) >= MAX_CONTENT_JOBS
                or self.pending_bytes + admission > MAX_PENDING_CONTENT_BYTES
            ):
                return Unavailable.OVERLOADED
            self.pending_bytes += admission
            capture: Final = _Capture(owner=owner, raw=admission)
            self.captures[key] = capture
        return capture

    def _unstarted(self, key: _SpanKey, capture: _Capture) -> None:
        """A capture whose content never reached a serializer gives back its raw reservation itself."""
        with self.lock:
            submitted: Final = capture.submitted
        if not submitted:
            self._release(capture.raw)
            self._settle(key, Unavailable.CANCELLED)

    def _start(self, key: _SpanKey, capture: _Capture, start_ms: int, work: ContentWork) -> None:
        try:
            task: Final = self.loop.create_task(self._store(key, capture, start_ms, work))
        except RuntimeError:
            self._unstarted(key, capture)
            return
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        task.add_done_callback(lambda _: self._unstarted(key, capture))

    async def _store(self, key: _SpanKey, capture: _Capture, start_ms: int, work: ContentWork) -> None:
        try:
            state: Final = await self._stored(capture, start_ms, work)
        except asyncio.CancelledError:
            self._settle(key, Unavailable.CANCELLED)
            return
        self._settle(key, state)

    async def _stored(self, capture: _Capture, start_ms: int, work: ContentWork) -> ContentState:
        try:
            encoding: Final[Future[_Encoded | Unavailable]] = self.executor.submit(
                _encoded, work, self.limits.object_bytes
            )
        except RuntimeError:
            return Unavailable.CANCELLED
        with self.lock:
            capture.submitted = True
        encoding.add_done_callback(lambda _: self._release(capture.raw))
        owner: Final = capture.owner
        encoded: Final = await asyncio.wrap_future(encoding)
        if isinstance(encoded, Unavailable):
            return encoded
        name: Final = object_name(owner, start_ms, encoded.sha256)
        if name is None:
            return Unavailable.UNSERIALIZABLE
        object_key: Final = self.store.object_key(name)
        if object_key is None:
            return Unavailable.STORAGE_NOT_CONFIGURED
        try:
            upload: Final = asyncio.ensure_future(self.store.upload(object_key, encoded.body))
        except Exception:  # noqa: BLE001  # a store that cannot start an upload leaves the content unavailable
            return Unavailable.UPLOAD_FAILED
        self._reserve(encoded.size)
        upload.add_done_callback(partial(self._uploaded, encoded.size))
        try:
            stored: Final = await asyncio.wait_for(asyncio.shield(upload), self.upload_timeout)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001  # a timed out or failed upload leaves the content unavailable
            return Unavailable.UPLOAD_FAILED
        if stored is not True:
            return Unavailable.UPLOAD_FAILED
        return Stored(object_key=object_key, size=encoded.size, sha256=encoded.sha256)

    def _reserve(self, size: int) -> None:
        with self.lock:
            self.pending_bytes += size

    def _release(self, size: int) -> None:
        with self.lock:
            self.pending_bytes -= size

    def _uploaded(self, size: int, upload: "asyncio.Future[bool]") -> None:
        self._release(size)
        if not upload.cancelled():
            _ = upload.exception()

    def _settle(self, key: _SpanKey, state: ContentState) -> None:
        """Records the outcome and exports the span if it already ended; a held span gives back its bytes then."""
        with self.lock:
            capture: Final = self.captures.get(key)
            if capture is None or capture.state is not None:
                return
            capture.state = state
            ended: Final = capture.ended
            forward: Final = capture.forward
            if ended is not None:
                del self.captures[key]
                self.pending_bytes -= capture.held
        if ended is not None and forward is not None:
            forward(self._referenced(ended, capture.owner, state))

    def route(self, span: ReadableSpan, forward: Callable[[ReadableSpan], None]) -> None:
        context: Final = span.context
        key: Final = (context.trace_id, context.span_id) if context is not None else (0, 0)
        with self.lock:
            capture: Final = self.captures.get(key)
            if capture is not None and capture.state is None:
                capture.ended = span
                capture.forward = forward
                capture.held = _retained_bytes(span.attributes or {}, MAX_PENDING_CONTENT_BYTES) or 0
                self.pending_bytes += capture.held
                return
            if capture is not None:
                del self.captures[key]
        if capture is None or capture.state is None:
            forward(span)
            return
        forward(self._referenced(span, capture.owner, capture.state))

    def _reference(self, owner: Owner, state: ContentState) -> str:
        return json.dumps(signed_reference(self.signing_key, owner, state), separators=(",", ":"))

    def _referenced(self, span: ReadableSpan, owner: Owner, state: ContentState) -> ReadableSpan:
        attributes: Final = MappingProxyType(
            {**(span.attributes or {}), REFERENCE_ATTRIBUTE: self._reference(owner, state)}
        )
        return CountedSpanView(span, attributes)

    def processor(self, inner: SpanProcessor) -> SpanProcessor:
        return _ContentSpanProcessor(inner, self)

    async def aclose(self) -> None:
        with self.lock:
            self.closing = True
        running: Final = tuple(self.tasks)
        for task in running:
            _ = task.cancel()
        _ = await asyncio.gather(*running, return_exceptions=True)
        self.executor.shutdown(wait=False, cancel_futures=True)


class _ContentSpanProcessor(SpanProcessor):
    def __init__(self, inner: SpanProcessor, spool: TraceContentSpool) -> None:
        self.inner: Final = inner
        self.spool: Final = spool

    @override
    def on_start(self, span: RecordingSpan, parent_context: Context | None = None) -> None:
        self.inner.on_start(span, parent_context)

    @override
    def on_end(self, span: ReadableSpan) -> None:
        self.spool.route(span, self.inner.on_end)

    @override
    def shutdown(self) -> None:
        self.inner.shutdown()

    @override
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.inner.force_flush(timeout_millis)


def _batches(spans: Sequence[ReadableSpan], limit: int) -> Iterable[tuple[ReadableSpan, ...]]:
    batch: list[ReadableSpan] = []  # mutable-ok: one batch under the byte limit at a time
    size = 0  # rebind-ok: bytes of the batch being filled
    for span, encoded in zip(spans, (encode_spans((span,)).ByteSize() for span in spans), strict=True):
        if batch and size + encoded > limit:
            yield tuple(batch)
            batch.clear()
            size = 0
        batch.append(span)
        size += encoded
    if batch:
        yield tuple(batch)


class BoundedBatchExporter(SpanExporter):
    """Sends a batch as requests of at most ``limit`` encoded bytes, so a few large spans cannot make Lens reject
    the small ones exported alongside them. A single span over the limit still goes alone."""

    def __init__(self, inner: SpanExporter, limit: int = MAX_EXPORT_BATCH_BYTES) -> None:
        self.inner: Final = inner
        self.limit: Final = limit

    @override
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        results: Final = tuple(self.inner.export(batch) for batch in _batches(spans, self.limit))
        return (
            SpanExportResult.SUCCESS
            if all(result is SpanExportResult.SUCCESS for result in results)
            else SpanExportResult.FAILURE
        )

    @override
    def shutdown(self) -> None:
        self.inner.shutdown()

    @override
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.inner.force_flush(timeout_millis)
