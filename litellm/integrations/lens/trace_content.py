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
from typing_extensions import override

from litellm.integrations.lens.content import (
    MAX_CONTENT_JOBS,
    MAX_NATIVE_UPLOADS,
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
from litellm.integrations.lens.exporter import (
    _is_mapping as is_mapping,  # pyright: ignore[reportPrivateUsage]  # the exporter's container guards
)
from litellm.integrations.lens.exporter import (
    _is_sequence as is_sequence,  # pyright: ignore[reportPrivateUsage]  # the exporter's container guards
)
from litellm.integrations.otel.model.metadata import RequestIdentity
from litellm.integrations.otel.plumbing.providers import (
    _CountedSpanView as CountedSpanView,  # pyright: ignore[reportPrivateUsage]  # the plumbing's span view keeps dropped counts
)
from litellm.integrations.otel.plumbing.providers import span_limits

MAX_EXPORT_BATCH_BYTES: Final = 16 * 1024 * 1024
DEFAULT_CONTENT_LIMITS: Final = ContentLimits()
MAX_REFUSALS: Final = 256
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


def _stringified(message: object, limit: int) -> tuple[object, int] | None | Unavailable:
    """One message as ``stringify_message`` round-trips it, with its text length, never holding more than ``limit``
    characters of that text."""
    if not is_mapping(message) or not isinstance(message, dict):
        return None
    try:
        text: Final = _bounded_json(message, limit, default=str, ascii_only=True)
    except Exception:  # noqa: BLE001  # mirrors stringify_message, which drops a message it cannot encode
        return None
    if text is None:
        return Unavailable.TOO_LARGE
    loaded: Final[object] = json.loads(text)  # pyright: ignore[reportAny]  # untyped JSON is re-encoded unchanged
    return loaded, len(text)


def serialized_messages(messages: Sequence[object], limit: int) -> str | None | Unavailable:
    """Exactly ``serialize_messages(messages)``, bounded to ``limit`` characters across all messages while it is
    built. The messages' texts add up to the final text minus its brackets and separators, so the shared budget
    stops at the first message that cannot fit. One escaped string is encoded whole before it is measured, so the
    transient text is at most 12 characters per character of the string being encoded."""
    remaining = limit  # rebind-ok: one budget shared by every message
    kept: Final[list[object]] = []  # mutable-ok: the round-tripped messages that fit the shared budget
    for message in messages:
        match _stringified(message, remaining):
            case Unavailable() as failure:
                return failure
            case (value, length):
                kept.append(value)
                remaining -= length
            case None:
                pass
    if not kept:
        return None
    text: Final = _bounded_json(kept, limit, default=None, ascii_only=True)
    return Unavailable.TOO_LARGE if text is None else text


def _kept_messages(messages: Sequence[object]) -> tuple[object, ...]:
    return tuple(message for message in messages if is_mapping(message) and isinstance(message, dict))


def _string_chars(value: object, budget: int, depth: int = 0) -> int | None:
    """Characters in every string of ``value``, keys included, or None once they pass ``budget``. JSON never
    shortens a string, so content past this lower bound cannot serialize within the budget."""
    if isinstance(value, str):
        return len(value) if len(value) <= budget else None
    if depth > _MAX_DEPTH:
        return None
    total = 0  # rebind-ok: consumes a finite budget across children
    for child in _children(value):
        match _string_chars(child, budget - total, depth + 1):
            case None:
                return None
            case int() as counted:
                total += counted
    return total


def _children(value: object) -> Iterable[object]:
    if is_mapping(value):
        return chain.from_iterable(value.items())
    if is_sequence(value):
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
    if body is None or data is None or len(data) > limit:
        return Unavailable.TOO_LARGE
    return _Encoded(body=body, size=len(data), sha256=hashlib.sha256(data).hexdigest())


class _Capture:
    """One admitted span's content job. Its raw reservation lasts until every holder of the content let go:
    the upload task, and the serializer thread once it was handed the content."""

    def __init__(self, owner: Owner, raw: int) -> None:
        self.owner: Final = owner
        self.raw: Final = raw
        self.holders = 1
        self.held = 0
        self.state: ContentState | None = None
        self.ended: ReadableSpan | None = None
        self.forward: Callable[[ReadableSpan], None] | None = None

    def hold(self) -> None:
        self.holders += 1

    def drop(self) -> bool:
        """Lets go of one holder; True for the last one."""
        self.holders -= 1
        return self.holders == 0

    def end(self, span: ReadableSpan, forward: Callable[[ReadableSpan], None], held: int) -> None:
        self.ended = span
        self.forward = forward
        self.held = held

    def settle(self, state: ContentState) -> None:
        self.state = state


class TraceContentSpool:
    """Content too large to ride inline on a Lens span, stored before the span is exported.

    Content is taken from the span while it is still open, before any attribute limit applies. A span
    whose content is being stored is held at its end and exported once the upload is acknowledged or
    fails, carrying a signed reference where the content was. The reference goes on the exported view,
    so no attribute limit can cut it. Spans without a capture pass straight through.
    """

    def __init__(
        self,
        store: ContentStore,
        signing_key: bytes,
        loop: asyncio.AbstractEventLoop,
        limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
        upload_timeout: float = UPLOAD_TIMEOUT_SECONDS,
        max_pending_bytes: int = MAX_PENDING_CONTENT_BYTES,
        max_jobs: int = MAX_CONTENT_JOBS,
        max_native_uploads: int = MAX_NATIVE_UPLOADS,
    ) -> None:
        self.store: Final = store
        self.signing_key: Final = signing_key
        self.loop: Final = loop
        self.limits: Final = limits
        self.upload_timeout: Final = upload_timeout
        self.max_pending_bytes: Final = max_pending_bytes
        self.max_jobs: Final = max_jobs
        self.max_native_uploads: Final = max_native_uploads
        self.lock: Final = threading.Lock()
        self.captures: Final[dict[_SpanKey, _Capture]] = {}  # mutable-ok: spans of at most MAX_CONTENT_JOBS jobs
        self.refusals: Final[dict[_SpanKey, tuple[Owner, Unavailable]]] = {}  # mutable-ok: at most MAX_REFUSALS
        self.tasks: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: at most MAX_CONTENT_JOBS uploads
        self.executor: Final = ThreadPoolExecutor(MAX_SERIALIZATIONS, thread_name_prefix="lens-content")
        self.jobs = 0
        self.native_uploads = 0
        self.pending_bytes = 0
        self.closing = False

    def _threshold(self, span: RecordingSpan) -> int:
        cap: Final = span_limits(span).max_span_attribute_length
        return self.limits.inline_bytes if cap is None else min(self.limits.inline_bytes, cap)

    def model_admission(
        self, span: RecordingSpan, messages_in: Sequence[object], outputs: Sequence[object]
    ) -> int | Unavailable | None:
        """None when the call's content stays inline exactly as today, else the bytes to reserve or why not."""
        threshold: Final = self._threshold(span)
        content: Final = (messages_in, outputs)
        floors: Final = tuple(_string_chars(_kept_messages(messages), threshold) for messages in content)
        if None in floors or sum(floor or 0 for floor in floors) > self.limits.inline_bytes:
            return self._admission(content)
        sent: Final = serialized_messages(messages_in, threshold)
        sent_size: Final = len(sent) if isinstance(sent, str) else 0
        answer: Final = (
            Unavailable.TOO_LARGE
            if sent is Unavailable.TOO_LARGE
            else serialized_messages(outputs, min(threshold, self.limits.inline_bytes - sent_size))
        )
        inline: Final = sent is not Unavailable.TOO_LARGE and answer is not Unavailable.TOO_LARGE
        return None if inline else self._admission(content)

    def text_admission(self, span: RecordingSpan, texts: Sequence[str]) -> int | Unavailable | None:
        """None when these attribute values stay inline exactly as today, else the bytes to reserve or why not."""
        threshold: Final = self._threshold(span)
        inline: Final = isinstance(check_size(tuple(texts), self.limits.inline_bytes), int) and all(
            len(text) <= threshold for text in texts
        )
        return None if inline else self._admission(tuple(texts))

    def _admission(self, content: object) -> int | Unavailable:
        measured: Final = check_size(content, self.limits.object_bytes)
        if measured is ExportFailure.TOO_LARGE:
            return Unavailable.TOO_LARGE
        if isinstance(measured, ExportFailure):
            return Unavailable.UNSERIALIZABLE
        retained: Final = _retained_bytes(content, self.max_pending_bytes)
        return Unavailable.OVERLOADED if retained is None else retained

    def capture(
        self, span: RecordingSpan, identity: RequestIdentity, admission: int | Unavailable, work: ContentWork
    ) -> bool:
        """Takes the open span's content; False leaves the content on the span, as when nothing will export it."""
        if not span.is_recording():
            return False
        context: Final = span.get_span_context()
        key: Final = (context.trace_id, context.span_id)
        owner: Final = Owner(
            team_id=identity.team_id or "",
            api_key=identity.key_hash or "",
            trace_id=format_trace_id(context.trace_id),
            span_id=format_span_id(context.span_id),
        )
        admitted: Final = self._admit(key, owner, admission)
        if isinstance(admitted, Unavailable):
            self._refuse(span, key, owner, admitted)
            return True
        start_ms: Final = (span.start_time or 0) // 1_000_000
        try:
            self.loop.call_soon_threadsafe(self._start, key, admitted, start_ms, work)
        except RuntimeError:
            self._drop(admitted)
            self._settle(key, Unavailable.CANCELLED)
        return True

    def _admit(self, key: _SpanKey, owner: Owner, admission: int | Unavailable) -> _Capture | Unavailable:
        if isinstance(admission, Unavailable):
            return admission
        with self.lock:
            if self.closing or self.jobs >= self.max_jobs or self.pending_bytes + admission > self.max_pending_bytes:
                return Unavailable.OVERLOADED
            self.jobs += 1
            self.pending_bytes += admission
            capture: Final = _Capture(owner, admission)
            self.captures[key] = capture
        return capture

    def _refuse(self, span: RecordingSpan, key: _SpanKey, owner: Owner, reason: Unavailable) -> None:
        """Marks a span whose content will not be stored. The mark goes on its exported view while there is room
        to remember it, else on the open span itself, where an attribute limit may still cut it."""
        with self.lock:
            remembered: Final = len(self.refusals) < MAX_REFUSALS
            if remembered:
                self.refusals[key] = (owner, reason)
        if not remembered:
            span.set_attribute(REFERENCE_ATTRIBUTE, self._reference(owner, reason))

    def _drop(self, capture: _Capture) -> None:
        with self.lock:
            if capture.drop():
                self.jobs -= 1
                self.pending_bytes -= capture.raw

    def _start(self, key: _SpanKey, capture: _Capture, start_ms: int, work: ContentWork) -> None:
        try:
            task: Final = self.loop.create_task(self._store(key, capture, start_ms, work))
        except RuntimeError:
            self._drop(capture)
            self._settle(key, Unavailable.CANCELLED)
            return
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        task.add_done_callback(lambda _: self._drop(capture))

    async def _store(self, key: _SpanKey, capture: _Capture, start_ms: int, work: ContentWork) -> None:
        try:
            state: Final = await self._stored(capture, start_ms, work)
        except asyncio.CancelledError:
            self._settle(key, Unavailable.CANCELLED)
            return
        self._settle(key, state)

    async def _stored(self, capture: _Capture, start_ms: int, work: ContentWork) -> ContentState:
        with self.lock:
            capture.hold()
        try:
            encoding: Final[Future[_Encoded | Unavailable]] = self.executor.submit(
                _encoded, work, self.limits.object_bytes
            )
        except RuntimeError:
            self._drop(capture)
            return Unavailable.CANCELLED
        encoding.add_done_callback(lambda _: self._drop(capture))
        encoded: Final = await asyncio.wrap_future(encoding)
        if isinstance(encoded, Unavailable):
            return encoded
        name: Final = object_name(capture.owner, start_ms, encoded.sha256)
        if name is None:
            return Unavailable.UNSERIALIZABLE
        object_key: Final = self.store.object_key(name)
        if object_key is None:
            return Unavailable.STORAGE_NOT_CONFIGURED
        held: Final = sys.getsizeof(encoded.body) + encoded.size
        if not self._admit_upload(held):
            return Unavailable.OVERLOADED
        try:
            upload: Final = asyncio.ensure_future(self.store.upload(object_key, encoded.body))
        except Exception:  # noqa: BLE001  # a store that cannot start an upload leaves the content unavailable
            self._uploaded_bytes(held)
            return Unavailable.UPLOAD_FAILED
        upload.add_done_callback(partial(self._uploaded, held))
        try:
            stored: Final = await asyncio.wait_for(asyncio.shield(upload), self.upload_timeout)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001  # a timed out or failed upload leaves the content unavailable
            return Unavailable.UPLOAD_FAILED
        if stored is not True:
            return Unavailable.UPLOAD_FAILED
        return Stored(object_key=object_key, size=encoded.size, sha256=encoded.sha256)

    def _admit_upload(self, size: int) -> bool:
        """Reserves one native upload and its body: the Python text and the UTF-8 copy the native client holds
        until it finishes."""
        with self.lock:
            if self.native_uploads >= self.max_native_uploads or self.pending_bytes + size > self.max_pending_bytes:
                return False
            self.native_uploads += 1
            self.pending_bytes += size
        return True

    def _uploaded_bytes(self, size: int) -> None:
        with self.lock:
            self.native_uploads -= 1
            self.pending_bytes -= size

    def _uploaded(self, size: int, upload: "asyncio.Future[bool]") -> None:
        self._uploaded_bytes(size)
        if not upload.cancelled():
            _ = upload.exception()

    def _settle(self, key: _SpanKey, state: ContentState) -> None:
        """Records the outcome and exports the span if it already ended, giving back the bytes it held."""
        with self.lock:
            capture: Final = self.captures.get(key)
            if capture is None or capture.state is not None:
                return
            capture.settle(state)
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
            refusal: Final = self.refusals.pop(key, None)
            capture: Final = self.captures.get(key)
            held: Final = (
                _retained_bytes(span.attributes or {}, self.max_pending_bytes - self.pending_bytes)
                if capture is not None and capture.state is None
                else None
            )
            if capture is not None and capture.state is None and held is not None:
                capture.end(span, forward, held)
                self.pending_bytes += held
                return
            if capture is not None:
                del self.captures[key]
        if refusal is not None:
            forward(self._referenced(span, *refusal))
        elif capture is None:
            forward(span)
        else:
            forward(self._referenced(span, capture.owner, capture.state or Unavailable.OVERLOADED))

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
    batch: Final[list[ReadableSpan]] = []  # mutable-ok: one batch under the byte limit at a time
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
    the small ones exported alongside them.

    A single span over the limit is still sent, alone, exactly as before: without configured content storage a
    large inline span is the only copy of its content, and Lens accepts requests up to its own body limit.
    """

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
