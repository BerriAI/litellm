import asyncio
import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import Final, TypeAlias

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace import Span as RecordingSpan
from opentelemetry.trace import format_span_id, format_trace_id
from typing_extensions import override

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
from litellm.integrations.lens.exporter import _check_size as check_size  # pyright: ignore[reportPrivateUsage]  # one bounded size walk for every Lens payload
from litellm.integrations.otel.model.metadata import RequestIdentity
from litellm.integrations.otel.plumbing.providers import _CountedSpanView as CountedSpanView  # pyright: ignore[reportPrivateUsage]  # the plumbing's span view keeps dropped counts

ContentAttributes: TypeAlias = Mapping[str, str]
_SpanKey: TypeAlias = tuple[int, int]
_Admission: TypeAlias = int | Unavailable


@dataclass(frozen=True, slots=True)
class _Encoded:
    body: str
    size: int
    sha256: str


def _encoded(work: Callable[[], ContentAttributes], limit: int) -> _Encoded | Unavailable:
    try:
        body: Final = json.dumps(dict(work()), ensure_ascii=False, separators=(",", ":"))
        data: Final = body.encode()
    except (ValueError, TypeError, RecursionError, RuntimeError):
        return Unavailable.UNSERIALIZABLE
    if len(data) > limit:
        return Unavailable.TOO_LARGE
    return _Encoded(body=body, size=len(data), sha256=hashlib.sha256(data).hexdigest())


@dataclass(slots=True)
class _Capture:
    owner: Owner
    state: ContentState | None = None
    ended: ReadableSpan | None = None
    forward: Callable[[ReadableSpan], None] | None = None


@dataclass(slots=True)
class TraceContentSpool:
    """Content too large to ride inline on a Lens span, stored before the span is exported.

    A span whose content is captured is held at its end until its upload is acknowledged or fails,
    then exported carrying a signed reference in place of the content. Every other span passes through.
    """

    store: ContentStore
    signing_key: bytes
    loop: asyncio.AbstractEventLoop
    limits: ContentLimits = ContentLimits()
    upload_timeout: float = UPLOAD_TIMEOUT_SECONDS
    lock: threading.Lock = field(default_factory=threading.Lock)
    captures: dict[_SpanKey, _Capture] = field(default_factory=dict)  # mutable-ok: bounded spans awaiting content
    tasks: set["asyncio.Task[None]"] = field(default_factory=set)  # mutable-ok: bounded content uploads
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(MAX_SERIALIZATIONS, thread_name_prefix="lens-content")
    )
    pending_bytes: int = 0
    closing: bool = False

    def admission(self, content: object) -> _Admission | None:
        """None for content that stays inline, else the bytes to reserve or why it cannot be stored."""
        if isinstance(check_size(content, self.limits.inline_bytes), int):
            return None
        measured: Final = check_size(content, self.limits.object_bytes)
        if measured is ExportFailure.TOO_LARGE:
            return Unavailable.TOO_LARGE
        if isinstance(measured, ExportFailure):
            return Unavailable.UNSERIALIZABLE
        return self.limits.object_bytes - measured

    def capture(
        self,
        span: RecordingSpan,
        identity: RequestIdentity,
        admission: _Admission,
        work: Callable[[], ContentAttributes],
    ) -> bool:
        """Takes the span's content before the span ends; False when the span will not be exported at all."""
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
        start_ms: Final = (span.start_time or 0) // 1_000_000
        with self.lock:
            if isinstance(admission, Unavailable):
                self.captures[key] = _Capture(owner=owner, state=admission)
                return True
            if (
                self.closing
                or len(self.captures) >= MAX_CONTENT_JOBS
                or self.pending_bytes + admission > MAX_PENDING_CONTENT_BYTES
            ):
                self.captures[key] = _Capture(owner=owner, state=Unavailable.OVERLOADED)
                return True
            self.pending_bytes += admission
            self.captures[key] = _Capture(owner=owner)
        try:
            self.loop.call_soon_threadsafe(self._start, key, owner, start_ms, admission, work)
        except RuntimeError:
            self._release(admission)
            self._settle(key, Unavailable.CANCELLED)
        return True

    def _start(
        self, key: _SpanKey, owner: Owner, start_ms: int, admission: int, work: Callable[[], ContentAttributes]
    ) -> None:
        task: Final = self.loop.create_task(self._store(key, owner, start_ms, admission, work))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _store(
        self, key: _SpanKey, owner: Owner, start_ms: int, admission: int, work: Callable[[], ContentAttributes]
    ) -> None:
        try:
            state: Final = await self._stored(owner, start_ms, admission, work)
        except asyncio.CancelledError:
            self._settle(key, Unavailable.CANCELLED)
            return
        self._settle(key, state)

    async def _stored(
        self, owner: Owner, start_ms: int, admission: int, work: Callable[[], ContentAttributes]
    ) -> ContentState:
        try:
            encoding: Final[Future[_Encoded | Unavailable]] = self.executor.submit(
                _encoded, work, self.limits.object_bytes
            )
        except RuntimeError:
            self._release(admission)
            return Unavailable.CANCELLED
        encoding.add_done_callback(lambda _: self._release(admission))
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
        with self.lock:
            self.pending_bytes += encoded.size
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

    def _release(self, size: int) -> None:
        with self.lock:
            self.pending_bytes -= size

    def _uploaded(self, size: int, upload: "asyncio.Future[bool]") -> None:
        self._release(size)
        if not upload.cancelled():
            _ = upload.exception()

    def _settle(self, key: _SpanKey, state: ContentState) -> None:
        with self.lock:
            capture: Final = self.captures.get(key)
            if capture is None:
                return
            capture.state = state
            ended: Final = capture.ended
            forward: Final = capture.forward
            if ended is not None:
                del self.captures[key]
        if ended is not None and forward is not None:
            forward(self._referenced(ended, capture.owner, state))

    def route(self, span: ReadableSpan, forward: Callable[[ReadableSpan], None]) -> None:
        context: Final = span.context
        key: Final = (context.trace_id, context.span_id) if context is not None else None
        with self.lock:
            capture: Final = self.captures.get(key) if key is not None else None
            if capture is not None and key is not None:
                if capture.state is None:
                    capture.ended = span
                    capture.forward = forward
                    return
                del self.captures[key]
        if capture is None or capture.state is None:
            forward(span)
            return
        forward(self._referenced(span, capture.owner, capture.state))

    def _referenced(self, span: ReadableSpan, owner: Owner, state: ContentState) -> ReadableSpan:
        reference: Final = json.dumps(signed_reference(self.signing_key, owner, state), separators=(",", ":"))
        return CountedSpanView(span, MappingProxyType({**(span.attributes or {}), REFERENCE_ATTRIBUTE: reference}))

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
