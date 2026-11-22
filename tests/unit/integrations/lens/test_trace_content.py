import asyncio
import json
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Final

import pytest
from opentelemetry.sdk.trace import ReadableSpan, Span, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import format_span_id, format_trace_id
from pydantic import TypeAdapter

from litellm.integrations.lens.content import (
    MAX_CONTENT_JOBS,
    REFERENCE_ATTRIBUTE,
    ContentLimits,
    Owner,
    StoredReference,
    Unavailable,
    content_signing_key,
    signed_reference,
    verified_stored,
)
from litellm.integrations.lens.trace_content import BoundedBatchExporter, TraceContentSpool, serialized_messages
from litellm.integrations.otel.mappers.utils import serialize_messages
from litellm.integrations.otel.model.config import OpenTelemetryV2Config
from litellm.integrations.otel.model.metadata import RequestIdentity
from litellm.integrations.otel.plumbing.providers import build_tracer_provider

_OBJECT: Final = TypeAdapter(dict[str, object])
_TOKEN: Final = "test-service-token-with-32-characters"
_KEY: Final = content_signing_key(_TOKEN)
_IDENTITY: Final = RequestIdentity(team_id="team-a", key_hash="hash-a")
_LIMITS: Final = ContentLimits(inline_bytes=1024, object_bytes=64 * 1024)
_CONTENT: Final = {"gen_ai.input.messages": '[{"content": "' + "界" * 600 + '"}]'}
_MIXED: Final = {"gen_ai.input.messages": "a" * 3000 + "\U0001f600"}


class _Store:
    def __init__(self) -> None:
        self.uploads: Final = asyncio.Queue[tuple[str, str, "asyncio.Future[bool]"]]()

    def object_key(self, name: str) -> str | None:
        return f"logs/{name}"

    def upload(self, key: str, body: str) -> "asyncio.Future[bool]":
        acked: Final = asyncio.get_running_loop().create_future()
        self.uploads.put_nowait((key, body, acked))
        return acked

    def download(self, key: str, max_bytes: int) -> "asyncio.Future[str | None]":
        raise AssertionError("the producer never reads content back")


async def _acknowledge(acked: "asyncio.Future[bool]") -> None:
    """Acknowledges an upload and returns once every callback the spool put on it has run."""
    settled: Final = asyncio.Event()
    acked.add_done_callback(lambda _: settled.set())
    acked.set_result(True)
    await settled.wait()


class _Exported(InMemorySpanExporter):
    """Keeps every exported span and announces each by name, so a test waits for the export it expects."""

    def __init__(self) -> None:
        super().__init__()
        self.names: Final = asyncio.Queue[str]()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            self.names.put_nowait(span.name)
        return super().export(spans)


class _Pipeline:
    def __init__(self, spool: TraceContentSpool, provider: TracerProvider, exported: "_Exported") -> None:
        self.spool: Final = spool
        self.provider: Final = provider
        self.exported: Final = exported

    def start(self, name: str = "chat model") -> Span:
        span: Final = self.provider.get_tracer("lens-test").start_span(name, attributes={"litellm.team.id": "team-a"})
        assert isinstance(span, Span)
        return span

    def finished(self) -> dict[str, ReadableSpan]:
        return {span.name: span for span in self.exported.get_finished_spans()}


@pytest.fixture
def store() -> _Store:
    return _Store()


@asynccontextmanager
async def _pipeline(
    store: _Store, upload_timeout: float = 5, **bounds: int
) -> AsyncGenerator[_Pipeline, None]:  # kwargs-ok: the spool's bounds
    spool: Final = TraceContentSpool(
        store, _KEY, asyncio.get_running_loop(), _LIMITS, upload_timeout=upload_timeout, **bounds
    )
    exported: Final = _Exported()
    provider: Final = build_tracer_provider(
        OpenTelemetryV2Config(exporter="in_memory"),
        exporter=exported,
        use_simple_processor=True,
        wrap_processor=spool.processor,
    )
    try:
        yield _Pipeline(spool, provider, exported)
    finally:
        await spool.aclose()
        provider.shutdown()


@pytest.fixture
async def pipeline(store: _Store) -> AsyncIterator[_Pipeline]:
    async with _pipeline(store) as built:
        yield built


def _owner(span: ReadableSpan) -> Owner:
    context: Final = span.get_span_context()
    assert context is not None
    return Owner(
        team_id="team-a",
        api_key="hash-a",
        trace_id=format_trace_id(context.trace_id),
        span_id=format_span_id(context.span_id),
    )


def _reference(span: ReadableSpan) -> dict[str, object]:
    raw: Final = (span.attributes or {})[REFERENCE_ATTRIBUTE]
    assert isinstance(raw, str)
    return _OBJECT.validate_json(raw)


def _capture(pipeline: _Pipeline, name: str = "chat model", content: dict[str, str] = _CONTENT) -> Span:
    span: Final = pipeline.start(name)
    admission: Final = pipeline.spool.text_admission(span, tuple(content.values()))
    assert admission is not None
    assert pipeline.spool.capture(span, _IDENTITY, admission, lambda limit: content)
    span.end()
    return span


@pytest.mark.asyncio
async def test_a_captured_span_is_exported_only_after_its_upload_is_acknowledged(
    pipeline: _Pipeline, store: _Store
) -> None:
    span: Final = _capture(pipeline)
    key, body, acked = await asyncio.wait_for(store.uploads.get(), timeout=5)
    assert pipeline.finished() == {}
    assert json.loads(body) == _CONTENT
    acked.set_result(True)
    await asyncio.gather(*pipeline.spool.tasks)
    exported: Final = pipeline.finished()["chat model"]
    reference: Final = StoredReference.model_validate(_reference(exported))
    stored: Final = verified_stored(_KEY, _owner(span), reference)
    assert stored is not None and stored.object_key == key and stored.size == len(body.encode())
    assert (exported.attributes or {})["litellm.team.id"] == "team-a"
    other_span: Final = Owner(team_id="team-a", api_key="hash-a", trace_id=_owner(span).trace_id, span_id="0" * 16)
    assert verified_stored(_KEY, other_span, reference) is None
    assert pipeline.spool.pending_bytes == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "reason"),
    (
        pytest.param(False, Unavailable.UPLOAD_FAILED, id="rejected"),
        pytest.param(OSError("connection reset"), Unavailable.UPLOAD_FAILED, id="raised"),
    ),
)
async def test_a_failed_upload_exports_the_span_as_unavailable_with_its_metadata(
    pipeline: _Pipeline, store: _Store, outcome: bool | OSError, reason: Unavailable
) -> None:
    span: Final = _capture(pipeline)
    _, _, acked = await asyncio.wait_for(store.uploads.get(), timeout=5)
    if isinstance(outcome, OSError):
        acked.set_exception(outcome)
    else:
        acked.set_result(outcome)
    await asyncio.gather(*pipeline.spool.tasks)
    exported: Final = pipeline.finished()["chat model"]
    assert _reference(exported) == dict(signed_reference(_KEY, _owner(span), reason))
    assert (exported.attributes or {})["litellm.team.id"] == "team-a"
    assert pipeline.spool.pending_bytes == 0


@pytest.mark.asyncio
async def test_an_upload_past_its_timeout_is_unavailable_and_keeps_its_bytes_until_the_native_upload_ends(
    store: _Store,
) -> None:
    async with _pipeline(store, upload_timeout=0) as pipeline:
        span: Final = _capture(pipeline, content=_MIXED)
        _, body, acked = await asyncio.wait_for(store.uploads.get(), timeout=5)
        await asyncio.gather(*pipeline.spool.tasks)
        assert _reference(pipeline.finished()["chat model"]) == dict(
            signed_reference(_KEY, _owner(span), Unavailable.UPLOAD_FAILED)
        )
        assert (pipeline.spool.native_uploads, pipeline.spool.pending_bytes) == (
            1,
            sys.getsizeof(body) + len(body.encode()),
        )
        await _acknowledge(acked)
        assert (pipeline.spool.native_uploads, pipeline.spool.pending_bytes) == (0, 0)


@pytest.mark.asyncio
async def test_shutdown_exports_held_spans_as_cancelled_and_refuses_new_captures(
    pipeline: _Pipeline, store: _Store
) -> None:
    span: Final = _capture(pipeline)
    _, _, acked = await asyncio.wait_for(store.uploads.get(), timeout=5)
    await pipeline.spool.aclose()
    assert _reference(pipeline.finished()["chat model"]) == dict(
        signed_reference(_KEY, _owner(span), Unavailable.CANCELLED)
    )
    late: Final = _capture(pipeline, "late call")
    assert _reference(pipeline.finished()["late call"]) == dict(
        signed_reference(_KEY, _owner(late), Unavailable.OVERLOADED)
    )
    await _acknowledge(acked)
    assert (pipeline.spool.pending_bytes, pipeline.spool.captures) == (0, {})


@pytest.mark.asyncio
async def test_captures_past_the_job_limit_are_unavailable_without_growing_the_held_set(store: _Store) -> None:
    async with _pipeline(store, max_native_uploads=MAX_CONTENT_JOBS) as pipeline:
        for index in range(MAX_CONTENT_JOBS):
            _ = _capture(pipeline, f"call {index}")
        overflow: Final = _capture(pipeline, "overflow")
        assert (len(pipeline.spool.captures), pipeline.spool.jobs) == (MAX_CONTENT_JOBS, MAX_CONTENT_JOBS)
        assert _reference(pipeline.finished()["overflow"]) == dict(
            signed_reference(_KEY, _owner(overflow), Unavailable.OVERLOADED)
        )
        acks: Final = tuple([await asyncio.wait_for(store.uploads.get(), timeout=5) for _ in range(MAX_CONTENT_JOBS)])
        for _, _, acked in acks:
            acked.set_result(True)
        await asyncio.gather(*pipeline.spool.tasks)
        assert len(pipeline.finished()) == MAX_CONTENT_JOBS + 1
        assert (pipeline.spool.pending_bytes, pipeline.spool.captures, pipeline.spool.jobs) == (0, {}, 0)


@pytest.mark.asyncio
async def test_content_over_the_object_limit_is_unavailable_and_never_uploaded(
    pipeline: _Pipeline, store: _Store
) -> None:
    span: Final = pipeline.start()
    admission: Final = pipeline.spool.text_admission(span, ("x" * (_LIMITS.object_bytes + 1),))
    assert admission is Unavailable.TOO_LARGE
    assert pipeline.spool.capture(span, _IDENTITY, admission, lambda limit: _CONTENT)
    span.end()
    assert _reference(pipeline.finished()["chat model"]) == dict(
        signed_reference(_KEY, _owner(span), Unavailable.TOO_LARGE)
    )
    assert store.uploads.empty()


@pytest.mark.asyncio
async def test_spans_without_a_capture_pass_through_unchanged(pipeline: _Pipeline, store: _Store) -> None:
    span: Final = pipeline.start("small call")
    span.set_attribute("gen_ai.input.messages", '[{"content": "hi"}]')
    assert pipeline.spool.text_admission(span, ('[{"content": "hi"}]',)) is None
    span.end()
    exported: Final = pipeline.finished()["small call"]
    assert exported.attributes == span.attributes
    assert store.uploads.empty()


@pytest.mark.asyncio
async def test_the_sdk_attribute_length_limit_makes_content_it_would_cut_spill_and_shorter_content_stay(
    monkeypatch: pytest.MonkeyPatch, store: _Store
) -> None:
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "64")
    spool: Final = TraceContentSpool(store, _KEY, asyncio.get_running_loop(), _LIMITS)
    provider: Final = build_tracer_provider(
        OpenTelemetryV2Config(exporter="in_memory"), exporter=InMemorySpanExporter(), use_simple_processor=True
    )
    span: Final = provider.get_tracer("lens-test").start_span("chat")
    try:
        assert isinstance(span, Span)
        short: Final = ({"role": "user", "content": "hi"},)
        long: Final = ({"role": "user", "content": "界" * 20},)
        assert len(serialize_messages(short) or "") <= 64 < len(serialize_messages(long) or "")
        assert spool.model_admission(span, short, ()) is None
        assert isinstance(spool.model_admission(span, long, ()), int)
        assert spool.text_admission(span, ("x" * 64,)) is None
        assert isinstance(spool.text_admission(span, ("x" * 65,)), int)
    finally:
        span.end()
        await spool.aclose()
        provider.shutdown()


class _Opaque:
    def __str__(self) -> str:
        return "opaque-界"


@pytest.mark.parametrize(
    "messages",
    (
        pytest.param(({"role": "user", "content": '界\x01"quoted"'},), id="escapes-and-unicode"),
        pytest.param(({"role": "user", "content": _Opaque()}, "not-a-dict"), id="default-str-and-skipped"),
        pytest.param((), id="empty"),
    ),
)
def test_bounded_serialization_matches_the_inline_attribute_exactly(messages: Sequence[object]) -> None:
    assert serialized_messages(messages, 1 << 20) == serialize_messages(messages)


def test_bounded_serialization_stops_at_its_limit() -> None:
    text: Final = serialize_messages(({"content": "x" * 100},)) or ""
    assert serialized_messages(({"content": "x" * 100},), len(text)) == text
    assert serialized_messages(({"content": "x" * 100},), len(text) - 1) is Unavailable.TOO_LARGE


class _Recording(SpanExporter):
    def __init__(self, fail_after: int | None = None) -> None:
        self.batches: Final[list[tuple[str, ...]]] = []  # mutable-ok: records each request the wrapper sends
        self.fail_after: Final = fail_after

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.batches.append(tuple(span.name for span in spans))
        failed: Final = self.fail_after is not None and len(self.batches) > self.fail_after
        return SpanExportResult.FAILURE if failed else SpanExportResult.SUCCESS


def _ended(provider: TracerProvider, name: str, size: int) -> ReadableSpan:
    span: Final = provider.get_tracer("lens-test").start_span(name, attributes={"payload": "x" * size})
    span.end()
    assert isinstance(span, ReadableSpan)
    return span


def test_batches_are_split_by_encoded_bytes_so_large_spans_never_take_small_ones_down() -> None:
    provider: Final = TracerProvider()
    spans: Final = (
        _ended(provider, "small-1", 10),
        _ended(provider, "large", 5000),
        _ended(provider, "small-2", 10),
        _ended(provider, "small-3", 10),
    )
    inner: Final = _Recording()
    assert BoundedBatchExporter(inner, limit=1000).export(spans) is SpanExportResult.SUCCESS
    assert inner.batches == [("small-1",), ("large",), ("small-2", "small-3")]
    failing: Final = _Recording(fail_after=1)
    assert BoundedBatchExporter(failing, limit=1000).export(spans) is SpanExportResult.FAILURE
    assert len(failing.batches) == 3


@pytest.mark.asyncio
async def test_a_body_past_the_shared_budget_is_unavailable_and_keeps_the_span_metadata(store: _Store) -> None:
    content: Final = {"gen_ai.input.messages": "界" * 3000}
    async with _pipeline(store, max_pending_bytes=8000) as pipeline:
        span: Final = _capture(pipeline, content=content)
        await asyncio.gather(*pipeline.spool.tasks)
        exported: Final = pipeline.finished()["chat model"]
        assert _reference(exported) == dict(signed_reference(_KEY, _owner(span), Unavailable.OVERLOADED))
        assert (exported.attributes or {})["litellm.team.id"] == "team-a"
        assert (store.uploads.empty(), pipeline.spool.pending_bytes, pipeline.spool.jobs) == (True, 0, 0)


@pytest.mark.asyncio
async def test_a_span_whose_metadata_would_pass_the_budget_is_exported_at_once_instead_of_held(store: _Store) -> None:
    async with _pipeline(store, max_pending_bytes=3000) as pipeline:
        span: Final = pipeline.start()
        span.set_attribute("litellm.metadata.note", "x" * 5000)
        admission: Final = pipeline.spool.text_admission(span, tuple(_CONTENT.values()))
        assert isinstance(admission, int)
        assert pipeline.spool.capture(span, _IDENTITY, admission, lambda limit: _CONTENT)
        span.end()
        assert _reference(pipeline.finished()["chat model"]) == dict(
            signed_reference(_KEY, _owner(span), Unavailable.OVERLOADED)
        )
        await asyncio.gather(*pipeline.spool.tasks)
        assert len(pipeline.exported.get_finished_spans()) == 1
        assert (pipeline.spool.captures, pipeline.spool.pending_bytes, pipeline.spool.jobs) == ({}, 0, 0)


@pytest.mark.asyncio
async def test_native_uploads_past_their_limit_are_unavailable_while_the_first_still_runs(store: _Store) -> None:
    async with _pipeline(store, max_native_uploads=1) as pipeline:
        first: Final = _capture(pipeline, "first")
        _, _, acked = await asyncio.wait_for(store.uploads.get(), timeout=5)
        second: Final = _capture(pipeline, "second")
        assert await asyncio.wait_for(pipeline.exported.names.get(), timeout=5) == "second"
        assert _reference(pipeline.finished()["second"]) == dict(
            signed_reference(_KEY, _owner(second), Unavailable.OVERLOADED)
        )
        assert store.uploads.empty() and "first" not in pipeline.finished()
        await _acknowledge(acked)
        await asyncio.gather(*pipeline.spool.tasks)
        stored: Final = StoredReference.model_validate(_reference(pipeline.finished()["first"]))
        assert verified_stored(_KEY, _owner(first), stored) is not None
        assert (pipeline.spool.native_uploads, pipeline.spool.pending_bytes) == (0, 0)


@pytest.mark.asyncio
async def test_content_that_json_escaping_grows_past_the_inline_bytes_spills_without_an_sdk_limit(
    pipeline: _Pipeline,
) -> None:
    span: Final = pipeline.start()
    try:
        assert pipeline.spool.model_admission(span, ({"content": "a" * 300},), ()) is None
        assert isinstance(pipeline.spool.model_admission(span, ({"content": "\x01" * 300},), ()), int)
    finally:
        span.end()


def test_a_provider_built_without_lens_wrappers_exports_spans_exactly_as_before() -> None:
    exported: Final = InMemorySpanExporter()
    provider: Final = build_tracer_provider(
        OpenTelemetryV2Config(exporter="in_memory"), exporter=exported, use_simple_processor=True
    )
    span: Final = provider.get_tracer("other-otel").start_span("chat", attributes={"gen_ai.input.messages": "x" * 5000})
    span.end()
    provider.shutdown()
    assert [dict(finished.attributes or {}) for finished in exported.get_finished_spans()] == [
        {"gen_ai.input.messages": "x" * 5000}
    ]


@pytest.mark.asyncio
async def test_a_body_stored_four_bytes_a_character_is_charged_as_stored_not_as_its_utf8_size(store: _Store) -> None:
    async with _pipeline(store, max_pending_bytes=20_000) as pipeline:
        span: Final = _capture(pipeline, content=_MIXED)
        await asyncio.gather(*pipeline.spool.tasks)
        assert _reference(pipeline.finished()["chat model"]) == dict(
            signed_reference(_KEY, _owner(span), Unavailable.OVERLOADED)
        )
        assert (store.uploads.empty(), pipeline.spool.pending_bytes, pipeline.spool.native_uploads) == (True, 0, 0)


class _Counted:
    """A non-JSON value that counts how often serialization renders it."""

    def __init__(self) -> None:
        self.calls = 0

    def __str__(self) -> str:
        self.calls += 1
        return "1"


@pytest.mark.asyncio
async def test_content_whose_text_alone_passes_the_threshold_is_refused_before_it_is_serialized(
    store: _Store,
) -> None:
    async with _pipeline(store, max_pending_bytes=100) as pipeline:
        span: Final = pipeline.start()
        counted: Final = _Counted()
        try:
            messages: Final = ({"extra": counted, "content": "x" * 2000},)
            assert pipeline.spool.model_admission(span, messages, ()) is Unavailable.OVERLOADED
            assert counted.calls == 0
        finally:
            span.end()


@pytest.mark.asyncio
async def test_many_small_messages_stop_being_serialized_once_their_shared_budget_is_spent(
    pipeline: _Pipeline,
) -> None:
    span: Final = pipeline.start()
    counted: Final = _Counted()
    try:
        messages: Final = tuple({"n": counted} for _ in range(1000))
        assert isinstance(pipeline.spool.model_admission(span, messages, ()), int)
        assert 0 < counted.calls <= _LIMITS.inline_bytes // len('{"n": "1"}') + 1
    finally:
        span.end()
