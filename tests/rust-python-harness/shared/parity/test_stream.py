from __future__ import annotations

import queue
from collections.abc import AsyncIterator, Iterator
from typing import Final, Literal, NoReturn

import pytest
from pydantic import BaseModel, PrivateAttr, ValidationError

from .models import (
    SDKBytesChunk,
    SDKError,
    SDKJsonChunk,
    SDKReport,
    SDKStreamCompleted,
    SDKStreamFailed,
    SDKStreamReport,
    sdk_error_report,
)
from .stream import (
    StreamCompleted,
    StreamFailed,
    StreamOutcome,
    assert_stream_parity,
    capture_async_stream,
    capture_sync_stream,
    consume_async_stream,
    consume_sync_stream,
    drain_async_stream,
    drain_sync_stream,
)


class _Chunk(BaseModel):
    value: str
    _hidden_params: dict[str, object] = PrivateAttr(default_factory=dict)

    def set_hidden_param(self, key: str, value: object) -> None:
        self._hidden_params[key] = value


class _NestedChunk(BaseModel):
    value: object


class _SyncStream:
    def __init__(self, chunks: tuple[object, ...], error: BaseException | None = None) -> None:
        self.chunks: Final = chunks
        self.error: Final = error

    def __iter__(self) -> Iterator[object]:
        yield from self.chunks
        if self.error is not None:
            raise self.error


class _AsyncStream:
    def __init__(self, chunks: tuple[object, ...], error: BaseException | None = None) -> None:
        self.chunks: Final = chunks
        self.error: Final = error

    async def __aiter__(self) -> AsyncIterator[object]:
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


class _PublicStreamError(Exception):
    def __init__(
        self,
        message: str = "invalid input",
        *,
        status_code: int = 400,
        llm_provider: str = "test",
        model: str = "test-model",
        code: str = "invalid_input",
        error_type: str = "validation_error",
        param: str = "input",
    ) -> None:
        super().__init__(message)
        self.status_code: Final = status_code
        self.llm_provider: Final = llm_provider
        self.model: Final = model
        self.code: Final = code
        self.type: Final = error_type
        self.param: Final = param


def _creation_error() -> NoReturn:
    raise _PublicStreamError(status_code=429, llm_provider="test", model="test-model")


async def _async_stream(chunks: tuple[object, ...], error: BaseException | None = None) -> _AsyncStream:
    return _AsyncStream(chunks, error)


async def _consume(
    mode: Literal["sync", "async"], chunks: tuple[object, ...], error: Exception | None = None
) -> StreamOutcome:
    if mode == "sync":
        return consume_sync_stream(lambda: _SyncStream(chunks, error))
    return await consume_async_stream(lambda: _async_stream(chunks, error))


async def _capture(
    mode: Literal["sync", "async"], chunks: tuple[object, ...], error: Exception | None = None
) -> SDKReport:
    if mode == "sync":
        return capture_sync_stream(lambda: _SyncStream(chunks, error))
    return await capture_async_stream(lambda: _async_stream(chunks, error))


def test_sync_stream_parity_compares_chunks_and_ignores_private_attrs() -> None:
    python_chunk: Final = _Chunk(value="same")
    accelerated_chunk: Final = _Chunk(value="same")
    python_chunk.set_hidden_param("request_id", "python")
    accelerated_chunk.set_hidden_param("request_id", "accelerated")
    python: Final = consume_sync_stream(lambda: _SyncStream((python_chunk,)))
    accelerated: Final = consume_sync_stream(lambda: _SyncStream((accelerated_chunk,)))

    assert python.supports_sync_iteration is True
    assert python.supports_async_iteration is False
    assert_stream_parity(python, accelerated)


def test_stream_parity_rejects_extra_chunk() -> None:
    python: Final = consume_sync_stream(lambda: _SyncStream((_Chunk(value="one"),)))
    accelerated: Final = consume_sync_stream(lambda: _SyncStream((_Chunk(value="one"), _Chunk(value="two"))))

    with pytest.raises(AssertionError):
        assert_stream_parity(python, accelerated)


def test_stream_parity_rejects_chunk_value_difference() -> None:
    python: Final = consume_sync_stream(lambda: _SyncStream((_Chunk(value="python"),)))
    accelerated: Final = consume_sync_stream(lambda: _SyncStream((_Chunk(value="accelerated"),)))

    with pytest.raises(AssertionError):
        assert_stream_parity(python, accelerated)


def test_stream_outcome_distinguishes_creation_and_iteration_errors() -> None:
    creation: Final = consume_sync_stream(_creation_error)
    iteration: Final = consume_sync_stream(
        lambda: _SyncStream(
            (_Chunk(value="before-error"),),
            _PublicStreamError(status_code=429, llm_provider="test", model="test-model"),
        )
    )

    assert isinstance(creation.terminal, StreamFailed)
    assert creation.terminal.phase == "creation"
    assert creation.chunks == ()
    assert isinstance(iteration.terminal, StreamFailed)
    assert iteration.terminal.phase == "iteration"
    assert len(iteration.chunks) == 1
    with pytest.raises(AssertionError):
        assert_stream_parity(creation, iteration)


@pytest.mark.asyncio
async def test_async_stream_uses_same_trace_contract() -> None:
    python_error: Final = _PublicStreamError(
        "invalid input\nTraceback (most recent call last):\npython detail", status_code=500
    )
    accelerated_error: Final = _PublicStreamError(
        "invalid input\nTraceback (most recent call last):\nrust detail", status_code=500
    )
    python: Final = await consume_async_stream(lambda: _async_stream((_Chunk(value="same"),), python_error))
    accelerated: Final = await consume_async_stream(lambda: _async_stream((_Chunk(value="same"),), accelerated_error))

    assert python.supports_sync_iteration is False
    assert python.supports_async_iteration is True
    assert_stream_parity(python, accelerated)


def test_stream_parity_accepts_route_specific_chunk_normalizer() -> None:
    python: Final = consume_sync_stream(lambda: _SyncStream((_Chunk(value="python-generated-id"),)))
    accelerated: Final = consume_sync_stream(lambda: _SyncStream((_Chunk(value="rust-generated-id"),)))

    assert_stream_parity(python, accelerated, normalize=lambda chunk: type(chunk))


def test_drain_sync_stream_exhausts_lazy_iterator() -> None:
    consumed: Final[queue.SimpleQueue[str]] = queue.SimpleQueue()

    def chunks() -> Iterator[object]:
        yield _Chunk(value="one")
        consumed.put("complete")

    drain_sync_stream(chunks())

    assert consumed.get_nowait() == "complete"


@pytest.mark.asyncio
async def test_drain_async_stream_exhausts_lazy_iterator() -> None:
    consumed: Final[queue.SimpleQueue[str]] = queue.SimpleQueue()

    async def chunks() -> AsyncIterator[object]:
        yield b"one"
        consumed.put("complete")

    await drain_async_stream(chunks())

    assert consumed.get_nowait() == "complete"


def test_capture_sync_stream_serializes_model_chunks_and_partial_failure() -> None:
    report: Final = capture_sync_stream(
        lambda: _SyncStream(
            (_Chunk(value="before-error"),),
            _PublicStreamError(status_code=429, llm_provider="test", model="test-model"),
        )
    )

    assert isinstance(report, SDKStreamReport)
    assert len(report.chunks) == 1
    chunk: Final = report.chunks[0]
    assert isinstance(chunk, SDKJsonChunk)
    assert chunk.value == {"value": "before-error"}
    assert isinstance(report.terminal, SDKStreamFailed)
    assert report.terminal.error.status_code == 429


@pytest.mark.asyncio
async def test_capture_async_stream_serializes_message_bytes_in_order() -> None:
    report: Final = await capture_async_stream(lambda: _async_stream((b"first", b"second")))

    assert isinstance(report, SDKStreamReport)
    assert tuple(chunk.data_bytes() for chunk in report.chunks if isinstance(chunk, SDKBytesChunk)) == (
        b"first",
        b"second",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize(
    "candidate_error",
    (
        ValueError("invalid input"),
        _PublicStreamError("changed message"),
        _PublicStreamError(status_code=429),
        _PublicStreamError(code="changed_code"),
        _PublicStreamError(error_type="changed_type"),
        _PublicStreamError(param="changed_param"),
        _PublicStreamError(model="changed_model"),
        _PublicStreamError(llm_provider="changed_provider"),
    ),
    ids=("exception", "message", "status", "code", "type", "param", "model", "provider"),
)
async def test_stream_parity_rejects_public_error_changes(
    mode: Literal["sync", "async"], candidate_error: Exception
) -> None:
    chunks: Final = (_Chunk(value="partial"),)
    baseline: Final = await _consume(mode, chunks, _PublicStreamError())
    candidate: Final = await _consume(mode, chunks, candidate_error)

    with pytest.raises(AssertionError):
        assert_stream_parity(baseline, candidate)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("candidate", (("one",), ("one", "two", "three"), ("two", "one"), ("one", "changed")))
async def test_stream_parity_checks_event_sequence(mode: Literal["sync", "async"], candidate: tuple[str, ...]) -> None:
    baseline: Final = await _consume(mode, (_Chunk(value="one"), _Chunk(value="two")))
    changed: Final = await _consume(mode, tuple(_Chunk(value=value) for value in candidate))

    with pytest.raises(AssertionError):
        assert_stream_parity(baseline, changed)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_stream_parity_preserves_nested_types_and_ignores_private_fields(mode: Literal["sync", "async"]) -> None:
    first: Final = _Chunk(value="same")
    second: Final = _Chunk(value="same")
    first.set_hidden_param("request_id", "first")
    second.set_hidden_param("request_id", "second")
    baseline: Final = await _consume(mode, (_NestedChunk(value=[first]),))
    candidate: Final = await _consume(mode, (_NestedChunk(value=[second]),))

    assert_stream_parity(baseline, candidate)
    changed: Final = await _consume(mode, (_NestedChunk(value=[{"value": "same"}]),))
    with pytest.raises(AssertionError, match=r"\$\.chunks\[0\]\.value\[0\]"):
        assert_stream_parity(baseline, changed)


def test_stream_parity_rejects_wrapper_and_chunk_type_changes() -> None:
    baseline: Final = consume_sync_stream(lambda: _SyncStream((_Chunk(value="same"),)))
    different_wrapper: Final = consume_sync_stream(lambda: iter((_Chunk(value="same"),)))
    different_chunk: Final = consume_sync_stream(lambda: _SyncStream(({"value": "same"},)))

    with pytest.raises(AssertionError):
        assert_stream_parity(baseline, different_wrapper)
    with pytest.raises(AssertionError):
        assert_stream_parity(baseline, different_chunk)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("error", (None, _PublicStreamError()))
async def test_capture_keeps_serialization_failures_out_of_sdk_errors(
    mode: Literal["sync", "async"], error: Exception | None
) -> None:
    with pytest.raises(ValidationError):
        await _capture(mode, (_Chunk(value="valid"), object()), error)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_empty_stream_completes(mode: Literal["sync", "async"]) -> None:
    outcome: Final = await _consume(mode, ())
    assert outcome.chunks == ()
    assert outcome.terminal == StreamCompleted()
    assert await _capture(mode, ()) == SDKStreamReport(chunks=(), terminal=SDKStreamCompleted())


@pytest.mark.asyncio
async def test_async_creation_error_matches_sync_capture() -> None:
    async def create() -> _AsyncStream:
        _creation_error()

    sync: Final = consume_sync_stream(_creation_error)
    asynchronous: Final = await consume_async_stream(create)
    assert_stream_parity(sync, asynchronous)
    assert isinstance(sync.terminal, StreamFailed)
    assert sync.terminal.phase == "creation"
    assert isinstance(capture_sync_stream(_creation_error), SDKError)
    assert capture_sync_stream(_creation_error) == await capture_async_stream(create) == sync.terminal.error


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_capture_preserves_partial_output_and_complete_error(mode: Literal["sync", "async"]) -> None:
    error: Final = _PublicStreamError()
    report: Final = await _capture(mode, (_Chunk(value="partial"),), error)
    assert report == SDKStreamReport(
        chunks=(SDKJsonChunk(value={"value": "partial"}),),
        terminal=SDKStreamFailed(error=sdk_error_report(error)),
    )
