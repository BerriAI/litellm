from __future__ import annotations

import queue
from collections.abc import AsyncIterator, Iterator
from typing import Final

import pytest
from pydantic import BaseModel, PrivateAttr

from tests.route_parity.models import SDKBytesChunk, SDKJsonChunk, SDKStreamFailed, SDKStreamReport
from tests.route_parity.stream import (
    StreamFailed,
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
        self, message: str = "runtime-specific message", *, status_code: int, llm_provider: str, model: str
    ) -> None:
        super().__init__(message)
        self.status_code: Final = status_code
        self.llm_provider: Final = llm_provider
        self.model: Final = model


def _creation_error() -> _SyncStream:
    raise _PublicStreamError(status_code=429, llm_provider="test", model="test-model")


async def _async_stream(chunks: tuple[object, ...], error: BaseException | None = None) -> _AsyncStream:
    return _AsyncStream(chunks, error)


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
        "python runtime detail", status_code=500, llm_provider="test", model="test-model"
    )
    accelerated_error: Final = _PublicStreamError(
        "rust runtime detail", status_code=500, llm_provider="test", model="test-model"
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
