from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from tests.route_parity.compare import assert_value_parity
from tests.route_parity.models import (
    SDKError,
    SDKReport,
    SDKStreamCompleted,
    SDKStreamFailed,
    SDKStreamReport,
    sdk_chunk,
    sdk_error_report,
)


@dataclass(frozen=True, slots=True)
class StreamCompleted:
    kind: Literal["completed"] = "completed"


@dataclass(frozen=True, slots=True)
class StreamFailed:
    phase: Literal["creation", "iteration"]
    exception_type: type[BaseException]
    error: SDKError
    kind: Literal["failed"] = "failed"


StreamTerminal: TypeAlias = StreamCompleted | StreamFailed


@dataclass(frozen=True, slots=True)
class StreamOutcome:
    wrapper_type: type[object] | None
    supports_sync_iteration: bool | None
    supports_async_iteration: bool | None
    chunks: tuple[object, ...]
    chunk_types: tuple[type[object], ...]
    terminal: StreamTerminal


ChunkNormalizer: TypeAlias = Callable[[object], object]


def drain_sync_stream(stream: Iterable[object]) -> None:
    for _ in stream:
        pass


async def drain_async_stream(stream: AsyncIterable[object]) -> None:
    async for _ in stream:
        pass


def capture_sync_stream(create: Callable[[], Iterable[object]]) -> SDKReport:
    return _stream_report(consume_sync_stream(create))


async def capture_async_stream(create: Callable[[], Awaitable[AsyncIterable[object]]]) -> SDKReport:
    return _stream_report(await consume_async_stream(create))


def _stream_report(outcome: StreamOutcome) -> SDKReport:
    terminal: Final = outcome.terminal
    if isinstance(terminal, StreamFailed) and terminal.phase == "creation":
        return terminal.error
    return SDKStreamReport(
        chunks=tuple(sdk_chunk(chunk) for chunk in outcome.chunks),
        terminal=SDKStreamFailed(error=terminal.error) if isinstance(terminal, StreamFailed) else SDKStreamCompleted(),
    )


def _failed(phase: Literal["creation", "iteration"], error: Exception) -> StreamFailed:
    return StreamFailed(
        phase=phase,
        exception_type=type(error),
        error=sdk_error_report(error),
    )


def consume_sync_stream(create: Callable[[], Iterable[object]]) -> StreamOutcome:
    try:
        stream: Final = create()
    except Exception as error:
        return StreamOutcome(
            wrapper_type=None,
            supports_sync_iteration=None,
            supports_async_iteration=None,
            chunks=(),
            chunk_types=(),
            terminal=_failed("creation", error),
        )

    chunks: list[object] = []  # mutable-ok: iterator consumption builds an ordered trace
    try:
        for chunk in stream:
            chunks.append(chunk)  # noqa: PERF402  # partial trace is required if iteration raises
    except Exception as error:
        recorded: Final = tuple(chunks)
        return StreamOutcome(
            wrapper_type=type(stream),
            supports_sync_iteration=hasattr(stream, "__iter__"),
            supports_async_iteration=hasattr(stream, "__aiter__"),
            chunks=recorded,
            chunk_types=tuple(type(chunk) for chunk in recorded),
            terminal=_failed("iteration", error),
        )
    completed_chunks: Final = tuple(chunks)
    return StreamOutcome(
        wrapper_type=type(stream),
        supports_sync_iteration=hasattr(stream, "__iter__"),
        supports_async_iteration=hasattr(stream, "__aiter__"),
        chunks=completed_chunks,
        chunk_types=tuple(type(chunk) for chunk in completed_chunks),
        terminal=StreamCompleted(),
    )


async def consume_async_stream(create: Callable[[], Awaitable[AsyncIterable[object]]]) -> StreamOutcome:
    try:
        stream: Final = await create()
    except Exception as error:
        return StreamOutcome(
            wrapper_type=None,
            supports_sync_iteration=None,
            supports_async_iteration=None,
            chunks=(),
            chunk_types=(),
            terminal=_failed("creation", error),
        )

    chunks: list[object] = []  # mutable-ok: iterator consumption builds an ordered trace
    try:
        async for chunk in stream:
            chunks.append(chunk)
    except Exception as error:
        recorded: Final = tuple(chunks)
        return StreamOutcome(
            wrapper_type=type(stream),
            supports_sync_iteration=hasattr(stream, "__iter__"),
            supports_async_iteration=hasattr(stream, "__aiter__"),
            chunks=recorded,
            chunk_types=tuple(type(chunk) for chunk in recorded),
            terminal=_failed("iteration", error),
        )
    completed_chunks: Final = tuple(chunks)
    return StreamOutcome(
        wrapper_type=type(stream),
        supports_sync_iteration=hasattr(stream, "__iter__"),
        supports_async_iteration=hasattr(stream, "__aiter__"),
        chunks=completed_chunks,
        chunk_types=tuple(type(chunk) for chunk in completed_chunks),
        terminal=StreamCompleted(),
    )


def normalize_chunk(chunk: object) -> object:
    return chunk


def assert_stream_parity(
    baseline: StreamOutcome,
    candidate: StreamOutcome,
    *,
    normalize: ChunkNormalizer = normalize_chunk,
) -> None:
    assert baseline.wrapper_type is candidate.wrapper_type
    assert baseline.supports_sync_iteration is candidate.supports_sync_iteration
    assert baseline.supports_async_iteration is candidate.supports_async_iteration
    assert baseline.chunk_types == candidate.chunk_types
    assert len(baseline.chunks) == len(candidate.chunks)
    for index, (baseline_chunk, candidate_chunk) in enumerate(zip(baseline.chunks, candidate.chunks, strict=True)):
        assert_value_parity(normalize(baseline_chunk), normalize(candidate_chunk), path=f"$.chunks[{index}]")
    assert baseline.terminal == candidate.terminal
