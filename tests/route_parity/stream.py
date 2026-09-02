from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel

from tests.route_parity.compare import public_model_copy
from tests.route_parity.models import (
    SDKChunk,
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
    status_code: int | None
    llm_provider: str | None
    model: str | None
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
    try:
        stream: Final = create()
    except Exception as error:
        return sdk_error_report(error)

    chunks: list[SDKChunk] = []  # mutable-ok: partial chunks must survive an iteration failure
    try:
        for chunk in stream:
            chunks.append(sdk_chunk(chunk))
    except Exception as error:
        return SDKStreamReport(chunks=tuple(chunks), terminal=SDKStreamFailed(error=sdk_error_report(error)))
    return SDKStreamReport(chunks=tuple(chunks), terminal=SDKStreamCompleted())


async def capture_async_stream(create: Callable[[], Awaitable[AsyncIterable[object]]]) -> SDKReport:
    try:
        stream: Final = await create()
    except Exception as error:
        return sdk_error_report(error)

    chunks: list[SDKChunk] = []  # mutable-ok: partial chunks must survive an iteration failure
    try:
        async for chunk in stream:
            chunks.append(sdk_chunk(chunk))
    except Exception as error:
        return SDKStreamReport(chunks=tuple(chunks), terminal=SDKStreamFailed(error=sdk_error_report(error)))
    return SDKStreamReport(chunks=tuple(chunks), terminal=SDKStreamCompleted())


def _failed(phase: Literal["creation", "iteration"], error: BaseException) -> StreamFailed:
    status_code: Final = getattr(error, "status_code", None)
    llm_provider: Final = getattr(error, "llm_provider", None)
    model: Final = getattr(error, "model", None)
    return StreamFailed(
        phase=phase,
        exception_type=type(error),
        status_code=status_code if isinstance(status_code, int) else None,
        llm_provider=llm_provider if isinstance(llm_provider, str) else None,
        model=model if isinstance(model, str) else None,
    )


def consume_sync_stream(create: Callable[[], Iterable[object]]) -> StreamOutcome:
    try:
        stream = create()
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
        stream = await create()
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
    if isinstance(chunk, BaseModel):
        return public_model_copy(chunk)
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
    for baseline_chunk, candidate_chunk in zip(baseline.chunks, candidate.chunks, strict=True):
        assert normalize(baseline_chunk) == normalize(candidate_chunk)
    assert baseline.terminal == candidate.terminal
