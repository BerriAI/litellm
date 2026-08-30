from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel

from tests.route_parity.compare import public_model_copy


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
    python: StreamOutcome,
    accelerated: StreamOutcome,
    *,
    normalize: ChunkNormalizer = normalize_chunk,
) -> None:
    assert python.wrapper_type is accelerated.wrapper_type
    assert python.supports_sync_iteration is accelerated.supports_sync_iteration
    assert python.supports_async_iteration is accelerated.supports_async_iteration
    assert python.chunk_types == accelerated.chunk_types
    assert len(python.chunks) == len(accelerated.chunks)
    for python_chunk, accelerated_chunk in zip(python.chunks, accelerated.chunks, strict=True):
        assert normalize(python_chunk) == normalize(accelerated_chunk)
    assert python.terminal == accelerated.terminal
