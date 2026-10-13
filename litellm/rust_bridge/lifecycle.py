from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Final, Protocol


@dataclass(frozen=True, slots=True)
class Await:
    awaitable: Awaitable[object]


@dataclass(frozen=True, slots=True)
class Complete:
    value: object


@dataclass(frozen=True, slots=True)
class StreamOpen:
    head: object


@dataclass(frozen=True, slots=True)
class StreamChunk:
    chunk: object


StreamStep = StreamOpen | StreamChunk
Settled = Complete | StreamStep
Step = Await | Settled


class Execution(Protocol):
    def send(self, value: object) -> Step: ...

    def throw(self, error: BaseException) -> Step: ...

    def close(self) -> None: ...


_REOPENED: Final = "execution opened a stream twice"


class StreamClosed(Exception):
    """Tells a streaming execution that its caller stopped reading."""


async def _settle(execution: Execution, step: Step) -> Settled:
    while isinstance(step, Await):
        try:
            value = await step.awaitable
        except GeneratorExit:
            raise
        except BaseException as error:
            step = execution.throw(error)  # rebind-ok: advance the execution protocol
        else:
            step = execution.send(value)  # rebind-ok: advance the execution protocol
    return step


def _settled(step: Step) -> Settled:
    if isinstance(step, Await):
        raise RuntimeError("sync call suspended")
    return step


def _hand_off(
    execution: Execution, step: Settled, open_stream: Callable[[Execution, object], object], cleanup: ExitStack
) -> object:
    match step:
        case Complete(value):
            return value
        case StreamOpen(head):
            stream: Final = open_stream(execution, head)
            cleanup.pop_all()
            return stream
        case StreamChunk():
            raise RuntimeError("execution yielded a chunk before opening a stream")


async def drive(execution: Execution, open_stream: Callable[[Execution, object], object]) -> object:
    with ExitStack() as cleanup:
        cleanup.callback(execution.close)
        return _hand_off(execution, await _settle(execution, execution.send(None)), open_stream, cleanup)


def run(execution: Execution, open_stream: Callable[[Execution, object], object]) -> object:
    with ExitStack() as cleanup:
        cleanup.callback(execution.close)
        return _hand_off(execution, _settled(execution.send(None)), open_stream, cleanup)


class AsyncStream(AsyncIterator[object]):
    """A streamed native call: each read resumes the execution until its next chunk."""

    def __init__(self, execution: Execution, head: object = None) -> None:
        self._execution: Final = execution
        self._done = False
        self.head: Final = head

    def __aiter__(self) -> AsyncStream:
        return self

    async def __anext__(self) -> object:
        if self._done:
            raise StopAsyncIteration
        try:
            step: Final = await _settle(self._execution, self._execution.send(None))
        except BaseException:
            self._finish()
            raise
        match step:
            case StreamChunk(chunk):
                return chunk
            case Complete():
                self._finish()
                raise StopAsyncIteration
            case StreamOpen():
                self._finish()
                raise RuntimeError(_REOPENED)

    async def aclose(self) -> None:
        if self._done:
            return
        try:
            await _settle(self._execution, self._execution.throw(StreamClosed()))
        finally:
            self._finish()

    def _finish(self) -> None:
        self._done = True
        self._execution.close()


class SyncStream(Iterator[object]):
    """The sync form of `AsyncStream`; its execution never suspends on an awaitable."""

    def __init__(self, execution: Execution, head: object = None) -> None:
        self._execution: Final = execution
        self._done = False
        self.head: Final = head

    def __iter__(self) -> SyncStream:
        return self

    def __next__(self) -> object:
        if self._done:
            raise StopIteration
        try:
            step: Final = _settled(self._execution.send(None))
        except BaseException:
            self._finish()
            raise
        match step:
            case StreamChunk(chunk):
                return chunk
            case Complete():
                self._finish()
                raise StopIteration
            case StreamOpen():
                self._finish()
                raise RuntimeError(_REOPENED)

    def close(self) -> None:
        if self._done:
            return
        try:
            _settled(self._execution.throw(StreamClosed()))
        finally:
            self._finish()

    def _finish(self) -> None:
        self._done = True
        self._execution.close()
