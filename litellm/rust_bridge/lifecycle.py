from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Iterator
from dataclasses import dataclass
from typing import Final, Protocol


@dataclass(frozen=True, slots=True)
class Await:
    awaitable: Awaitable[object]


@dataclass(frozen=True, slots=True)
class Complete:
    value: object


@dataclass(frozen=True, slots=True)
class Open:
    value: None


@dataclass(frozen=True, slots=True)
class Yield:
    value: object


Settled = Complete | Open | Yield
Step = Await | Settled


class Execution(Protocol):
    def start(self) -> Step: ...

    def resume_value(self, value: object) -> Step: ...

    def resume_error(self, error: BaseException) -> Step: ...

    def close(self) -> None: ...


class StreamClosed(Exception):
    """Tells a streaming execution that its caller stopped reading."""


async def _settle(execution: Execution, step: Step) -> Settled:
    while isinstance(step, Await):
        try:
            value = await step.awaitable  # rebind-ok: each selected await produces the next protocol input
        except GeneratorExit:
            raise
        except BaseException as error:
            step = execution.resume_error(error)  # rebind-ok: advance the execution protocol
        else:
            step = execution.resume_value(value)  # rebind-ok: advance the execution protocol
    return step


def _settled(step: Step) -> Settled:
    if isinstance(step, Await):
        raise RuntimeError("sync call suspended")
    return step


async def drive(execution: Execution) -> object:
    handed_off = False  # rebind-ok: set once the execution belongs to the returned stream
    try:
        step: Final = await _settle(execution, execution.start())
        if isinstance(step, Open):
            handed_off = True
            return Stream(execution)
        return step.value
    finally:
        if not handed_off:
            execution.close()


class Stream(AsyncIterator[object]):
    """A streamed native call: each read resumes the execution until its next chunk."""

    def __init__(self, execution: Execution) -> None:
        self._execution: Final = execution
        self._done = False

    def __aiter__(self) -> Stream:
        return self

    async def __anext__(self) -> object:
        if self._done:
            raise StopAsyncIteration
        try:
            step: Final = await _settle(self._execution, self._execution.resume_value(None))
        except BaseException:
            self._finish()
            raise
        if isinstance(step, Yield):
            return step.value
        self._finish()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        if self._done:
            return
        try:
            await _settle(self._execution, self._execution.resume_error(StreamClosed()))
        finally:
            self._finish()

    def _finish(self) -> None:
        self._done = True
        self._execution.close()


class SyncStream(Iterator[object]):
    """The sync form of `Stream`; its execution never suspends on an awaitable."""

    def __init__(self, execution: Execution) -> None:
        self._execution: Final = execution
        self._done = False

    def __iter__(self) -> SyncStream:
        return self

    def __next__(self) -> object:
        if self._done:
            raise StopIteration
        try:
            step: Final = _settled(self._execution.resume_value(None))
        except BaseException:
            self._finish()
            raise
        if isinstance(step, Yield):
            return step.value
        self._finish()
        raise StopIteration

    def close(self) -> None:
        if self._done:
            return
        try:
            _settled(self._execution.resume_error(StreamClosed()))
        finally:
            self._finish()

    def _finish(self) -> None:
        self._done = True
        self._execution.close()
