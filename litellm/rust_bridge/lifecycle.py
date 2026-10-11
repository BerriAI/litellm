from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import Final, Protocol

KEEPALIVE_INTERVAL_SECONDS: Final = 15.0


@dataclass(frozen=True, slots=True)
class Await:
    awaitable: Awaitable[object]


@dataclass(frozen=True, slots=True)
class Keepalive:
    """A wait a held-back stream sits through. While it outlasts the keepalive interval the
    stream hands its consumer `ping` instead, and resumes the same wait on the next read."""

    awaitable: asyncio.Future[object]
    ping: object


@dataclass(frozen=True, slots=True)
class Complete:
    value: object


@dataclass(frozen=True, slots=True)
class Open:
    value: object


@dataclass(frozen=True, slots=True)
class Yield:
    value: object


Settled = Complete | Open | Yield
Step = Await | Keepalive | Settled


class Execution(Protocol):
    def start(self) -> Step: ...

    def resume_value(self, value: object) -> Step: ...

    def resume_error(self, error: BaseException) -> Step: ...

    def close(self) -> None: ...


class StreamClosed(Exception):
    """Tells a streaming execution that its caller stopped reading."""


async def _settle(execution: Execution, step: Step) -> Settled:
    while isinstance(step, Await | Keepalive):
        try:
            value = await step.awaitable
        except GeneratorExit:
            raise
        except BaseException as error:
            step = execution.resume_error(error)  # rebind-ok: advance the execution protocol
        else:
            step = execution.resume_value(value)  # rebind-ok: advance the execution protocol
    return step


def _settled(step: Step) -> Settled:
    if isinstance(step, Await | Keepalive):
        raise RuntimeError("sync call suspended")
    return step


async def drive(execution: Execution, stream_factory: Callable[[Execution, object], object] | None = None) -> object:
    handed_off = False  # rebind-ok: set once the execution belongs to the returned stream
    try:
        step: Final = await _settle(execution, execution.start())
        if isinstance(step, Open):
            factory: Final = Stream if stream_factory is None else stream_factory
            stream: Final = factory(execution, step.value)
            handed_off = True
            return stream
        return step.value
    finally:
        if not handed_off:
            execution.close()


class Stream(AsyncIterator[object]):
    """A streamed native call: each read resumes the execution until its next chunk."""

    def __init__(self, execution: Execution, head: object = None) -> None:
        self._execution: Final = execution
        self._done = False
        self._waiting: Keepalive | None = None
        self.head: Final = head

    def __aiter__(self) -> Stream:
        return self

    async def __anext__(self) -> object:
        if self._done:
            raise StopAsyncIteration
        try:
            step: Final = await self._advance()
        except BaseException:
            self._finish()
            raise
        if isinstance(step, Yield):
            return step.value
        self._finish()
        raise StopAsyncIteration

    async def _advance(self) -> Settled | Yield:
        if self._waiting is None:
            return await self._settle_or_wait(self._execution.resume_value(None))
        waiting: Final = self._waiting
        self._waiting = None
        return await self._wait(waiting)

    async def _settle_or_wait(self, step: Step) -> Settled | Yield:
        if isinstance(step, Keepalive):
            return await self._wait(step)
        return await _settle(self._execution, step)

    async def _wait(self, step: Keepalive) -> Settled | Yield:
        done, _ = await asyncio.wait({step.awaitable}, timeout=KEEPALIVE_INTERVAL_SECONDS)
        if not done:
            self._waiting = step
            return Yield(step.ping)
        error: Final = step.awaitable.exception()
        resumed: Final = (
            self._execution.resume_error(error)
            if error is not None
            else self._execution.resume_value(step.awaitable.result())
        )
        return await self._settle_or_wait(resumed)

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
