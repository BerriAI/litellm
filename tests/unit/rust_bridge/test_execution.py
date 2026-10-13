from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Final

import pytest

from litellm.rust_bridge.execution import (
    AsyncStream,
    Await,
    Complete,
    Execution,
    Step,
    StreamChunk,
    StreamOpen,
    SyncStream,
    drive,
)


class ScriptedExecution:
    """Plays scripted steps and records how it was resumed and whether it was closed."""

    def __init__(self, steps: Sequence[Step]) -> None:
        self._steps: Final = list(steps)
        self.resumed: list[tuple[str, object]] = []
        self.closed = False

    def send(self, value: object) -> Step:
        self.resumed.append(("value", value))
        return self._steps.pop(0)

    def throw(self, error: BaseException) -> Step:
        self.resumed.append(("error", type(error)))
        return self._steps.pop(0)

    def close(self) -> None:
        self.closed = True


async def ready(value: object) -> object:
    return value


async def failing() -> object:
    raise ValueError("boom")


def test_drive_resumes_each_await_with_its_result_or_error_and_returns_the_completed_value() -> None:
    execution: Final = ScriptedExecution([Await(ready(1)), Await(failing()), Complete("done")])

    assert asyncio.run(drive(execution, AsyncStream)) == "done"

    assert execution.resumed == [("value", None), ("value", 1), ("error", ValueError)]
    assert execution.closed


@pytest.mark.parametrize("factory_fails", (False, True))
def test_stream_handoff_preserves_head_identity_and_closes_on_construction_failure(factory_fails: bool) -> None:
    head: Final = object()
    stream: Final = object()
    execution: Final = ScriptedExecution([StreamOpen(head)])
    failure: Final = ValueError("stream construction failed")

    def construct(owner: Execution, received: object) -> object:
        assert owner is execution
        assert received is head
        if factory_fails:
            raise failure
        return stream

    if factory_fails:
        with pytest.raises(ValueError, match="stream construction failed") as caught:
            asyncio.run(drive(execution, construct))
        assert caught.value is failure
        assert execution.closed
    else:
        assert asyncio.run(drive(execution, construct)) is stream
        assert not execution.closed
        execution.close()


def test_drive_rejects_a_chunk_before_the_stream_opens_and_closes_the_execution() -> None:
    execution: Final = ScriptedExecution([StreamChunk("chunk")])

    with pytest.raises(RuntimeError, match="chunk before opening a stream"):
        asyncio.run(drive(execution, AsyncStream))

    assert execution.closed


@pytest.mark.parametrize("asynchronous", (False, True))
def test_stream_read_rejects_a_second_open_and_closes_the_execution(asynchronous: bool) -> None:
    execution: Final = ScriptedExecution([StreamChunk("first"), StreamOpen("head")])

    async def read() -> tuple[object, ...]:
        stream: Final = AsyncStream(execution, {})
        return (await anext(stream), await anext(stream))

    def read_sync() -> tuple[object, ...]:
        stream: Final = SyncStream(execution, {})
        return (next(stream), next(stream))

    with pytest.raises(RuntimeError, match="opened a stream twice"):
        asyncio.run(read()) if asynchronous else read_sync()

    assert execution.closed
