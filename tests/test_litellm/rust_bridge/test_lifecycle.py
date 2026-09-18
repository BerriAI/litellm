from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Final

import pytest

from litellm.rust_bridge.lifecycle import Await, Complete, Step, Yield, drive, drive_stream


class ScriptedExecution:
    """Plays scripted steps and records how it was resumed and whether it was closed."""

    def __init__(self, steps: Sequence[Step]) -> None:
        self._steps: Final = list(steps)
        self.resumed: list[tuple[str, object]] = []
        self.closed = False

    def start(self) -> Step:
        return self._steps.pop(0)

    def resume_value(self, value: object) -> Step:
        self.resumed.append(("value", value))
        return self._steps.pop(0)

    def resume_error(self, error: BaseException) -> Step:
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

    assert asyncio.run(drive(execution)) == "done"

    assert execution.resumed == [("value", 1), ("error", ValueError)]
    assert execution.closed


def test_drive_stream_yields_each_chunk_and_resumes_with_none_until_complete() -> None:
    execution: Final = ScriptedExecution([Yield("a"), Await(ready(2)), Yield("b"), Complete(None)])

    async def collect() -> list[object]:
        return [chunk async for chunk in drive_stream(execution)]

    assert asyncio.run(collect()) == ["a", "b"]
    assert execution.resumed == [("value", None), ("value", 2), ("value", None)]
    assert execution.closed


def test_drive_stream_closes_the_execution_when_the_consumer_stops_early() -> None:
    execution: Final = ScriptedExecution([Yield("a"), Yield("b"), Complete(None)])

    async def take_one() -> object:
        stream: Final = drive_stream(execution)
        first: Final = await stream.__anext__()
        await stream.aclose()
        return first

    assert asyncio.run(take_one()) == "a"
    assert execution.resumed == []
    assert execution.closed


def test_drive_rejects_a_streaming_execution_and_still_closes_it() -> None:
    execution: Final = ScriptedExecution([Yield("a")])

    with pytest.raises(RuntimeError, match="drive_stream"):
        asyncio.run(drive(execution))

    assert execution.closed
