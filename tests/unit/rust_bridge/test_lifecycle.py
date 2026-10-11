from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Final

import pytest

from litellm.rust_bridge import lifecycle
from litellm.rust_bridge.lifecycle import Await, Complete, Execution, Keepalive, Open, Step, Yield, drive


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


@pytest.mark.parametrize("factory_fails", (False, True))
def test_stream_handoff_preserves_head_identity_and_closes_on_construction_failure(factory_fails: bool) -> None:
    head: Final = object()
    stream: Final = object()
    execution: Final = ScriptedExecution([Open(head)])
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


def test_a_stream_pings_while_a_keepalive_wait_outlasts_the_interval_and_then_resumes_with_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lifecycle, "KEEPALIVE_INTERVAL_SECONDS", 0.01)

    async def exercise() -> list[object]:
        loop: Final = asyncio.get_running_loop()
        slow: Final[asyncio.Future[object]] = loop.create_future()
        loop.call_later(0.05, slow.set_result, "assembled")
        execution: Final = ScriptedExecution([Open("head"), Keepalive(slow, b"ping"), Yield(b"chunk"), Complete(None)])
        stream: Final = await drive(execution)
        assert isinstance(stream, lifecycle.Stream)
        chunks: Final = [chunk async for chunk in stream]
        assert execution.resumed[-2:] == [("value", "assembled"), ("value", None)]
        return chunks

    chunks: Final = asyncio.run(exercise())

    assert chunks[-1] == b"chunk"
    assert len(chunks) >= 2
    assert all(chunk == b"ping" for chunk in chunks[:-1])


def test_a_keepalive_wait_that_fails_resumes_the_execution_with_the_error() -> None:
    async def exercise() -> list[object]:
        loop: Final = asyncio.get_running_loop()
        failing: Final[asyncio.Future[object]] = loop.create_future()
        failing.set_exception(ValueError("upstream broke"))
        execution: Final = ScriptedExecution([Open("head"), Keepalive(failing, b"ping"), Complete(None)])
        stream: Final = await drive(execution)
        chunks: Final = [chunk async for chunk in stream]
        assert execution.resumed == [("value", None), ("error", ValueError)]
        return chunks

    assert asyncio.run(exercise()) == []
