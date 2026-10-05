from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Final

import pytest

from litellm.rust_bridge.lifecycle import Await, Complete, Execution, Open, Step, drive


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
