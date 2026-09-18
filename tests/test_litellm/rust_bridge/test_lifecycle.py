from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Final

from litellm.rust_bridge.lifecycle import Await, Complete, drive


class ScriptedExecution:
    """Plays scripted steps and records how it was resumed and whether it was closed."""

    def __init__(self, steps: Sequence[Await | Complete]) -> None:
        self._steps: Final = list(steps)
        self.resumed: list[tuple[str, object]] = []
        self.closed = False

    def start(self) -> Await | Complete:
        return self._steps.pop(0)

    def resume_value(self, value: object) -> Await | Complete:
        self.resumed.append(("value", value))
        return self._steps.pop(0)

    def resume_error(self, error: BaseException) -> Await | Complete:
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
