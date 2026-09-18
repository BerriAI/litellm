from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Await:
    awaitable: Awaitable[object]


@dataclass(frozen=True, slots=True)
class Complete:
    value: object


class Execution(Protocol):
    def start(self) -> Await | Complete: ...

    def resume_value(self, value: object) -> Await | Complete: ...

    def resume_error(self, error: BaseException) -> Await | Complete: ...

    def close(self) -> None: ...


async def drive(execution: Execution) -> object:
    try:
        step = execution.start()  # rebind-ok: the execution protocol advances after each selected await
        while isinstance(step, Await):
            try:
                value = await step.awaitable  # rebind-ok: each selected await produces the next protocol input
            except GeneratorExit:
                raise
            except BaseException as error:
                step = execution.resume_error(error)  # rebind-ok: advance the execution protocol
            else:
                step = execution.resume_value(value)  # rebind-ok: advance the execution protocol
        return step.value
    finally:
        execution.close()
