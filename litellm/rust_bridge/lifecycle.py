from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass
from typing import Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class Await:
    awaitable: Awaitable[object]


@dataclass(frozen=True, slots=True)
class Yield:
    value: object


@dataclass(frozen=True, slots=True)
class Complete:
    value: object


Step: TypeAlias = Await | Yield | Complete


class Execution(Protocol):
    def start(self) -> Step: ...

    def resume_value(self, value: object) -> Step: ...

    def resume_error(self, error: BaseException) -> Step: ...

    def close(self) -> None: ...


async def _advance(execution: Execution, step: Await) -> Step:
    try:
        value = await step.awaitable
    except GeneratorExit:
        raise
    except BaseException as error:
        return execution.resume_error(error)
    return execution.resume_value(value)


async def drive(execution: Execution) -> object:
    try:
        step = execution.start()  # rebind-ok: the execution protocol advances after each selected await
        while not isinstance(step, Complete):
            if isinstance(step, Yield):
                raise RuntimeError("a streaming execution must be driven with drive_stream")
            step = await _advance(execution, step)  # rebind-ok: advance the execution protocol
        return step.value
    finally:
        execution.close()


async def drive_stream(execution: Execution) -> AsyncIterator[object]:
    """Yield each chunk in the caller's task; the execution completes after the last one."""
    try:
        step = execution.start()  # rebind-ok: the execution protocol advances after each selected await
        while not isinstance(step, Complete):
            if isinstance(step, Yield):
                yield step.value
                step = execution.resume_value(None)  # rebind-ok: the consumer asked for the next chunk
                continue
            step = await _advance(execution, step)  # rebind-ok: advance the execution protocol
    finally:
        execution.close()
