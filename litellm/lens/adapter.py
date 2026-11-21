from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Final, TypeAlias

from litellm.harness.evaluation import EvalContext
from litellm.harness.types import Event, Result, StopReason, Usage


@dataclass(frozen=True, slots=True)
class AgentInput:
    task_id: str
    prompt: str
    model: str
    workspace: Path
    emit: Callable[[Event], None]


@dataclass(frozen=True, slots=True)
class AgentOutput:
    text: str
    events: tuple[Event, ...] = ()
    cost: float | None = None
    model_calls: int | None = None
    stop_reason: StopReason = "done"


PythonAgent: TypeAlias = Callable[[AgentInput], AgentOutput | Awaitable[AgentOutput]]


class _Events:
    def __init__(self, emit: Callable[[Event], None]) -> None:
        self.emit = emit
        self.events: tuple[Event, ...] = ()

    def __call__(self, event: Event) -> None:
        self.emit(event)
        self.events = (*self.events, event)


def _require_output(value: object) -> AgentOutput:
    if not isinstance(value, AgentOutput):
        raise TypeError("Python agents must return AgentOutput")
    return value


def python_agent(agent: PythonAgent) -> Callable[[EvalContext], Awaitable[Result]]:
    """Use an async callable for live, concurrent Python harnesses."""

    @wraps(agent)
    async def run(context: EvalContext) -> Result:
        events: Final = _Events(context.emit)
        pending: Final = agent(
            AgentInput(
                task_id=context.task.id,
                prompt=context.task.prompt,
                model=context.target.model,
                workspace=Path(context.sandbox.workdir),
                emit=events,
            )
        )
        output: Final = _require_output(await pending if inspect.isawaitable(pending) else pending)
        for event in output.events:
            if event not in events.events:
                events(event)
        return Result(
            text=output.text,
            output=None,
            files=[],
            events=list(events.events),
            usage=Usage(calls=output.model_calls or 0),
            cost=output.cost if output.cost is not None else 0,
            stop_reason=output.stop_reason,
            session_id=str(context.metadata["agent_eval_run"]),
        )

    return run
