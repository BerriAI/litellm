from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Final, Literal

from .profiler import FunctionTraceEvent

Engine = Literal["python", "rust"]


@dataclass(frozen=True, slots=True)
class Step:
    name: str
    python: re.Pattern[str] | None
    rust: str | None


def step(name: str, python: str | None = None, rust: str | None = None) -> Step:
    return Step(name, re.compile(python) if python is not None else None, rust)


def pipeline_issues(
    engine: Engine,
    events: Sequence[FunctionTraceEvent],
    steps: Sequence[Step],
    edges: Sequence[tuple[str, str]],
) -> tuple[str, ...]:
    names: Final = tuple(event.function for event in events)
    required: Final = tuple(item.name for item in steps if getattr(item, engine) is not None)
    missing: Final = tuple(f"missing {name}" for name in required if name not in names)
    ordering: Final = tuple(
        f"{before} must precede {after}"
        for before, after in edges
        if before in names and after in names and names.index(before) >= names.index(after)
    )
    return missing + ordering


def _canonical_name(engine: Engine, function: str, steps: Sequence[Step]) -> str | None:
    for item in steps:
        if engine == "python":
            if item.python is not None and item.python.search(function):
                return item.name
        elif item.rust is not None and function == item.rust:
            return item.name
    return function if engine == "rust" else None


@dataclass(frozen=True, slots=True)
class _Projection:
    shown: tuple[FunctionTraceEvent, ...] = ()
    stack: tuple[tuple[int, int], ...] = ()
    seen: frozenset[str] = frozenset()


def _project(engine: Engine, steps: Sequence[Step], state: _Projection, event: FunctionTraceEvent) -> _Projection:
    stack: Final = tuple(pair for pair in state.stack if event.depth > pair[0])
    name: Final = _canonical_name(engine, event.function, steps)
    if name is None or name in state.seen:
        return _Projection(state.shown, stack, state.seen)
    depth: Final = (
        next(
            (
                kept.depth + 1
                for ancestor in event.ancestors
                for kept in state.shown
                if kept.function == _canonical_name(engine, ancestor, steps)
            ),
            0,
        )
        if event.ancestors is not None
        else stack[-1][1] + 1
        if stack
        else 0
    )
    return _Projection(
        state.shown + (FunctionTraceEvent(function=name, depth=depth),),
        stack + ((event.depth, depth),),
        state.seen | {name},
    )


def pipeline_steps(
    engine: Engine, events: Sequence[FunctionTraceEvent], steps: Sequence[Step]
) -> tuple[FunctionTraceEvent, ...]:
    projection: Final = reduce(lambda state, event: _project(engine, steps, state, event), events, _Projection())
    return projection.shown


@dataclass(frozen=True, slots=True)
class TraceDiff:
    python_only: tuple[str, ...]
    rust_only: tuple[str, ...]
    shared_order_matches: bool

    @property
    def matches(self) -> bool:
        return not self.python_only and not self.rust_only and self.shared_order_matches


def trace_diff(python: Sequence[FunctionTraceEvent], rust: Sequence[FunctionTraceEvent]) -> TraceDiff:
    python_names: Final = {event.function for event in python}
    rust_names: Final = {event.function for event in rust}
    shared_python: Final = tuple(event.function for event in python if event.function in rust_names)
    shared_rust: Final = tuple(event.function for event in rust if event.function in python_names)
    return TraceDiff(
        python_only=tuple(event.function for event in python if event.function not in rust_names),
        rust_only=tuple(event.function for event in rust if event.function not in python_names),
        shared_order_matches=bool(shared_python) and shared_python == shared_rust,
    )
