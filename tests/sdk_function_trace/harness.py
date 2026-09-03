from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import FunctionType
from typing import Final, cast

from tests.sdk_function_trace.profiler import FunctionTraceEvent, profile_python


@dataclass(frozen=True, slots=True)
class TraceStep:
    function: FunctionType
    depth: int


@dataclass(frozen=True, slots=True)
class TraceScenario:
    steps: tuple[TraceStep, ...]
    invoke_python: Callable[[], object]
    invoke_rust: Callable[[], Sequence[FunctionTraceEvent]]


def assert_function_trace_parity(scenario: TraceScenario) -> None:
    expected: Final = tuple(
        FunctionTraceEvent(function=step.function.__name__, depth=step.depth) for step in scenario.steps
    )
    functions: Final = cast(tuple[FunctionType, ...], tuple(step.function for step in scenario.steps))
    with profile_python(functions) as profiler:
        scenario.invoke_python()
    python_trace: Final = tuple(profiler.events)
    rust_trace: Final = tuple(scenario.invoke_rust())

    if python_trace != expected:
        raise AssertionError(f"Python function trace differs: {python_trace!r} != {expected!r}")
    if rust_trace != expected:
        raise AssertionError(f"Rust function trace differs: {rust_trace!r} != {expected!r}")
    if python_trace != rust_trace:
        raise AssertionError(f"Python and Rust function traces differ: {python_trace!r} != {rust_trace!r}")
