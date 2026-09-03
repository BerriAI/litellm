from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import FunctionType
from typing import Final, cast

import pytest

from tests.sdk_function_trace import (
    FunctionTraceEvent,
    TraceScenario,
    TraceStep,
    assert_function_trace_parity,
)
from tests.sdk_function_trace.profiler import profile_python


class First:
    @staticmethod
    def run() -> None:
        return None


class Second:
    @staticmethod
    def run() -> None:
        return None


def test_profiler_matches_code_objects_and_keeps_repeated_calls() -> None:
    with profile_python((First.run,)) as profiler:
        Second.run()
        First.run()
        First.run()

    assert profiler.events == [
        FunctionTraceEvent(function="run", depth=0),
        FunctionTraceEvent(function="run", depth=0),
    ]


def test_profiler_records_selected_function_nesting_depth() -> None:
    class Nested:
        @staticmethod
        def run() -> None:
            First.run()

    with profile_python((Nested.run, First.run)) as profiler:
        Nested.run()

    assert profiler.events == [
        FunctionTraceEvent(function="run", depth=0),
        FunctionTraceEvent(function="run", depth=1),
    ]


def test_profiler_restores_previous_profiler_after_failure() -> None:
    previous: Final = sys.getprofile()

    with profile_python((First.run,)) as outer:
        with pytest.raises(RuntimeError, match="stop"):
            with profile_python((Second.run,)):
                raise RuntimeError("stop")
        assert sys.getprofile() is outer
        First.run()

    assert sys.getprofile() is previous
    assert outer.events == [FunctionTraceEvent(function="run", depth=0)]


def test_profiler_does_not_count_coroutine_resumption_as_another_call() -> None:
    async def suspended() -> None:
        await asyncio.sleep(0)
        First.run()
        await asyncio.sleep(0)

    with profile_python((suspended, First.run)) as profiler:
        asyncio.run(suspended())

    assert profiler.events == [
        FunctionTraceEvent(function="suspended", depth=0),
        FunctionTraceEvent(function="run", depth=1),
    ]


def test_source_profiler_records_real_frame_ancestry() -> None:
    def outer() -> None:
        First.run()

    with profile_python(source_root=Path(__file__).parent) as profiler:
        outer()
        Second.run()

    outer_event, first_event, second_event = (
        event for event in profiler.events if event.function.startswith("test_profiler.py:")
    )
    assert first_event.ancestors is not None
    assert outer_event.function in first_event.ancestors
    assert second_event.ancestors is not None
    assert outer_event.function not in second_event.ancestors


@pytest.mark.parametrize(
    "rust_trace",
    [
        (),
        (FunctionTraceEvent(function="renamed", depth=0),),
        (FunctionTraceEvent(function="run", depth=1),),
        (FunctionTraceEvent(function="run", depth=0),) * 2,
    ],
    ids=["missing", "renamed", "wrong-depth", "extra-call"],
)
def test_harness_rejects_rust_function_trace_drift(rust_trace: tuple[FunctionTraceEvent, ...]) -> None:
    with pytest.raises(AssertionError, match="Rust function trace differs"):
        assert_function_trace_parity(
            TraceScenario(
                steps=(TraceStep(cast(FunctionType, First.run), depth=0),),
                invoke_python=First.run,
                invoke_rust=lambda: rust_trace,
            )
        )


def test_harness_rejects_python_function_trace_drift() -> None:
    with pytest.raises(AssertionError, match="Python function trace differs"):
        assert_function_trace_parity(
            TraceScenario(
                steps=(TraceStep(cast(FunctionType, First.run), depth=0),),
                invoke_python=Second.run,
                invoke_rust=lambda: (FunctionTraceEvent(function="run", depth=0),),
            )
        )


def test_harness_accepts_matching_traces() -> None:
    assert_function_trace_parity(
        TraceScenario(
            steps=(TraceStep(cast(FunctionType, First.run), depth=0),),
            invoke_python=First.run,
            invoke_rust=lambda: (FunctionTraceEvent(function="run", depth=0),),
        )
    )


def test_harness_rejects_reordered_calls() -> None:
    def begin() -> None:
        return None

    def finish() -> None:
        return None

    with pytest.raises(AssertionError, match="Rust function trace differs"):
        assert_function_trace_parity(
            TraceScenario(
                steps=(
                    TraceStep(cast(FunctionType, begin), depth=0),
                    TraceStep(cast(FunctionType, finish), depth=0),
                ),
                invoke_python=lambda: (begin(), finish()),
                invoke_rust=lambda: (
                    FunctionTraceEvent(function="finish", depth=0),
                    FunctionTraceEvent(function="begin", depth=0),
                ),
            )
        )
