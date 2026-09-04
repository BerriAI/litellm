from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import Final

import pytest

from .profiler import FunctionTraceEvent, PythonProfiler, profile_python, profile_python_function_usage


def _events_named(profiler: PythonProfiler, name: str) -> tuple[FunctionTraceEvent, ...]:
    return tuple(event for event in profiler.events if event.function.endswith(name))


def test_profiler_keeps_repeated_calls() -> None:
    def called() -> None:
        return None

    with profile_python(Path(__file__).parent) as profiler:
        called()
        called()

    assert len(_events_named(profiler, "called")) == 2


def test_profiler_records_real_frame_ancestry() -> None:
    def called() -> None:
        return None

    def outer() -> None:
        called()

    with profile_python(Path(__file__).parent) as profiler:
        outer()

    outer_event, called_event = (event for event in profiler.events if event.function.endswith(("outer", "called")))
    assert called_event.parent_id == outer_event.id


def test_profiler_restores_previous_profiler_after_failure() -> None:
    previous: Final = sys.getprofile()

    with pytest.raises(RuntimeError, match="stop"):
        with profile_python(Path(__file__).parent):
            raise RuntimeError("stop")

    assert sys.getprofile() is previous


def test_profiler_does_not_count_coroutine_resumption_as_another_call() -> None:
    async def suspended() -> None:
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    with profile_python(Path(__file__).parent) as profiler:
        asyncio.run(suspended())

    assert len(_events_named(profiler, "suspended")) == 1


def test_profiler_preserves_parent_across_coroutine_suspension() -> None:
    def called() -> None:
        return None

    async def suspended() -> None:
        await asyncio.sleep(0)
        called()

    with profile_python(Path(__file__).parent) as profiler:
        asyncio.run(suspended())

    suspended_event: Final = _events_named(profiler, "suspended")[0]
    called_event: Final = _events_named(profiler, "called")[0]
    assert called_event.parent_id == suspended_event.id


def test_profiler_captures_worker_threads_when_enabled() -> None:
    def called() -> None:
        return None

    with profile_python(Path(__file__).parent, threads=True) as profiler:
        thread: Final = threading.Thread(target=called)
        thread.start()
        thread.join()

    called_event: Final = _events_named(profiler, "called")[0]
    assert called_event.parent_id is None


def test_function_usage_profiler_records_only_selected_functions() -> None:
    def selected() -> None:
        return None

    def ignored() -> None:
        return None

    source_root: Final = Path(__file__).parent
    function: Final = PythonProfiler(source_root).function_name(selected.__code__)
    assert function is not None

    with profile_python_function_usage(source_root, frozenset((function,))) as profiler:
        selected()
        ignored()

    assert profiler.called == {function}
