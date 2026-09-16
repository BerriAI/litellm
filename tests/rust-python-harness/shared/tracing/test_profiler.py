from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
from types import FrameType, FunctionType
from typing import Final, ParamSpec, TypeVar, cast

import pytest

from .profiler import (
    FunctionTraceEvent,
    PythonProfiler,
    _module_qualnames,
    profile_python,
    profile_python_function_usage,
)

_P = ParamSpec("_P")
_T = TypeVar("_T")


def _passthrough(function: Callable[_P, _T]) -> Callable[_P, _T]:
    @wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        return function(*args, **kwargs)

    return wrapper


class Decorated:
    @_passthrough
    def call(self) -> None:
        return None


def _events_named(profiler: PythonProfiler, name: str) -> tuple[FunctionTraceEvent, ...]:
    return tuple(event for event in profiler.events if event.function.endswith(name))


@pytest.mark.parametrize("threads", (False, True))
def test_profiler_keeps_repeated_calls(threads: bool) -> None:
    def called() -> None:
        return None

    with profile_python(Path(__file__).parent, threads=threads) as profiler:
        called()
        called()

    assert len(_events_named(profiler, "called")) == 2


def test_profiler_qualifies_decorated_methods_by_class() -> None:
    with profile_python(Path(__file__).parent) as profiler:
        Decorated().call()

    assert any(event.function.endswith(" Decorated.call") for event in profiler.events)
    assert _module_qualnames(__name__)[cast(FunctionType, Decorated.call.__wrapped__).__code__] == "Decorated.call"


@pytest.mark.parametrize("threads", (False, True))
def test_profiler_records_real_frame_ancestry(threads: bool) -> None:
    def called() -> None:
        return None

    def outer() -> None:
        called()

    with profile_python(Path(__file__).parent, threads=threads) as profiler:
        outer()

    outer_event, called_event = (event for event in profiler.events if event.function.endswith(("outer", "called")))
    assert called_event.parent_id == outer_event.id


def test_profiler_restores_previous_profiler_after_failure() -> None:
    previous: Final = sys.getprofile()

    with pytest.raises(RuntimeError, match="stop"):
        with profile_python(Path(__file__).parent):
            raise RuntimeError("stop")

    assert sys.getprofile() is previous


@pytest.mark.parametrize("threads", (False, True))
def test_profiler_does_not_count_coroutine_resumption_as_another_call(threads: bool) -> None:
    async def suspended() -> None:
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    with profile_python(Path(__file__).parent, threads=threads) as profiler:
        asyncio.run(suspended())

    assert len(_events_named(profiler, "suspended")) == 1


@pytest.mark.parametrize("threads", (False, True))
def test_profiler_preserves_parent_across_coroutine_suspension(threads: bool) -> None:
    def called() -> None:
        return None

    async def suspended() -> None:
        await asyncio.sleep(0)
        called()

    with profile_python(Path(__file__).parent, threads=threads) as profiler:
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


@pytest.mark.skipif(sys.version_info < (3, 12), reason="existing worker capture requires sys.monitoring")
@pytest.mark.parametrize("prewarm", (False, True))
def test_profiler_captures_reused_workers_without_leaking_between_sessions(prewarm: bool) -> None:
    def called() -> None:
        return None

    with ThreadPoolExecutor(max_workers=1) as executor:
        if prewarm:
            executor.submit(called).result(timeout=5)
        with profile_python(Path(__file__).parent, threads=True) as first:
            executor.submit(called).result(timeout=5)
        executor.submit(called).result(timeout=5)
        with profile_python(Path(__file__).parent, threads=True) as second:
            executor.submit(called).result(timeout=5)
        executor.submit(called).result(timeout=5)

    assert len(_events_named(first, "called")) == 1
    assert len(_events_named(second, "called")) == 1


def test_profiler_restores_main_and_worker_hooks_after_failure() -> None:
    previous: Final = sys.getprofile()
    previous_thread: Final = threading.getprofile()

    with ThreadPoolExecutor(max_workers=1) as executor:
        worker_previous: Final = executor.submit(sys.getprofile).result(timeout=5)
        with pytest.raises(RuntimeError, match="stop"):
            with profile_python(Path(__file__).parent, threads=True):
                raise RuntimeError("stop")
        assert executor.submit(sys.getprofile).result(timeout=5) is worker_previous

    assert sys.getprofile() is previous
    assert threading.getprofile() is previous_thread


@pytest.mark.skipif(sys.version_info < (3, 12), reason="existing worker capture requires sys.monitoring")
def test_function_usage_profiler_captures_reused_workers() -> None:
    def selected() -> None:
        return None

    function: Final = f"{Path(__file__).name}:{selected.__code__.co_firstlineno} {selected.__qualname__}"
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(selected).result(timeout=5)
        with profile_python_function_usage(Path(__file__).parent, frozenset((function,)), threads=True) as profiler:
            executor.submit(selected).result(timeout=5)

    assert profiler.called == {function}


@pytest.mark.skipif(sys.version_info < (3, 12), reason="independent thread hooks require sys.monitoring")
def test_threaded_profiler_preserves_custom_worker_hook_and_releases_monitoring_slot() -> None:
    def worker_hook(_frame: FrameType, _event: str, _arg: object) -> None:
        return None

    def fail_with_profile(executor: ThreadPoolExecutor) -> None:
        with profile_python(Path(__file__).parent, threads=True):
            assert executor.submit(sys.getprofile).result(timeout=5) is worker_hook
            raise RuntimeError("stop")

    tools_before: Final = tuple(sys.monitoring.get_tool(slot) for slot in range(6))
    with ThreadPoolExecutor(max_workers=1, initializer=lambda: sys.setprofile(worker_hook)) as executor:
        assert executor.submit(sys.getprofile).result(timeout=5) is worker_hook
        with pytest.raises(RuntimeError, match="stop"):
            fail_with_profile(executor)
        assert executor.submit(sys.getprofile).result(timeout=5) is worker_hook

    assert tuple(sys.monitoring.get_tool(slot) for slot in range(6)) == tools_before


@pytest.mark.skipif(sys.version_info < (3, 12), reason="existing worker capture requires sys.monitoring")
def test_threaded_profiler_keeps_concurrent_event_ids_and_parent_links() -> None:
    def child() -> None:
        return None

    def parent() -> None:
        child()

    with ThreadPoolExecutor(max_workers=4) as executor:
        with profile_python(Path(__file__).parent, threads=True) as profiler:
            futures: Final = tuple(executor.submit(parent) for _ in range(200))
            for future in futures:
                future.result(timeout=5)

    parent_ids: Final = frozenset(event.id for event in _events_named(profiler, "parent"))
    children: Final = _events_named(profiler, "child")
    assert len(parent_ids) == len(children) == 200
    assert frozenset(event.parent_id for event in children) == parent_ids
    assert tuple(event.id for event in profiler.events) == tuple(range(len(profiler.events)))


def test_function_usage_profiler_records_only_selected_functions() -> None:
    def selected() -> None:
        return None

    def ignored() -> None:
        return None

    source_root: Final = Path(__file__).parent
    function: Final = f"{Path(__file__).name}:{selected.__code__.co_firstlineno} {selected.__qualname__}"

    with profile_python_function_usage(source_root, frozenset((function,))) as profiler:
        selected()
        ignored()

    assert profiler.called == {function}
