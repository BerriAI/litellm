"""Shared setup keeping CodSpeed measurements hermetic.

CodSpeed's callgrind instrumentation counts instructions from every thread while
a measurement window is open, and valgrind serializes all threads onto one
virtual CPU. Work deferred to litellm's shared logging executor would therefore
be attributed to whichever benchmark the valgrind scheduler resumes it under,
flipping results between runs. Running the executor inline keeps each
benchmark's cost self-contained and deterministic.

The benchmarks also disable Python's cyclic garbage collector for the
duration of the measurement window. CPython's GC is non-deterministic and
runs on its own clock; a collection triggered mid-benchmark inflates the
per-call instruction count in a way that depends on when (and whether) the
collector happened to fire rather than on anything the code under test does.
CodSpeed's per-iteration measurement is small enough (~hundreds of
microseconds) that this noise dominates the signal for the multi-turn
benchmark. ``mock_response`` already isolates the benchmarks from any
real network I/O, so the synthetic allocations here have no live-tracing
implications: deferring GC until the session ends is safe.
"""

import gc
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from typing import ParamSpec, TypeVar

import pytest

from litellm.litellm_core_utils.thread_pool_executor import executor

P = ParamSpec("P")
R = TypeVar("R")


def _submit_inline(fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> Future[R]:
    future: Future[R] = Future()
    try:
        future.set_result(fn(*args, **kwargs))
    except BaseException as exc:
        future.set_exception(exc)
    return future


@pytest.fixture(autouse=True, scope="session")
def inline_logging_executor() -> Iterator[None]:
    executor.submit = _submit_inline
    yield
    del executor.submit


@pytest.fixture(autouse=True, scope="session")
def disable_gc_during_benchmarks() -> Iterator[None]:
    """Disable CPython's cyclic GC for the duration of the benchmark session.

    CodSpeed counts instructions per measured iteration; a GC that happens to
    run mid-iteration shows up as a deterministic-looking inflation that flips
    between runs (because ``gc.collect()`` fires on its own clock). ``mock_response``
    keeps the SDK from allocating anything that escapes the benchmark loop, so
    the deferred collection at session teardown stays bounded.
    """
    gc.disable()
    try:
        yield
    finally:
        gc.enable()
        gc.collect()
