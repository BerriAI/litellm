"""Shared setup keeping CodSpeed measurements hermetic.

CodSpeed's callgrind instrumentation counts instructions from every thread while
a measurement window is open, and valgrind serializes all threads onto one
virtual CPU. Work deferred to litellm's shared logging executor would therefore
be attributed to whichever benchmark the valgrind scheduler resumes it under,
flipping results between runs. Running the executor inline keeps each
benchmark's cost self-contained and deterministic.
"""

import os
import sys
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from pathlib import Path
from typing import ParamSpec, TypeVar

import pytest

from litellm.litellm_core_utils.thread_pool_executor import executor

P = ParamSpec("P")
R = TypeVar("R")


def pytest_configure(config: pytest.Config) -> None:
    if os.environ.get("LITELLM_REQUIRE_INSTALLED_WHEEL") != "1":
        return

    import litellm
    import litellm.rust_bridge._native as native

    prefix = Path(sys.prefix).resolve()
    for name, module_file in (("litellm", litellm.__file__), ("litellm.rust_bridge._native", native.__file__)):
        path = Path(module_file).resolve()
        if not path.is_relative_to(prefix):
            raise pytest.UsageError(f"{name} resolved outside the benchmark environment: {path}")
        print(f"{name}: {path}")  # noqa: T201  # provenance evidence must be visible in CI logs


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
