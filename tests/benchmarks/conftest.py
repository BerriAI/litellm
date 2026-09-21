"""Shared setup keeping CodSpeed measurements hermetic.

CodSpeed's callgrind instrumentation counts instructions from every thread while
a measurement window is open, and valgrind serializes all threads onto one
virtual CPU. Work deferred to litellm's shared logging executor would therefore
be attributed to whichever benchmark the valgrind scheduler resumes it under,
flipping results between runs. Running the executor inline keeps each
benchmark's cost self-contained and deterministic.
"""

import importlib.metadata
import json
import os
import sys
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future
from pathlib import Path
from typing import Final, ParamSpec, TypeVar, cast

import pytest

from litellm.litellm_core_utils.thread_pool_executor import executor

P = ParamSpec("P")
R = TypeVar("R")


def pytest_configure(config: pytest.Config) -> None:
    if os.environ.get("LITELLM_REQUIRE_INSTALLED_WHEEL") != "1":
        return

    import litellm
    import litellm.rust_bridge._native as native

    prefix: Final = Path(sys.prefix).resolve()
    module_paths: Final = (
        ("litellm", Path(litellm.__file__).resolve()),
        ("litellm.rust_bridge._native", Path(native.__file__).resolve()),
    )
    for module_name, module_path in module_paths:
        sys.stdout.write(f"{module_name}: {module_path}\n")

    path_failures: Final = tuple(
        f"{module_name} is not under sys.prefix: {module_path}"
        for module_name, module_path in module_paths
        if not module_path.is_relative_to(prefix)
    )
    if path_failures:
        raise pytest.UsageError("; ".join(path_failures))

    direct_url: Final = importlib.metadata.distribution("litellm").read_text("direct_url.json")
    if direct_url is None:
        return

    try:
        direct_url_data: Final = cast(object, json.loads(direct_url))
    except json.JSONDecodeError as error:
        raise pytest.UsageError("litellm distribution direct_url.json is invalid JSON") from error

    if isinstance(direct_url_data, Mapping):
        direct_url_mapping: Final = cast(Mapping[str, object], direct_url_data)
        dir_info: Final = direct_url_mapping.get("dir_info")
        if isinstance(dir_info, Mapping) and cast(Mapping[str, object], dir_info).get("editable") is True:
            raise pytest.UsageError("litellm distribution is editable")


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
