import asyncio
import os
from collections.abc import AsyncIterator, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from types import ModuleType
from typing import Final, cast

import pytest
import pytest_asyncio

import litellm
from litellm import utils
from litellm.litellm_core_utils import litellm_logging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.rust_bridge.configuration import (  # pyright: ignore[reportPrivateUsage]  # share the canonical env parsing with the module under test
    _parse_env_bool,
    reset_rust_configuration,
)
from tests._prometheus_helpers import isolated_prometheus_registry
from tests.test_litellm_rust.callback_recorder import drain_logging
from tests.test_litellm_rust.integrations import (
    otel,  # noqa: F401  # pytest fixture export
    prometheus,  # noqa: F401  # pytest fixture export
    provider,  # noqa: F401  # pytest fixture export
)
from tests.test_litellm_rust.recording_server import recording_server  # noqa: F401  # pytest fixture export

CALLBACK_ATTRIBUTES: Final = (
    "callbacks",
    "input_callback",
    "success_callback",
    "failure_callback",
    "_async_input_callback",
    "_async_success_callback",
    "_async_failure_callback",
)


def _list_attribute(container: ModuleType, attribute: str) -> list[object]:
    value: Final = getattr(container, attribute)
    if not isinstance(value, list):
        raise AssertionError(f"{container.__name__}.{attribute} is not a list")
    return cast(list[object], value)


@contextmanager
def _isolated_list(container: ModuleType, attribute: str) -> Generator[None]:
    source: Final = _list_attribute(container, attribute)
    original: Final = list(source)
    source.clear()  # mutable-ok: test isolation mutates global registries by design
    try:
        yield
    finally:
        source.clear()
        source.extend(original)
        setattr(container, attribute, source)


@contextmanager
def _rebound(container: ModuleType, attribute: str, value: object) -> Generator[None]:
    original: Final[object] = getattr(container, attribute)
    setattr(container, attribute, value)
    try:
        yield
    finally:
        setattr(container, attribute, original)


@contextmanager
def _rust_mode() -> Generator[None]:
    reset_rust_configuration()
    litellm.rust(True)
    try:
        yield
    finally:
        reset_rust_configuration()


@pytest_asyncio.fixture(autouse=True, loop_scope="function")
async def isolate_rust_state() -> AsyncIterator[ExitStack]:
    with ExitStack() as stack:
        for attribute in CALLBACK_ATTRIBUTES:
            stack.enter_context(_isolated_list(litellm, attribute))
        stack.enter_context(_isolated_list(litellm_logging, "_in_memory_loggers"))  # pyright: ignore[reportPrivateUsage]  # string-name callback cache has no public accessor
        stack.enter_context(_rebound(utils, "callback_list", []))  # rebind-ok: legacy global registry mutated by set_callbacks
        stack.enter_context(_rebound(litellm, "cache", None))  # test-quality-ok: isolate the process-global cache from native extension tests
        stack.enter_context(isolated_prometheus_registry())
        stack.enter_context(_rust_mode())
        executor: Final = ThreadPoolExecutor(thread_name_prefix="rust-test-logging")
        stack.enter_context(_rebound(utils, "executor", executor))
        try:
            yield stack
        finally:
            try:
                await drain_logging()
            finally:
                await asyncio.to_thread(executor.shutdown, wait=True)
                await GLOBAL_LOGGING_WORKER.stop()


def pytest_collection_modifyitems(items):
    if not _parse_env_bool(os.environ.get("LITELLM_RUST")):
        skip = pytest.mark.skip(reason="requires LITELLM_RUST=1 and a compiled Rust extension")
        for item in items:
            item.add_marker(skip)
        return

    try:
        from litellm.rust_bridge import _native  # noqa: F401  # validates the installed extension
    except ImportError as error:
        raise pytest.UsageError("LITELLM_RUST=1 requires a compiled litellm.rust_bridge._native extension") from error
