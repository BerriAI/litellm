import asyncio
from collections.abc import AsyncIterator, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from types import ModuleType
from typing import Final, cast

import pytest_asyncio

import litellm
from litellm import utils
from litellm.litellm_core_utils import litellm_logging, thread_pool_executor
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.rust_bridge.configuration import _CONFIGURATION
from tests._support.callback_recorder import drain_logging

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
    source.clear()
    try:
        yield
    finally:
        source.clear()
        source.extend(original)
        setattr(container, attribute, source)


@contextmanager
def rebound(container: object, attribute: str, value: object) -> Generator[None]:
    original: Final[object] = getattr(container, attribute)
    setattr(container, attribute, value)
    try:
        yield
    finally:
        setattr(container, attribute, original)


@contextmanager
def isolated_callback_registries() -> Generator[None]:
    with ExitStack() as stack:
        for attribute in CALLBACK_ATTRIBUTES:
            stack.enter_context(_isolated_list(litellm, attribute))
        stack.enter_context(_isolated_list(litellm_logging, "_in_memory_loggers"))
        stack.enter_context(rebound(utils, "callback_list", []))
        yield


@pytest_asyncio.fixture(autouse=True, loop_scope="function")
async def isolate_ocr_test_state() -> AsyncIterator[None]:
    with ExitStack() as stack:
        stack.enter_context(isolated_callback_registries())
        stack.enter_context(rebound(litellm, "cache", None))  # test-quality-ok: isolate process-global cache
        stack.enter_context(rebound(_CONFIGURATION, "override", None))
        executor: Final = ThreadPoolExecutor(thread_name_prefix="rust-ocr-test-logging")
        stack.enter_context(rebound(litellm_logging, "executor", executor))
        stack.enter_context(rebound(utils, "executor", executor))
        stack.enter_context(rebound(thread_pool_executor, "executor", executor))
        try:
            yield
        finally:
            try:
                await drain_logging()
            finally:
                await asyncio.to_thread(executor.shutdown, wait=True)
                await GLOBAL_LOGGING_WORKER.stop()
