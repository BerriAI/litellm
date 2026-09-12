import asyncio
import os
from collections.abc import AsyncIterator, Generator, Iterator
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
from litellm.rust_bridge.configuration import (  # pyright: ignore[reportPrivateUsage]  # preserve raw configuration state in test isolation
    _CONFIGURATION,
    _parse_env_bool,
)
from tests.test_litellm_rust.support.callback_recorder import drain_logging
from tests.test_litellm_rust.support.recording_server import RecordingServer, recording_service

CALLBACK_ATTRIBUTES: Final = (
    "callbacks",
    "input_callback",
    "success_callback",
    "failure_callback",
    "_async_input_callback",
    "_async_success_callback",
    "_async_failure_callback",
)
EXPECTED_FAILURE_REASONS: Final = {
    "ocr/test_callbacks.py": "requires the OCR callback lifecycle implementation from #40070",
    "ocr/test_guardrails.py": "requires the OCR guardrail lifecycle implementation from #40070",
    "ocr/test_requests.py": "requires the OCR request and Azure authentication implementation from #40070",
}


def _list_attribute(container: ModuleType, attribute: str) -> list[object]:
    value: Final = getattr(container, attribute)
    if not isinstance(value, list):
        raise AssertionError(f"{container.__name__}.{attribute} is not a list")
    return cast(list[object], value)


@contextmanager
def _isolated_list(container: ModuleType, attribute: str) -> Iterator[None]:
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
def _rebound(container: object, attribute: str, value: object) -> Iterator[None]:
    original: Final[object] = getattr(container, attribute)
    setattr(container, attribute, value)
    try:
        yield
    finally:
        setattr(container, attribute, original)


@pytest_asyncio.fixture(autouse=True, loop_scope="function")
async def isolate_ocr_test_state() -> AsyncIterator[None]:
    with ExitStack() as stack:
        for attribute in CALLBACK_ATTRIBUTES:
            stack.enter_context(_isolated_list(litellm, attribute))
        stack.enter_context(_isolated_list(litellm_logging, "_in_memory_loggers"))  # pyright: ignore[reportPrivateUsage]  # no public callback-cache accessor
        stack.enter_context(_rebound(utils, "callback_list", []))  # rebind-ok: isolate legacy callback registry
        stack.enter_context(_rebound(litellm, "cache", None))  # test-quality-ok: isolate process-global cache
        stack.enter_context(_rebound(_CONFIGURATION, "override", None))
        executor: Final = ThreadPoolExecutor(thread_name_prefix="rust-ocr-test-logging")
        stack.enter_context(_rebound(utils, "executor", executor))
        try:
            yield
        finally:
            try:
                await drain_logging()
            finally:
                await asyncio.to_thread(executor.shutdown, wait=True)
                await GLOBAL_LOGGING_WORKER.stop()


@pytest.fixture
def recording_server() -> Generator[RecordingServer]:
    with recording_service() as server:
        yield server


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "test_litellm_rust" not in item.path.parts:
            continue
        relative_path: Final = "/".join(item.path.parts[item.path.parts.index("test_litellm_rust") + 1 :])
        reason: Final = EXPECTED_FAILURE_REASONS.get(relative_path)
        if reason is not None:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=False))

    if not _parse_env_bool(os.environ.get("LITELLM_RUST")):
        skip: Final = pytest.mark.skip(reason="requires LITELLM_RUST=1 and a compiled Rust extension")
        for item in items:
            if "test_litellm_rust" in item.path.parts:
                item.add_marker(skip)
        return

    try:
        from litellm.rust_bridge import _native  # noqa: F401  # validates the installed extension
    except ImportError as error:
        raise pytest.UsageError("LITELLM_RUST=1 requires a compiled litellm.rust_bridge._native extension") from error


@pytest.fixture
def isolated_azure_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AZURE_AI_API_KEY",
        "AZURE_AI_API_BASE",
        "AZURE_AD_TOKEN",
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "AZURE_USERNAME",
        "AZURE_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "enable_azure_ad_token_refresh", False)
