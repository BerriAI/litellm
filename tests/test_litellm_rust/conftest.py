import asyncio
import os
from collections.abc import AsyncIterator, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, asynccontextmanager, contextmanager
from threading import Lock
from types import ModuleType
from typing import Final, Literal, cast

import pytest
import pytest_asyncio
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import litellm
from litellm import utils
from litellm.litellm_core_utils import litellm_logging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.rust_bridge.configuration import (  # pyright: ignore[reportPrivateUsage]  # preserve raw configuration state in test isolation
    _CONFIGURATION,
    _parse_env_bool,
)
from tests._prometheus_helpers import isolated_prometheus_registry
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
Backend = Literal["python", "rust"]
EXPECTED_FAILURE_FILES: Final = frozenset(
    {
        "chat/test_callback_mutation.py",
        "integrations/test_backend_parity.py",
        "integrations/test_callback_lifecycle.py",
        "integrations/test_composition.py",
        "integrations/test_exporters.py",
        "integrations/test_guardrails.py",
        "messages/test_callback_mutation.py",
        "messages/test_streaming.py",
        "ocr/test_callbacks.py",
        "ocr/test_dispatch.py",
        "ocr/test_requests.py",
        "test_provenance.py",
    }
)


class _BackendScopes:
    def __init__(self) -> None:
        self._owner: asyncio.Task[object] | None = None
        self._lock: Final = Lock()

    @contextmanager
    def enter(self) -> Generator[None]:
        task: Final = asyncio.current_task()
        with self._lock:
            previous: Final = self._owner
            if previous is not None and previous is not task:
                raise RuntimeError("isolated_backend scopes cannot overlap across tasks; run them sequentially")
            self._owner = task
        try:
            yield
        finally:
            with self._lock:
                self._owner = previous


_BACKEND_SCOPES: Final = _BackendScopes()


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
def _rebound(container: object, attribute: str, value: object) -> Generator[None]:
    original: Final[object] = getattr(container, attribute)
    setattr(container, attribute, value)
    try:
        yield
    finally:
        setattr(container, attribute, original)


@contextmanager
def _rust_mode(enabled: bool) -> Generator[None]:
    with _rebound(_CONFIGURATION, "override", enabled):
        yield


@asynccontextmanager
async def isolated_backend(backend: Backend) -> AsyncIterator[ExitStack]:
    with _BACKEND_SCOPES.enter():
        async with _isolated_backend(backend) as stack:
            yield stack


@asynccontextmanager
async def _isolated_backend(backend: Backend) -> AsyncIterator[ExitStack]:
    with ExitStack() as stack:
        for attribute in CALLBACK_ATTRIBUTES:
            stack.enter_context(_isolated_list(litellm, attribute))
        stack.enter_context(_isolated_list(litellm_logging, "_in_memory_loggers"))  # pyright: ignore[reportPrivateUsage]  # string-name callback cache has no public accessor
        stack.enter_context(
            _rebound(utils, "callback_list", [])
        )  # rebind-ok: legacy global registry mutated by set_callbacks
        stack.enter_context(
            _rebound(litellm, "cache", None)
        )  # test-quality-ok: isolate the process-global cache from native extension tests
        stack.enter_context(isolated_prometheus_registry())
        stack.enter_context(_rust_mode(backend == "rust"))
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


@pytest_asyncio.fixture(autouse=True, loop_scope="function")
async def isolate_rust_state() -> AsyncIterator[ExitStack]:
    # pytest-asyncio runs fixture setup and the test body in different tasks.
    async with _isolated_backend("rust") as stack:
        yield stack


@pytest.fixture
def recording_server() -> Generator[RecordingServer]:
    with recording_service() as server:
        yield server


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    expected_failure: Final = pytest.mark.xfail(
        reason="requires the retained callback implementation from #40070",
        strict=False,
    )
    for item in items:
        if "test_litellm_rust" not in item.path.parts:
            continue
        relative_path: Final = "/".join(item.path.parts[item.path.parts.index("test_litellm_rust") + 1 :])
        if relative_path in EXPECTED_FAILURE_FILES:
            item.add_marker(expected_failure)

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
