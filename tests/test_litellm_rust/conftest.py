import asyncio
from collections.abc import AsyncIterator, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from importlib import import_module
from typing import Final

import pytest
import pytest_asyncio

import litellm
from litellm import utils
from litellm.litellm_core_utils import litellm_logging, thread_pool_executor
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.rust_bridge.configuration import _CONFIGURATION
from tests.test_litellm_rust.support.caching import (
    cache as cache,
)
from tests.test_litellm_rust.support.caching import (
    call as call,
)
from tests.test_litellm_rust.support.caching import (
    calls as calls,
)
from tests.test_litellm_rust.support.caching import (
    engine as engine,
)
from tests.test_litellm_rust.support.caching import (
    fake_redis as fake_redis,
)
from tests.test_litellm_rust.support.caching import redis_url as redis_url
from tests.test_litellm_rust.support.callback_recorder import drain_logging
from tests.test_litellm_rust.support.clickhouse import clickhouse_url as clickhouse_url
from tests.test_litellm_rust.support.isolation import isolated_callback_registries, rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, recording_service


@pytest_asyncio.fixture(autouse=True, loop_scope="function")
async def isolate_test_state() -> AsyncIterator[None]:
    with ExitStack() as stack:
        stack.enter_context(isolated_callback_registries())
        stack.enter_context(rebound(litellm, "cache", None))  # test-quality-ok: isolate process-global cache
        stack.enter_context(rebound(_CONFIGURATION, "override", None))
        executor: Final = ThreadPoolExecutor(thread_name_prefix="rust-test-logging")
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


@pytest.fixture
def recording_server() -> Generator[RecordingServer]:
    with recording_service() as server:
        yield server


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    try:
        import_module("litellm.rust_bridge._native")
    except ImportError:
        skip: Final = pytest.mark.skip(reason="requires a compiled litellm.rust_bridge._native extension")
        for item in items:
            if "test_litellm_rust" in item.path.parts:
                item.add_marker(skip)


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
