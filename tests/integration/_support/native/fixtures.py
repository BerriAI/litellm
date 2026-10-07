import asyncio
import threading
from collections.abc import AsyncIterator, Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from typing import Final

import fakeredis
import pytest
import pytest_asyncio

import litellm
from litellm import utils
from litellm.litellm_core_utils import litellm_logging, thread_pool_executor
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.rust_bridge.configuration import _CONFIGURATION
from tests._support.recording_server import RecordingServer, recording_service
from tests.integration._support.native.callback_recorder import drain_logging
from tests.integration._support.native.isolation import isolated_callback_registries, rebound


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


@pytest.fixture
def recording_server() -> Generator[RecordingServer]:
    with recording_service() as server:
        yield server


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


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


@pytest.fixture
def redis_url() -> Generator[str]:
    server: Final = fakeredis.TcpFakeServer(("127.0.0.1", 0), server_type="redis")
    worker: Final = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"redis://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
