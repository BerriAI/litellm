from collections.abc import AsyncIterator, Generator
from importlib import import_module
from typing import Final

import pytest
import pytest_asyncio

import litellm
from tests.test_litellm_rust.support.clickhouse import clickhouse_url as clickhouse_url
from tests.test_litellm_rust.support.isolation import isolated_test_state
from tests.test_litellm_rust.support.recording_server import RecordingServer, recording_service


@pytest_asyncio.fixture(autouse=True, loop_scope="function")
async def isolate_test_state() -> AsyncIterator[None]:
    async with isolated_test_state():
        yield


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
