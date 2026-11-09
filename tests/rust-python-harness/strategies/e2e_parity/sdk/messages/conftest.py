from collections.abc import AsyncIterator, Generator
from importlib import import_module
from pathlib import Path
from typing import Final

import pytest
import pytest_asyncio

from tests.test_litellm_rust.support.isolation import isolated_test_state
from tests.test_litellm_rust.support.recording_server import RecordingServer, recording_service

PARITY_ROOT: Final = Path(__file__).parent


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
            if PARITY_ROOT in item.path.parents:
                item.add_marker(skip)
