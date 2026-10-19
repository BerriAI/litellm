from collections.abc import AsyncIterator, Generator, Iterator
from importlib import import_module
from pathlib import Path
from typing import Final

import pytest
import pytest_asyncio

from litellm.responses import dispatch
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from tests.test_litellm_rust.support.isolation import isolated_test_state, rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, recording_service

from .support import FallbackProbe

PARITY_ROOT: Final = Path(__file__).parent


@pytest_asyncio.fixture(autouse=True, loop_scope="function")
async def isolate_test_state() -> AsyncIterator[None]:
    async with isolated_test_state():
        yield


@pytest.fixture
def recording_server() -> Generator[RecordingServer]:
    with recording_service() as server:
        yield server


@pytest.fixture
def native_responses() -> Iterator[FallbackProbe]:
    probe: Final = FallbackProbe()
    python_aresponses: Final = dispatch._PYTHON_ARESPONSES  # pyright: ignore[reportPrivateUsage]  # the fallback under observation

    async def python_fallback(*args: object, **kwargs: object) -> object:
        probe.calls += 1
        return await python_aresponses(*args, **kwargs)

    with (
        rebound(catalog, "RULES", (RouteRule(Route.RESPONSES, Rollout.RUST_OPT_OUT), *catalog.RULES)),
        rebound(dispatch, "_PYTHON_ARESPONSES", python_fallback),
    ):
        yield probe


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    try:
        import_module("litellm.rust_bridge._native")
    except ImportError:
        skip: Final = pytest.mark.skip(reason="requires a compiled litellm.rust_bridge._native extension")
        for item in items:
            if PARITY_ROOT in item.path.parents:
                item.add_marker(skip)
