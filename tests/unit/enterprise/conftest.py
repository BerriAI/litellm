# conftest.py

import asyncio
from typing import Final

import pytest

import litellm
import litellm.proxy.proxy_server
from tests._module_state import collect_flushable_caches, flush_caches, restore_mutable_state, snapshot_mutable_state

_LITELLM_STATE: Final = snapshot_mutable_state(litellm)
_PROXY_SERVER_STATE: Final = snapshot_mutable_state(litellm.proxy.proxy_server)
_FLUSHABLE_CACHES: Final = collect_flushable_caches((litellm, litellm.proxy.proxy_server))


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    restore_mutable_state(litellm, _LITELLM_STATE)
    restore_mutable_state(litellm.proxy.proxy_server, _PROXY_SERVER_STATE)
    flush_caches(_FLUSHABLE_CACHES)

    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    loop.close()
    asyncio.set_event_loop(None)


def pytest_collection_modifyitems(config, items):
    # Separate tests in 'test_amazing_proxy_custom_logger.py' and other tests
    custom_logger_tests = [
        item for item in items if "custom_logger" in item.parent.name
    ]
    other_tests = [item for item in items if "custom_logger" not in item.parent.name]

    # Sort tests based on their names
    custom_logger_tests.sort(key=lambda x: x.name)
    other_tests.sort(key=lambda x: x.name)

    # Reorder the items list
    items[:] = custom_logger_tests + other_tests
