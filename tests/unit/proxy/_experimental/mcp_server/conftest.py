import asyncio
import importlib
import os

import pytest

import litellm
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER


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
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    importlib.reload(litellm)
    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    yield

    # Teardown code (executes after the yield point)
    # LoggingWorker carries still-queued coroutines onto the next test's loop, where they'd log into that test's callbacks
    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    loop.close()  # Close the loop created earlier
    asyncio.set_event_loop(None)  # Remove the reference to the loop


@pytest.fixture(scope="function", autouse=True)
async def drain_logging_worker():
    """
    The logging queue is bound to the running loop, so anything left queued when a test's loop
    goes away is carried onto the next test's loop and fires against its callbacks.
    """
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    yield

    try:
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.clear_queue(), timeout=10)
    except asyncio.TimeoutError:
        pass


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


@pytest.fixture
def config_only_mcp_manager_factory():
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager

    class ConfigOnlyManager(MCPServerManager):
        def initialize_tool_name_to_mcp_server_name_mapping(self):
            return None

    return ConfigOnlyManager


@pytest.fixture(autouse=True)
def _hermetic_mcp_server_registry():
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    saved_registry = dict(global_mcp_server_manager.registry)
    saved_config_servers = dict(global_mcp_server_manager.config_mcp_servers)
    saved_tool_mapping = dict(global_mcp_server_manager.tool_name_to_mcp_server_name_mapping)
    saved_oauth_slots = global_mcp_server_manager._oauth_discovery_slots
    global_mcp_server_manager.registry.clear()
    global_mcp_server_manager.config_mcp_servers.clear()
    global_mcp_server_manager.tool_name_to_mcp_server_name_mapping.clear()
    global_mcp_server_manager._oauth_discovery_slots = ()
    try:
        yield
    finally:
        global_mcp_server_manager.registry.clear()
        global_mcp_server_manager.registry.update(saved_registry)
        global_mcp_server_manager.config_mcp_servers.clear()
        global_mcp_server_manager.config_mcp_servers.update(saved_config_servers)
        global_mcp_server_manager.tool_name_to_mcp_server_name_mapping.clear()
        global_mcp_server_manager.tool_name_to_mcp_server_name_mapping.update(saved_tool_mapping)
        global_mcp_server_manager._oauth_discovery_slots = saved_oauth_slots


@pytest.fixture(autouse=True)
def _hermetic_server_root_path():
    saved = os.environ.pop("SERVER_ROOT_PATH", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["SERVER_ROOT_PATH"] = saved


@pytest.fixture
def _mcp_request_ctx():
    def _mcp_request_ctx(**overrides):
        from types import SimpleNamespace

        from mcp.server.context import ServerRequestContext

        kwargs = {
            "session": SimpleNamespace(),
            "lifespan_context": {},
            "protocol_version": "2025-06-18",
            "method": "",
            "params": None,
            "request_id": 1,
            "meta": None,
            "request": None,
        }
        kwargs.update(overrides)
        return ServerRequestContext(**kwargs)

    return _mcp_request_ctx
