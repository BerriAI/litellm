import os

import pytest

from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    global_mcp_server_manager,
)


@pytest.fixture(autouse=True)
def _hermetic_mcp_server_registry():
    """Restore the singleton ``global_mcp_server_manager``'s registry state around every
    test, so entries seeded by one test never leak into another on a shared shard."""
    saved_registry = dict(global_mcp_server_manager.registry)
    saved_config_servers = dict(global_mcp_server_manager.config_mcp_servers)
    saved_tool_mapping = dict(global_mcp_server_manager.tool_name_to_mcp_server_name_mapping)
    saved_oauth_slots = global_mcp_server_manager._oauth_discovery_slots
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
    """Isolate MCP discovery tests from a leaked ``SERVER_ROOT_PATH``.

    ``tests/test_litellm/proxy/test_custom_proxy.py`` sets ``SERVER_ROOT_PATH`` at import time
    (its app mounts under a custom path) and never restores it, so in a shared shard the value
    leaks into this process. The discovery routes and the 401 challenges read it, so a leaked
    value would silently rewrite every ``resource_metadata`` URL and make these tests depend on
    shard ordering. Clearing it here pins the default (root-mounted) deployment; a test that
    exercises a sub-path deployment sets the value explicitly within its own body.
    """
    saved = os.environ.pop("SERVER_ROOT_PATH", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["SERVER_ROOT_PATH"] = saved
