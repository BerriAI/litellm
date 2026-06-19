"""Tests for the v2 egress manager factory + skeleton (step 6a)."""

from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
    _make_global_mcp_server_manager,
)
from litellm.proxy._experimental.mcp_server.mcp_server_manager_v2 import (
    MCPServerManagerV2,
)


def test_v2_is_a_manager_subclass():
    assert issubclass(MCPServerManagerV2, MCPServerManager)


def test_factory_constructs_the_v2_manager():
    # v2 is the egress implementation; there is no opt-in flag.
    assert isinstance(_make_global_mcp_server_manager(), MCPServerManagerV2)
