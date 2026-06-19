"""Tests for the v2 egress manager factory + skeleton (step 6a)."""

from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    MCPServerManager,
    _make_global_mcp_server_manager,
)
from litellm.proxy._experimental.mcp_server.mcp_server_manager_v2 import (
    MCPServerManagerV2,
)

FLAG = "LITELLM_USE_V2_MCP_EGRESS"


def test_v2_is_a_manager_subclass():
    assert issubclass(MCPServerManagerV2, MCPServerManager)


def test_factory_returns_v2_when_egress_enabled(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    assert isinstance(_make_global_mcp_server_manager(), MCPServerManagerV2)


def test_factory_returns_v1_when_egress_disabled(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    manager = _make_global_mcp_server_manager()
    assert isinstance(manager, MCPServerManager)
    assert not isinstance(manager, MCPServerManagerV2)
