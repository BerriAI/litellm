"""Regression tests for MCP stateful-session owner fingerprinting (#35383)."""

import pytest

from litellm.proxy._types import UserAPIKeyAuth


@pytest.mark.asyncio
async def test_owner_fingerprint_prefers_session_owner_header_over_shared_api_key():
    """
    Shared service-account keys collapse every caller into one owner bucket.
    An explicit session-owner header must take priority so IDE / plugin users
    behind the same key get independent session caps (#35383).
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _owner_fingerprint_for,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    shared_auth = UserAPIKeyAuth(api_key="shared-service-account-key")
    fp_user_a = _owner_fingerprint_for(
        shared_auth,
        request_headers={"x-litellm-mcp-session-owner": "ide-user-a"},
    )
    fp_user_b = _owner_fingerprint_for(
        shared_auth,
        request_headers={"X-LiteLLM-MCP-Session-Owner": "ide-user-b"},
    )
    fp_user_a_again = _owner_fingerprint_for(
        shared_auth,
        request_headers={"x-litellm-mcp-session-owner": "ide-user-a"},
    )
    fp_key_only = _owner_fingerprint_for(shared_auth)

    assert fp_user_a != fp_user_b
    assert fp_user_a == fp_user_a_again
    assert fp_user_a.startswith("hdr:")
    assert fp_key_only.startswith("key:")
    assert fp_user_a != fp_key_only
    assert "ide-user-a" not in fp_user_a
    assert "shared-service-account-key" not in fp_user_a


@pytest.mark.asyncio
async def test_owner_fingerprint_prefer_ip_over_shared_api_key(monkeypatch):
    """
    When LITELLM_MCP_SESSION_OWNER_PREFER_IP is enabled, client IP must win
    over a shared API key so distinct source IPs get separate session caps.
    """
    try:
        from litellm.proxy._experimental.mcp_server import server as mcp_server
        from litellm.proxy._experimental.mcp_server.server import (
            _owner_fingerprint_for,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    monkeypatch.setattr(mcp_server, "MCP_SESSION_OWNER_PREFER_IP", True)

    shared_auth = UserAPIKeyAuth(api_key="shared-service-account-key")
    fp_ip_a = _owner_fingerprint_for(shared_auth, client_ip="10.0.0.1")
    fp_ip_b = _owner_fingerprint_for(shared_auth, client_ip="10.0.0.2")
    fp_key_only = _owner_fingerprint_for(shared_auth)

    assert fp_ip_a != fp_ip_b
    assert fp_ip_a.startswith("ip:")
    assert fp_key_only.startswith("key:")
    assert fp_ip_a != fp_key_only
    assert "10.0.0.1" not in fp_ip_a


@pytest.mark.asyncio
async def test_max_stateful_sessions_per_owner_reads_from_env_constant():
    """Session cap must come from LITELLM_MCP_MAX_STATEFUL_SESSIONS_PER_OWNER."""
    try:
        from litellm.constants import MCP_MAX_STATEFUL_SESSIONS_PER_OWNER
        from litellm.proxy._experimental.mcp_server import server as mcp_server
    except ImportError:
        pytest.skip("MCP server not available")

    assert mcp_server._MAX_STATEFUL_SESSIONS_PER_OWNER == MCP_MAX_STATEFUL_SESSIONS_PER_OWNER
    assert isinstance(mcp_server._MAX_STATEFUL_SESSIONS_PER_OWNER, int)
    assert mcp_server._MAX_STATEFUL_SESSIONS_PER_OWNER > 0


@pytest.mark.asyncio
async def test_owner_fingerprint_default_still_uses_api_key_when_no_overrides():
    """Default behavior must remain key-based when header/prefer-ip are unused."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _owner_fingerprint_for,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    auth = UserAPIKeyAuth(api_key="custom-master-key")
    fp = _owner_fingerprint_for(auth, client_ip="10.0.0.1")

    assert fp.startswith("key:")
    assert "custom-master-key" not in fp
    assert "10.0.0.1" not in fp
