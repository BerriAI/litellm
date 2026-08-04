"""Regression tests for MCP stateful-session owner fingerprinting (#35383)."""

import pytest

from litellm.constants import _positive_int_env
from litellm.proxy._types import UserAPIKeyAuth


@pytest.mark.asyncio
async def test_owner_fingerprint_binds_session_owner_header_to_api_key():
    """
    Shared service-account keys collapse every caller into one owner bucket.
    An explicit session-owner header must subdivide that bucket while remaining
    bound to the API key so a different key cannot hijack the session (#35383).
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _owner_fingerprint_for,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    shared_auth = UserAPIKeyAuth(api_key="shared-service-account-key")
    other_auth = UserAPIKeyAuth(api_key="different-service-account-key")
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
    fp_other_key_same_header = _owner_fingerprint_for(
        other_auth,
        request_headers={"x-litellm-mcp-session-owner": "ide-user-a"},
    )
    fp_key_only = _owner_fingerprint_for(shared_auth)

    assert fp_user_a != fp_user_b
    assert fp_user_a == fp_user_a_again
    assert fp_user_a.startswith("key+hdr:")
    assert fp_key_only.startswith("key:")
    assert fp_user_a != fp_key_only
    # Different API keys with the same owner header must not collide.
    assert fp_user_a != fp_other_key_same_header
    assert "ide-user-a" not in fp_user_a
    assert "shared-service-account-key" not in fp_user_a


@pytest.mark.asyncio
async def test_owner_fingerprint_binds_prefer_ip_to_api_key(monkeypatch):
    """
    When LITELLM_MCP_SESSION_OWNER_PREFER_IP is enabled, client IP subdivides
    the authenticated key bucket — it must not replace the key identity.
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
    other_auth = UserAPIKeyAuth(api_key="different-service-account-key")
    fp_ip_a = _owner_fingerprint_for(shared_auth, client_ip="10.0.0.1")
    fp_ip_b = _owner_fingerprint_for(shared_auth, client_ip="10.0.0.2")
    fp_other_key_same_ip = _owner_fingerprint_for(other_auth, client_ip="10.0.0.1")
    fp_key_only = _owner_fingerprint_for(shared_auth)

    assert fp_ip_a != fp_ip_b
    assert fp_ip_a.startswith("key+ip:")
    assert fp_key_only.startswith("key:")
    assert fp_ip_a != fp_key_only
    assert fp_ip_a != fp_other_key_same_ip
    assert "10.0.0.1" not in fp_ip_a


@pytest.mark.asyncio
async def test_auth_identity_fingerprint_ignores_owner_header_and_ip():
    """Hard-ceiling identity must stay stable when headers/IPs rotate."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _auth_identity_fingerprint_for,
            _owner_fingerprint_for,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    auth = UserAPIKeyAuth(api_key="shared-service-account-key")
    auth_fp = _auth_identity_fingerprint_for(auth)
    auth_fp_with_header = _auth_identity_fingerprint_for(auth)
    owner_fp = _owner_fingerprint_for(
        auth,
        request_headers={"x-litellm-mcp-session-owner": "rotating-1"},
    )

    assert auth_fp == auth_fp_with_header
    assert auth_fp.startswith("key:")
    assert owner_fp.startswith("key+hdr:")
    assert auth_fp != owner_fp


@pytest.mark.asyncio
async def test_owner_fingerprint_ignores_header_when_unauthenticated():
    """
    Unauthenticated callers must not open a new owner bucket by rotating
    x-litellm-mcp-session-owner — that would bypass both session caps.
    Fall back to IP (or anonymous) instead.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _owner_fingerprint_for,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    anon = UserAPIKeyAuth()
    fp_hdr_a = _owner_fingerprint_for(
        anon,
        client_ip="10.0.0.1",
        request_headers={"x-litellm-mcp-session-owner": "rotating-a"},
    )
    fp_hdr_b = _owner_fingerprint_for(
        anon,
        client_ip="10.0.0.1",
        request_headers={"x-litellm-mcp-session-owner": "rotating-b"},
    )
    fp_ip_only = _owner_fingerprint_for(anon, client_ip="10.0.0.1")

    assert fp_hdr_a == fp_hdr_b == fp_ip_only
    assert fp_ip_only.startswith("ip:")
    assert not fp_ip_only.startswith("hdr:")


@pytest.mark.asyncio
async def test_max_stateful_sessions_caps_are_positive():
    """Session caps must come from env helpers and stay strictly positive."""
    try:
        from litellm.constants import (
            MCP_MAX_STATEFUL_SESSIONS_PER_AUTH_IDENTITY,
            MCP_MAX_STATEFUL_SESSIONS_PER_OWNER,
        )
        from litellm.proxy._experimental.mcp_server import server as mcp_server
    except ImportError:
        pytest.skip("MCP server not available")

    assert mcp_server._MAX_STATEFUL_SESSIONS_PER_OWNER == MCP_MAX_STATEFUL_SESSIONS_PER_OWNER
    assert mcp_server._MAX_STATEFUL_SESSIONS_PER_AUTH_IDENTITY == MCP_MAX_STATEFUL_SESSIONS_PER_AUTH_IDENTITY
    assert MCP_MAX_STATEFUL_SESSIONS_PER_OWNER > 0
    assert MCP_MAX_STATEFUL_SESSIONS_PER_AUTH_IDENTITY >= MCP_MAX_STATEFUL_SESSIONS_PER_OWNER


def test_positive_int_env_rejects_non_positive_and_invalid(monkeypatch):
    """Invalid / non-positive env values must fall back to the default."""
    monkeypatch.setenv("LITELLM_TEST_POSITIVE_INT", "0")
    assert _positive_int_env("LITELLM_TEST_POSITIVE_INT", 100) == 100
    monkeypatch.setenv("LITELLM_TEST_POSITIVE_INT", "-5")
    assert _positive_int_env("LITELLM_TEST_POSITIVE_INT", 100) == 100
    monkeypatch.setenv("LITELLM_TEST_POSITIVE_INT", "not-an-int")
    assert _positive_int_env("LITELLM_TEST_POSITIVE_INT", 100) == 100
    monkeypatch.setenv("LITELLM_TEST_POSITIVE_INT", "42")
    assert _positive_int_env("LITELLM_TEST_POSITIVE_INT", 100) == 42
    monkeypatch.delenv("LITELLM_TEST_POSITIVE_INT", raising=False)
    assert _positive_int_env("LITELLM_TEST_POSITIVE_INT", 100) == 100


@pytest.mark.asyncio
async def test_enforce_auth_identity_ceiling_evicts_across_owner_sub_buckets():
    """
    Rotating session-owner headers under one API key must still hit the
    per-auth-identity hard ceiling (Veria / Greptile concern).
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    try:
        from litellm.proxy._experimental.mcp_server import server as mcp_server
    except ImportError:
        pytest.skip("MCP server not available")

    auth_identity = "key:shared-auth"
    owners = {
        "s0": "key+hdr:owner-0",
        "s1": "key+hdr:owner-1",
        "s2": "key+hdr:owner-2",
    }
    identities = {sid: auth_identity for sid in owners}
    last_seen = {"s0": 1.0, "s1": 2.0, "s2": 3.0}
    mock_transport = MagicMock()
    mock_transport.terminate = AsyncMock()
    server_instances = {sid: mock_transport for sid in owners}

    with (
        patch.object(mcp_server, "_MAX_STATEFUL_SESSIONS_PER_AUTH_IDENTITY", 2),
        patch.dict(mcp_server._stateful_session_owners, owners, clear=True),
        patch.dict(mcp_server._stateful_session_auth_identities, identities, clear=True),
        patch.dict(mcp_server._stateful_session_auth_context_last_seen, last_seen, clear=True),
        patch.dict(mcp_server._stateful_session_active_request_counts, {}, clear=True),
        patch.object(
            mcp_server.session_manager_stateful,
            "_server_instances",
            server_instances,
        ),
    ):
        allowed = await mcp_server._enforce_stateful_session_cap_for_auth_identity(auth_identity)
        assert allowed is True
        # Evicts oldest idle sessions until there is room for one new initialize
        # (ceiling 2 → leave 1 live session).
        assert "s0" not in mcp_server._stateful_session_auth_identities
        assert "s1" not in mcp_server._stateful_session_auth_identities
        assert "s2" in mcp_server._stateful_session_auth_identities
        assert len(mcp_server._stateful_session_auth_identities) == 1
        mock_transport.terminate.assert_awaited()


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
