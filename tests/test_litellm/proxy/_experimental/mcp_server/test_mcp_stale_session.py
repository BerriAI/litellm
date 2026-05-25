"""
Tests for MCP stale session ID handling (Fixes #20292).

When clients reconnect to LiteLLM's MCP endpoint after a server restart or reload,
they may send a stale `mcp-session-id` header. This test verifies that:
1. For non-DELETE requests: stale session IDs are stripped so new sessions are created
2. For DELETE requests: idempotent behavior returns success even if session doesn't exist
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from litellm.types.mcp import MCPAuth


class TestHandleStaleMcpSession:
    """Unit tests for the _handle_stale_mcp_session helper."""

    @pytest.mark.asyncio
    async def test_strips_stale_session_id_for_non_delete(self):
        """Non-DELETE requests should have stale session IDs stripped."""
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                _handle_stale_mcp_session,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        scope = {
            "method": "POST",
            "headers": [
                (b"content-type", b"application/json"),
                (b"mcp-session-id", b"stale-id"),
            ],
        }
        receive = AsyncMock()
        send = AsyncMock()
        mgr = MagicMock()
        mgr._server_instances = {}  # no active sessions

        handled = await _handle_stale_mcp_session(scope, receive, send, mgr)

        # Should not be fully handled (returns False)
        assert handled is False
        # Header should be stripped
        header_names = [k for k, _ in scope["headers"]]
        assert b"mcp-session-id" not in header_names

    @pytest.mark.asyncio
    async def test_delete_stale_session_returns_success(self):
        """DELETE requests for non-existent sessions should return success (idempotent)."""
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                _handle_stale_mcp_session,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        scope = {
            "type": "http",
            "method": "DELETE",
            "headers": [
                (b"content-type", b"application/json"),
                (b"mcp-session-id", b"stale-id"),
            ],
        }
        receive = AsyncMock()
        send = AsyncMock()
        mgr = MagicMock()
        mgr._server_instances = {}  # no active sessions

        handled = await _handle_stale_mcp_session(scope, receive, send, mgr)

        # Should be fully handled (returns True)
        assert handled is True
        # Should have sent a success response
        assert send.called
        # Header should NOT be stripped (DELETE needs the session ID)
        header_names = [k for k, _ in scope["headers"]]
        assert b"mcp-session-id" in header_names

    @pytest.mark.asyncio
    async def test_preserves_valid_session_id(self):
        """Valid session IDs should not be modified."""
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                _handle_stale_mcp_session,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        scope = {
            "method": "POST",
            "headers": [
                (b"content-type", b"application/json"),
                (b"mcp-session-id", b"valid-id"),
            ],
        }
        receive = AsyncMock()
        send = AsyncMock()
        mgr = MagicMock()
        mgr._server_instances = {"valid-id": MagicMock()}

        handled = await _handle_stale_mcp_session(scope, receive, send, mgr)

        # Should not be handled (returns False)
        assert handled is False
        # Header should be preserved
        header_names = [k for k, _ in scope["headers"]]
        assert b"mcp-session-id" in header_names

    @pytest.mark.asyncio
    async def test_no_op_when_no_session_header(self):
        """No session header should result in no-op."""
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                _handle_stale_mcp_session,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        scope = {
            "method": "POST",
            "headers": [
                (b"content-type", b"application/json"),
            ],
        }
        receive = AsyncMock()
        send = AsyncMock()
        mgr = MagicMock()
        mgr._server_instances = {}

        handled = await _handle_stale_mcp_session(scope, receive, send, mgr)

        assert handled is False
        assert len(scope["headers"]) == 1

    @pytest.mark.asyncio
    async def test_no_op_when_server_instances_missing(self):
        """If _server_instances attr doesn't exist, don't crash."""
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                _handle_stale_mcp_session,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        scope = {
            "method": "POST",
            "headers": [
                (b"mcp-session-id", b"some-id"),
            ],
        }
        receive = AsyncMock()
        send = AsyncMock()
        mgr = MagicMock(spec=[])  # no attributes

        handled = await _handle_stale_mcp_session(scope, receive, send, mgr)

        # Should not be handled, header should be kept
        assert handled is False
        header_names = [k for k, _ in scope["headers"]]
        assert b"mcp-session-id" in header_names

    @pytest.mark.asyncio
    async def test_delete_valid_session_not_handled(self):
        """DELETE requests for existing sessions should not be intercepted."""
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                _handle_stale_mcp_session,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        scope = {
            "method": "DELETE",
            "headers": [
                (b"mcp-session-id", b"valid-id"),
            ],
        }
        receive = AsyncMock()
        send = AsyncMock()
        mgr = MagicMock()
        mgr._server_instances = {"valid-id": MagicMock()}

        handled = await _handle_stale_mcp_session(scope, receive, send, mgr)

        # Should not be handled - let session manager handle it
        assert handled is False
        # Should not have sent any response
        assert not send.called


@pytest.mark.asyncio
async def test_stale_mcp_session_id_is_stripped():
    """
    When the mcp-session-id header references a session that no longer exists,
    handle_streamable_http_mcp should strip the header before forwarding the
    request to the session manager so a fresh session is created.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    stale_session_id = "stale-session-id-12345"

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [
            (b"content-type", b"application/json"),
            (b"mcp-session-id", stale_session_id.encode()),
            (b"authorization", b"Bearer test-key"),
        ],
    }

    receive = AsyncMock()
    send = AsyncMock()

    # Simulate: session manager has NO sessions (the stale one was cleaned up)
    captured_scope = {}

    async def mock_handle_request(s, r, se):
        # Capture the scope that was actually passed
        captured_scope.update(s)

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(MagicMock(), None, None, None, None, None),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.set_auth_context",
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
            True,
        ),
        patch.object(
            session_manager,
            "handle_request",
            side_effect=mock_handle_request,
        ),
        patch.object(
            session_manager,
            "_server_instances",
            {},  # Empty dict = no active sessions
        ),
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    # Verify the mcp-session-id header was stripped
    header_names = [k for k, v in captured_scope.get("headers", [])]
    assert (
        b"mcp-session-id" not in header_names
    ), "Stale mcp-session-id header should have been stripped from the scope"


@pytest.mark.asyncio
async def test_delete_stale_mcp_session_returns_success():
    """
    When a DELETE request is made for a session that no longer exists,
    handle_streamable_http_mcp should return success (200) immediately
    without forwarding to the session manager (idempotent DELETE).
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    stale_session_id = "stale-session-id-12345"

    scope = {
        "type": "http",
        "method": "DELETE",
        "path": "/mcp",
        "headers": [
            (b"content-type", b"application/json"),
            (b"mcp-session-id", stale_session_id.encode()),
            (b"authorization", b"Bearer test-key"),
        ],
    }

    receive = AsyncMock()
    send = AsyncMock()

    # Mock handle_request should NOT be called for stale DELETE
    mock_handle_request = AsyncMock()

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(MagicMock(), None, None, None, None, None),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.set_auth_context",
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
            True,
        ),
        patch.object(
            session_manager,
            "handle_request",
            side_effect=mock_handle_request,
        ),
        patch.object(
            session_manager,
            "_server_instances",
            {},  # Empty dict = no active sessions
        ),
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    # Verify session manager was NOT called (request was handled early)
    assert (
        not mock_handle_request.called
    ), "Session manager should not be called for DELETE on non-existent session"

    # Verify a success response was sent
    assert send.called, "A response should have been sent"


@pytest.mark.asyncio
async def test_valid_mcp_session_id_is_preserved():
    """
    When the mcp-session-id header references a session that still exists,
    handle_streamable_http_mcp should NOT strip the header.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    valid_session_id = "valid-session-id-67890"

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [
            (b"content-type", b"application/json"),
            (b"mcp-session-id", valid_session_id.encode()),
            (b"authorization", b"Bearer test-key"),
        ],
    }

    receive = AsyncMock()
    send = AsyncMock()

    captured_scope = {}

    async def mock_handle_request(s, r, se):
        captured_scope.update(s)

    # Session manager HAS this session
    mock_instances = {valid_session_id: MagicMock()}

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(MagicMock(), None, None, None, None, None),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.set_auth_context",
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
            True,
        ),
        patch.object(
            session_manager,
            "handle_request",
            side_effect=mock_handle_request,
        ),
        patch.object(
            session_manager,
            "_server_instances",
            mock_instances,
        ),
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    # Verify the mcp-session-id header was preserved
    header_names = [k for k, v in captured_scope.get("headers", [])]
    assert (
        b"mcp-session-id" in header_names
    ), "Valid mcp-session-id header should have been preserved"


@pytest.mark.asyncio
async def test_no_mcp_session_id_header_works_normally():
    """
    When no mcp-session-id header is present (initial connection),
    handle_streamable_http_mcp should work without any issues.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer test-key"),
        ],
    }

    receive = AsyncMock()
    send = AsyncMock()

    captured_scope = {}

    async def mock_handle_request(s, r, se):
        captured_scope.update(s)

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(MagicMock(), None, None, None, None, None),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.set_auth_context",
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
            True,
        ),
        patch.object(
            session_manager,
            "handle_request",
            side_effect=mock_handle_request,
        ),
        patch.object(
            session_manager,
            "_server_instances",
            {},
        ),
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    # Verify headers are unchanged (no mcp-session-id was added or anything weird)
    header_names = [k for k, v in captured_scope.get("headers", [])]
    assert b"mcp-session-id" not in header_names
    assert b"content-type" in header_names


@pytest.mark.asyncio
async def test_per_user_oauth_missing_stored_token_returns_preemptive_401():
    """
    Per-user OAuth server with no stored token should fail fast with 401 +
    WWW-Authenticate so PKCE can start.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [
            (b"content-type", b"application/json"),
        ],
    }
    receive = AsyncMock()
    send = AsyncMock()
    user_auth = MagicMock()
    user_auth.user_id = "test-user-id"
    oauth_server = MagicMock()
    oauth_server.auth_type = MCPAuth.oauth2
    oauth_server.needs_user_oauth_token = True

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(user_auth, None, ["repro_oauth_server"], None, None, None),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.set_auth_context",
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
            True,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._handle_stale_mcp_session",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._get_user_oauth_extra_headers_from_db",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_get_stored_token,
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_name",
            return_value=oauth_server,
        ),
        patch.object(
            session_manager,
            "handle_request",
            new_callable=AsyncMock,
        ) as mock_handle_request,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await handle_streamable_http_mcp(scope, receive, send)

    exc = exc_info.value
    assert exc.status_code == 401
    assert "www-authenticate" in exc.headers
    assert mock_get_stored_token.await_count == 1
    assert mock_handle_request.await_count == 0


@pytest.mark.asyncio
async def test_per_user_oauth_with_stored_token_skips_preemptive_401():
    """
    Per-user OAuth server with an existing stored token should skip pre-emptive
    401 and continue to session manager request handling.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [
            (b"content-type", b"application/json"),
        ],
    }
    receive = AsyncMock()
    send = AsyncMock()
    user_auth = MagicMock()
    user_auth.user_id = "test-user-id"
    oauth_server = MagicMock()
    oauth_server.auth_type = MCPAuth.oauth2
    oauth_server.needs_user_oauth_token = True

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(user_auth, None, ["repro_oauth_server"], None, None, None),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.set_auth_context",
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
            True,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._handle_stale_mcp_session",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._get_user_oauth_extra_headers_from_db",
            new_callable=AsyncMock,
            return_value={"Authorization": "Bearer cached-token"},
        ) as mock_get_stored_token,
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_name",
            return_value=oauth_server,
        ),
        patch.object(
            session_manager,
            "handle_request",
            new_callable=AsyncMock,
        ) as mock_handle_request,
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    assert mock_get_stored_token.await_count == 1
    assert mock_handle_request.await_count == 1


@pytest.mark.asyncio
async def test_handle_streamable_http_mcp_emits_401_for_delegated_server_without_token():
    """
    OAuth2 server with ``delegate_auth_to_upstream=True`` and no Authorization
    header must still emit a pre-emptive 401 with WWW-Authenticate so the
    client kicks off PKCE. The 401 points at LiteLLM's discovery shim, which
    in turn delegates to the upstream OAuth issuer.
    """
    from fastapi import HTTPException

    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [
            (b"content-type", b"application/json"),
            (b"host", b"litellm.example.com"),
        ],
    }
    receive = AsyncMock()
    send = AsyncMock()
    user_auth = MagicMock()
    user_auth.user_id = None
    delegated_server = MagicMock()
    delegated_server.auth_type = MCPAuth.oauth2
    delegated_server.delegate_auth_to_upstream = True
    delegated_server.needs_user_oauth_token = True

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(
                user_auth,
                None,
                ["delegated_oauth_server"],
                None,
                None,
                None,
            ),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.set_auth_context",
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
            True,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._handle_stale_mcp_session",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_name",
            return_value=delegated_server,
        ),
        patch.object(
            session_manager,
            "handle_request",
            new_callable=AsyncMock,
        ) as mock_handle_request,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await handle_streamable_http_mcp(scope, receive, send)

    assert exc_info.value.status_code == 401
    assert "www-authenticate" in exc_info.value.headers
    assert mock_handle_request.await_count == 0
