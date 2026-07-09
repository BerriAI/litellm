"""
Tests for MCP stale session ID handling (Fixes #20292).

When clients reconnect to LiteLLM's MCP endpoint after a server restart or reload,
they may send a stale `mcp-session-id` header. This test verifies that:
1. For non-DELETE requests: stale session IDs are stripped so new sessions are created
2. For DELETE requests: idempotent behavior returns success even if session doesn't exist
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from litellm.types.mcp import MCPAuth
import pytest


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
                _stateful_session_active_request_counts,
                _stateful_session_auth_context_last_seen,
                _stateful_session_auth_contexts,
                _stateful_session_locks,
                _stateful_session_owners,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        stale_session_id = "stale-id"
        scope = {
            "type": "http",
            "method": "DELETE",
            "headers": [
                (b"content-type", b"application/json"),
                (b"mcp-session-id", stale_session_id.encode()),
            ],
        }
        receive = AsyncMock()
        send = AsyncMock()
        mgr = MagicMock()
        mgr._server_instances = {}  # no active sessions
        _stateful_session_auth_contexts[stale_session_id] = MagicMock()
        _stateful_session_auth_context_last_seen[stale_session_id] = 1.0
        _stateful_session_owners[stale_session_id] = "owner"
        _stateful_session_locks[stale_session_id] = MagicMock()
        _stateful_session_active_request_counts[stale_session_id] = 1

        try:
            handled = await _handle_stale_mcp_session(scope, receive, send, mgr)

            # Should be fully handled (returns True)
            assert handled is True
            # Should have sent a success response
            assert send.called
            # Header should NOT be stripped (DELETE needs the session ID)
            header_names = [k for k, _ in scope["headers"]]
            assert b"mcp-session-id" in header_names
            assert stale_session_id not in _stateful_session_auth_contexts
            assert stale_session_id not in _stateful_session_auth_context_last_seen
            assert stale_session_id not in _stateful_session_owners
            assert stale_session_id not in _stateful_session_locks
            assert stale_session_id not in _stateful_session_active_request_counts
        finally:
            _stateful_session_auth_contexts.pop(stale_session_id, None)
            _stateful_session_auth_context_last_seen.pop(stale_session_id, None)
            _stateful_session_owners.pop(stale_session_id, None)
            _stateful_session_locks.pop(stale_session_id, None)
            _stateful_session_active_request_counts.pop(stale_session_id, None)

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
            session_manager_stateful,
            session_manager_stateless,
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
    stateful_handle_request = AsyncMock()

    async def _stateless_capture(s, r, se):
        # Capture the scope that was actually passed
        captured_scope.update(s)

    stateless_handle_request = AsyncMock(side_effect=_stateless_capture)

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
            session_manager_stateless,
            "handle_request",
            new=stateless_handle_request,
        ),
        patch.object(
            session_manager_stateless,
            "_server_instances",
            {},
        ),
        patch.object(
            session_manager_stateful,
            "handle_request",
            side_effect=stateful_handle_request,
        ),
        patch.object(
            session_manager_stateful,
            "_server_instances",
            {},  # Empty dict = no active sessions
        ),
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    # Verify the mcp-session-id header was stripped
    header_names = [k for k, v in captured_scope.get("headers", [])]
    assert b"mcp-session-id" not in header_names, "Stale mcp-session-id header should have been stripped from the scope"
    assert stateless_handle_request.called, "Stale non-initialize requests should route stateless"
    assert not stateful_handle_request.called, "Stale non-initialize requests should not route stateful"


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
    assert not mock_handle_request.called, "Session manager should not be called for DELETE on non-existent session"

    # Verify a success response was sent
    assert send.called, "A response should have been sent"


@pytest.mark.asyncio
async def test_failed_delete_preserves_stateful_session_tracking():
    """
    When the SDK fails to terminate an existing stateful session, keep the
    owner/auth tracking so the session cannot be hijacked or hidden from cleanup.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _owner_fingerprint_for,
            _stateful_session_auth_context_last_seen,
            _stateful_session_auth_contexts,
            _stateful_session_locks,
            _stateful_session_owners,
            handle_streamable_http_mcp,
            session_manager_stateful,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    session_id = "delete-failure-session"
    user_auth = MagicMock()
    user_auth.api_key = "sk-test"
    user_auth.user_id = "test-user"
    auth_context = MagicMock()
    session_lock = asyncio.Lock()
    mock_instances = {session_id: MagicMock()}

    scope = {
        "type": "http",
        "method": "DELETE",
        "path": "/mcp",
        "headers": [
            (b"content-type", b"application/json"),
            (b"mcp-session-id", session_id.encode()),
            (b"authorization", b"Bearer sk-test"),
        ],
    }
    receive = AsyncMock()
    send = AsyncMock()

    _stateful_session_auth_contexts[session_id] = auth_context
    _stateful_session_auth_context_last_seen[session_id] = 1.0
    _stateful_session_owners[session_id] = _owner_fingerprint_for(user_auth)
    _stateful_session_locks[session_id] = session_lock

    try:
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
                new_callable=AsyncMock,
                return_value=(user_auth, None, None, None, None, None),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
                True,
            ),
            patch.object(
                session_manager_stateful,
                "handle_request",
                new_callable=AsyncMock,
                side_effect=RuntimeError("delete failed"),
            ) as mock_handle_request,
            patch.object(
                session_manager_stateful,
                "_server_instances",
                mock_instances,
            ),
        ):
            await handle_streamable_http_mcp(scope, receive, send)

        assert mock_handle_request.await_count == 1
        assert _stateful_session_auth_contexts[session_id] is auth_context
        assert _stateful_session_auth_context_last_seen[session_id] == 1.0
        assert _stateful_session_owners[session_id] == _owner_fingerprint_for(user_auth)
        assert _stateful_session_locks[session_id] is session_lock
        assert session_id in mock_instances
    finally:
        _stateful_session_auth_contexts.pop(session_id, None)
        _stateful_session_auth_context_last_seen.pop(session_id, None)
        _stateful_session_owners.pop(session_id, None)
        _stateful_session_locks.pop(session_id, None)


@pytest.mark.asyncio
async def test_valid_mcp_session_id_is_preserved():
    """
    When the mcp-session-id header references a session that still exists,
    handle_streamable_http_mcp should NOT strip the header.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateful,
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

    # Stateful session manager HAS this session (requests with mcp-session-id route there)
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
            session_manager_stateful,
            "handle_request",
            side_effect=mock_handle_request,
        ),
        patch.object(
            session_manager_stateful,
            "_server_instances",
            mock_instances,
        ),
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    # Verify the mcp-session-id header was preserved
    header_names = [k for k, v in captured_scope.get("headers", [])]
    assert b"mcp-session-id" in header_names, "Valid mcp-session-id header should have been preserved"


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
    from fastapi import HTTPException

    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateless,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "scheme": "http",
        "query_string": b"",
        "root_path": "",
        "server": ("localhost", 8000),
        "headers": [
            (b"content-type", b"application/json"),
            (b"host", b"localhost:8000"),
        ],
    }
    receive = AsyncMock()
    send = AsyncMock()
    user_auth = MagicMock()
    user_auth.user_id = "test-user-id"
    oauth_server = MagicMock()
    oauth_server.auth_type = MCPAuth.oauth2
    oauth_server.needs_user_oauth_token = True
    oauth_server.delegate_auth_to_upstream = False

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
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.has_user_oauth_token",
            new_callable=AsyncMock,
            return_value=False,
        ) as mock_has_token,
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_name",
            return_value=oauth_server,
        ),
        patch.object(
            session_manager_stateless,
            "handle_request",
            new_callable=AsyncMock,
        ) as mock_handle_request,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await handle_streamable_http_mcp(scope, receive, send)

    # Verify a 401 was raised
    assert mock_has_token.await_count == 1
    assert mock_handle_request.await_count == 0
    assert exc_info.value.status_code == 401
    assert "www-authenticate" in exc_info.value.headers
    assert "Bearer authorization_uri=" in exc_info.value.headers["www-authenticate"]


@pytest.mark.asyncio
async def test_handle_streamable_http_mcp_delegated_server_surfaces_upstream_challenge():
    """
    OAuth2 server with ``delegate_auth_to_upstream=True`` where the client
    already presents a bearer token the upstream rejects: the request bypasses
    the preemptive challenge (a token is present) and reaches the session
    manager, whose upstream ``MCPUpstreamAuthError`` is surfaced verbatim so the
    client sees the upstream's RFC 9728 challenge rather than the gateway's.
    """
    from fastapi import HTTPException

    try:
        from litellm.proxy._experimental.mcp_server.exceptions import (
            MCPUpstreamAuthError,
        )
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateful,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp/delegated_oauth_server",
        "scheme": "https",
        "query_string": b"",
        "root_path": "",
        "server": ("litellm.example.com", 443),
        "headers": [
            (b"content-type", b"application/json"),
            (b"host", b"litellm.example.com"),
        ],
    }
    receive = AsyncMock(
        return_value={
            "type": "http.request",
            "body": b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}',
            "more_body": False,
        }
    )
    send = AsyncMock()
    user_auth = MagicMock()
    user_auth.user_id = None
    delegated_server = MagicMock()
    delegated_server.auth_type = MCPAuth.oauth2
    delegated_server.delegate_auth_to_upstream = True
    delegated_server.needs_user_oauth_token = True
    delegated_server.is_oauth_passthrough = False
    delegated_server.is_oauth_delegate = False
    delegated_server.is_true_passthrough = False
    delegated_server.server_id = "delegated-oauth-server"

    upstream_challenge = 'Bearer resource_metadata="https://upstream.example.com/.well-known/oauth-protected-resource"'

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(
                user_auth,
                None,
                ["delegated_oauth_server"],
                None,
                {"Authorization": "Bearer upstream-token-the-upstream-rejects"},
                None,
            ),
        ),
        patch("litellm.proxy._experimental.mcp_server.server.set_auth_context"),
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
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_name",
            return_value=delegated_server,
        ),
        patch.object(
            session_manager_stateful,
            "handle_request",
            new_callable=AsyncMock,
            side_effect=MCPUpstreamAuthError(
                status_code=401,
                www_authenticate=upstream_challenge,
                server_name="delegated_oauth_server",
            ),
        ) as mock_handle_request,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await handle_streamable_http_mcp(scope, receive, send)

    assert mock_handle_request.await_count == 1
    assert exc_info.value.status_code == 401
    assert exc_info.value.headers == {"www-authenticate": upstream_challenge}


@pytest.mark.asyncio
async def test_per_user_oauth_with_stored_token_skips_preemptive_401():
    """
    Per-user OAuth server with an existing stored token should skip pre-emptive
    401 and continue to session manager request handling.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateless,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "scheme": "http",
        "query_string": b"",
        "root_path": "",
        "server": ("localhost", 8000),
        "headers": [
            (b"content-type", b"application/json"),
            (b"host", b"localhost:8000"),
        ],
    }
    receive = AsyncMock(
        return_value={
            "type": "http.request",
            "body": b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}',
            "more_body": False,
        }
    )
    send = AsyncMock()
    user_auth = MagicMock()
    user_auth.user_id = "test-user-id"
    oauth_server = MagicMock()
    oauth_server.auth_type = MCPAuth.oauth2
    oauth_server.needs_user_oauth_token = True
    oauth_server.delegate_auth_to_upstream = False

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
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.has_user_oauth_token",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_has_token,
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_name",
            return_value=oauth_server,
        ),
        patch.object(
            session_manager_stateless,
            "handle_request",
            new_callable=AsyncMock,
        ) as mock_handle_request,
        patch.object(
            session_manager_stateless,
            "_server_instances",
            {},
        ),
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    assert mock_has_token.await_count == 1
    assert mock_handle_request.await_count == 1


@pytest.mark.asyncio
async def test_handle_streamable_http_mcp_delegated_server_without_token_returns_preemptive_resource_metadata_401():
    """
    OAuth2 server with ``delegate_auth_to_upstream=True`` and no stored token
    must fail fast with a 401 carrying a ``resource_metadata=`` challenge that
    points at the gateway's proxied oauth-protected-resource well-known, so the
    MCP client starts PKCE against the upstream IdP. On ``initialize`` the
    gateway answers locally and never probes upstream, so this preemptive
    challenge is the only thing that can drive the client into the OAuth flow;
    falling through to the session manager (or emitting the gateway
    ``authorization_uri=`` challenge) leaves the client with no tools and no
    sign-in prompt.
    """
    from fastapi import HTTPException

    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateful,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp/delegated_oauth_server",
        "_original_path": "/delegated_oauth_server/mcp",
        "scheme": "https",
        "query_string": b"",
        "root_path": "",
        "server": ("litellm.example.com", 443),
        "headers": [
            (b"content-type", b"application/json"),
            (b"host", b"litellm.example.com"),
        ],
    }
    receive = AsyncMock(
        return_value={
            "type": "http.request",
            "body": b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}',
            "more_body": False,
        }
    )
    send = AsyncMock()
    user_auth = MagicMock()
    user_auth.user_id = None
    delegated_server = MagicMock()
    delegated_server.auth_type = MCPAuth.oauth2
    delegated_server.delegate_auth_to_upstream = True
    delegated_server.needs_user_oauth_token = True
    delegated_server.server_id = "delegated-oauth-server"

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
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.has_user_oauth_token",
            new_callable=AsyncMock,
        ) as mock_has_token,
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_name",
            return_value=delegated_server,
        ),
        patch.object(
            session_manager_stateful,
            "handle_request",
            new_callable=AsyncMock,
        ) as mock_handle_request,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await handle_streamable_http_mcp(scope, receive, send)

    # Delegate-auth servers raise the resource_metadata challenge before any
    # per-user existence check, so the v2 token store is never consulted.
    assert mock_has_token.await_count == 0
    assert mock_handle_request.await_count == 0
    assert exc_info.value.status_code == 401
    challenge = exc_info.value.headers["www-authenticate"]
    assert "resource_metadata=" in challenge
    assert "authorization_uri=" not in challenge
    assert "/.well-known/oauth-protected-resource/delegated_oauth_server/mcp" in challenge


@pytest.mark.asyncio
async def test_handle_streamable_http_mcp_token_exchange_without_subject_returns_preemptive_resource_metadata_401():
    """An ``oauth2_token_exchange`` (OBO) server with no caller subject token must fail fast at
    connect with a 401 carrying the RFC 9728 ``resource_metadata`` + RFC 6750 ``invalid_token``
    challenge, so the client discovers the IdP and retries with a subject token. A tool-call-time
    401 would be wrapped into a JSON-RPC error and the WWW-Authenticate lost, so this preemptive
    challenge is what drives the discovery flow."""
    from fastapi import HTTPException

    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateful,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp/obo_server",
        "_original_path": "/mcp/obo_server",
        "scheme": "https",
        "query_string": b"",
        "root_path": "",
        "server": ("litellm.example.com", 443),
        "headers": [
            (b"content-type", b"application/json"),
            (b"host", b"litellm.example.com"),
        ],
    }
    receive = AsyncMock(
        return_value={
            "type": "http.request",
            "body": b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}',
            "more_body": False,
        }
    )
    send = AsyncMock()
    user_auth = MagicMock()
    user_auth.user_id = None
    obo_server = MagicMock()
    obo_server.auth_type = MCPAuth.oauth2_token_exchange
    obo_server.alias = None
    obo_server.server_name = "obo_server"
    obo_server.name = "obo_server"
    obo_server.server_id = "obo-server"

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(user_auth, None, ["obo_server"], None, None, None),
        ),
        patch("litellm.proxy._experimental.mcp_server.server.set_auth_context"),
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
            return_value=obo_server,
        ),
        patch.object(
            session_manager_stateful,
            "handle_request",
            new_callable=AsyncMock,
        ) as mock_handle_request,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await handle_streamable_http_mcp(scope, receive, send)

    assert mock_handle_request.await_count == 0
    assert exc_info.value.status_code == 401
    headers = {k.lower(): v for k, v in (exc_info.value.headers or {}).items()}
    challenge = headers["www-authenticate"]
    # Structural invariants only: the exact root-path prefix is exercised in the adapter's
    # oauth_protected_resource_path unit test, so this handler test stays hermetic w.r.t.
    # SERVER_ROOT_PATH (which other tests in the shard may have left set in the environment).
    assert "resource_metadata=" in challenge
    assert "/.well-known/oauth-protected-resource" in challenge
    assert challenge.split('resource_metadata="', 1)[1].split('"', 1)[0].endswith("/mcp/obo_server")
    assert 'error="invalid_token"' in challenge


def _passthrough_mode_scope(server_name: str, extra_headers=None):
    headers = [
        (b"content-type", b"application/json"),
        (b"host", b"litellm.example.com"),
    ] + list(extra_headers or [])
    return {
        "type": "http",
        "method": "POST",
        "path": f"/mcp/{server_name}",
        "_original_path": f"/{server_name}/mcp",
        "scheme": "https",
        "query_string": b"",
        "root_path": "",
        "server": ("litellm.example.com", 443),
        "headers": headers,
    }


def _build_passthrough_mode_server(server_name: str, auth_type):
    from litellm.proxy._types import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    return MCPServer(
        server_id=f"{server_name}-id",
        name=server_name,
        server_name=server_name,
        alias=server_name,
        url="https://upstream.example.com/mcp",
        transport=MCPTransport.http,
        auth_type=auth_type,
    )


@pytest.mark.asyncio
async def test_handle_streamable_http_mcp_oauth_delegate_without_token_returns_gateway_proxied_401():
    """oauth_delegate is admitted with the LiteLLM key but still owns upstream
    OAuth. With no forwarded upstream token the gateway must challenge with the
    proxied resource_metadata (which advertises the upstream IdP), never the
    gateway authorization_uri and never a silent 200."""
    from fastapi import HTTPException

    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateful,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    scope = _passthrough_mode_scope("od_server")
    receive = AsyncMock(
        return_value={
            "type": "http.request",
            "body": b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}',
            "more_body": False,
        }
    )
    send = AsyncMock()
    user_auth = MagicMock()
    user_auth.user_id = "u1"
    od_server = _build_passthrough_mode_server("od_server", MCPAuth.oauth_delegate)

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(user_auth, None, ["od_server"], None, None, None),
        ),
        patch("litellm.proxy._experimental.mcp_server.server.set_auth_context"),
        patch(
            "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
            True,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_name",
            return_value=od_server,
        ),
        patch.object(
            session_manager_stateful,
            "handle_request",
            new_callable=AsyncMock,
        ) as mock_handle_request,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await handle_streamable_http_mcp(scope, receive, send)

    assert mock_handle_request.await_count == 0
    assert exc_info.value.status_code == 401
    challenge = exc_info.value.headers["www-authenticate"]
    assert "resource_metadata=" in challenge
    assert "authorization_uri=" not in challenge
    assert "/.well-known/oauth-protected-resource/od_server/mcp" in challenge


@pytest.mark.asyncio
async def test_handle_streamable_http_mcp_oauth_delegate_with_forwarded_token_skips_challenge():
    """When the oauth_delegate caller carries both the LiteLLM key and a separate
    upstream Authorization, the gateway must forward to the session manager, not
    re-challenge. Guards the ``_get_forwarded_auth_from_scope(...) is None``
    condition: dropping it would 401 even a fully-authenticated request."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateless,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    scope = _passthrough_mode_scope(
        "od_server",
        extra_headers=[
            (b"x-litellm-api-key", b"Bearer sk-1234"),
            (b"authorization", b"Bearer upstream-token"),
        ],
    )
    receive = AsyncMock(
        return_value={
            "type": "http.request",
            "body": b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}',
            "more_body": False,
        }
    )
    send = AsyncMock()
    user_auth = MagicMock()
    user_auth.user_id = "u1"
    od_server = _build_passthrough_mode_server("od_server", MCPAuth.oauth_delegate)

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(user_auth, None, ["od_server"], None, None, None),
        ),
        patch("litellm.proxy._experimental.mcp_server.server.set_auth_context"),
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
            "litellm.proxy._experimental.mcp_server.server._check_passthrough_upstream_auth",
            new_callable=AsyncMock,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_name",
            return_value=od_server,
        ),
        patch.object(
            session_manager_stateless,
            "handle_request",
            new_callable=AsyncMock,
        ) as mock_handle_request,
        patch.object(session_manager_stateless, "_server_instances", {}),
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    assert mock_handle_request.await_count == 1


async def _run_passthrough_connect(
    *,
    auth_type,
    server_names,
    mcp_server_auth_headers,
    scope_extra_headers=None,
):
    """Drive handle_streamable_http_mcp through the preemptive-401 gate and report whether it
    challenged (raised) or forwarded to the session manager. Returns (challenged, www_authenticate)."""
    from litellm.proxy._experimental.mcp_server.server import (
        handle_streamable_http_mcp,
        session_manager_stateless,
    )

    scope = _passthrough_mode_scope(server_names[0], extra_headers=scope_extra_headers)
    receive = AsyncMock(
        return_value={
            "type": "http.request",
            "body": b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}',
            "more_body": False,
        }
    )
    send = AsyncMock()
    user_auth = MagicMock()
    user_auth.user_id = "u1"
    server = _build_passthrough_mode_server(server_names[0], auth_type)

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(user_auth, None, server_names, mcp_server_auth_headers, None, None),
        ),
        patch("litellm.proxy._experimental.mcp_server.server.set_auth_context"),
        patch("litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED", True),
        patch(
            "litellm.proxy._experimental.mcp_server.server._handle_stale_mcp_session",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._check_passthrough_upstream_auth",
            new_callable=AsyncMock,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_name",
            return_value=server,
        ),
        patch.object(session_manager_stateless, "handle_request", new_callable=AsyncMock) as mock_handle_request,
        patch.object(session_manager_stateless, "_server_instances", {}),
    ):
        try:
            await handle_streamable_http_mcp(scope, receive, send)
        except HTTPException as exc:
            return True, (exc.headers or {}).get("www-authenticate")
    return mock_handle_request.await_count == 0, None


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_type", [MCPAuth.oauth_delegate, MCPAuth.true_passthrough])
async def test_handle_streamable_http_mcp_per_server_header_skips_preemptive_challenge(auth_type):
    """A per-server x-mcp-{alias}-authorization header binds the upstream token to one server; the
    connect gate must recognize it and forward instead of spuriously 401-ing, since egress already
    honors it. Without this, the mandatory multi-server binding is unusable at connect."""
    try:
        from litellm.proxy._experimental.mcp_server.server import handle_streamable_http_mcp  # noqa: F401
    except ImportError:
        pytest.skip("MCP server not available")

    challenged, _ = await _run_passthrough_connect(
        auth_type=auth_type,
        server_names=["pt_server"],
        mcp_server_auth_headers={"pt_server": {"Authorization": "Bearer upstream-token"}},
    )
    assert challenged is False


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_type", [MCPAuth.oauth_delegate, MCPAuth.true_passthrough])
async def test_handle_streamable_http_mcp_sanitized_per_server_header_skips_preemptive_challenge(auth_type):
    """A dashboard client sends x-mcp-{sanitize_mcp_alias_for_header(alias)}-authorization, so the
    alias 'pt-server' arrives as the header key 'pt_server'. Egress resolves that via the sanitized
    alias, so the connect gate must too, or it 401s a token egress would forward."""
    try:
        from litellm.proxy._experimental.mcp_server.server import handle_streamable_http_mcp  # noqa: F401
    except ImportError:
        pytest.skip("MCP server not available")

    challenged, _ = await _run_passthrough_connect(
        auth_type=auth_type,
        server_names=["pt-server"],
        mcp_server_auth_headers={"pt_server": {"Authorization": "Bearer upstream-token"}},
    )
    assert challenged is False


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_type", [MCPAuth.oauth_delegate, MCPAuth.true_passthrough])
async def test_handle_streamable_http_mcp_aggregate_does_not_preemptively_challenge(auth_type):
    """A multi-server aggregate must degrade gracefully: the preemptive 401 is single-server only, so
    one server missing a token cannot 401 the whole connect (the listing absorbs per-server failures)."""
    try:
        from litellm.proxy._experimental.mcp_server.server import handle_streamable_http_mcp  # noqa: F401
    except ImportError:
        pytest.skip("MCP server not available")

    challenged, _ = await _run_passthrough_connect(
        auth_type=auth_type,
        server_names=["pt_server", "pt_server_2"],
        mcp_server_auth_headers=None,
    )
    assert challenged is False


@pytest.mark.asyncio
async def test_handle_streamable_http_mcp_true_passthrough_without_token_surfaces_verbatim_upstream_challenge():
    """true_passthrough is a transparent proxy: with no client Authorization the
    gateway probes the upstream and surfaces its own WWW-Authenticate verbatim,
    so the client discovers and authorizes against the upstream directly. Guards
    against answering initialize locally with a silent 200."""
    from fastapi import HTTPException

    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateful,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    upstream_challenge = 'Bearer resource_metadata="https://upstream.example.com/.well-known/oauth-protected-resource"'
    probe_response = MagicMock()
    probe_response.status_code = 401
    probe_response.headers = {"www-authenticate": upstream_challenge}
    probe_client = MagicMock()
    probe_client.post = AsyncMock(return_value=probe_response)

    scope = _passthrough_mode_scope("tp_server")
    receive = AsyncMock(
        return_value={
            "type": "http.request",
            "body": b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}',
            "more_body": False,
        }
    )
    send = AsyncMock()
    user_auth = MagicMock()
    user_auth.user_id = None
    tp_server = _build_passthrough_mode_server("tp_server", MCPAuth.true_passthrough)

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(user_auth, None, ["tp_server"], None, None, None),
        ),
        patch("litellm.proxy._experimental.mcp_server.server.set_auth_context"),
        patch(
            "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
            True,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.get_async_httpx_client",
            return_value=probe_client,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_name",
            return_value=tp_server,
        ),
        patch.object(
            session_manager_stateful,
            "handle_request",
            new_callable=AsyncMock,
        ) as mock_handle_request,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await handle_streamable_http_mcp(scope, receive, send)

    assert mock_handle_request.await_count == 0
    assert exc_info.value.status_code == 401
    assert exc_info.value.headers["www-authenticate"] == upstream_challenge
    probe_client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_streamable_http_mcp_true_passthrough_with_token_skips_probe_and_challenge():
    """When the true_passthrough caller already carries an Authorization the
    gateway must forward without probing or challenging. Guards the
    ``not _scope_has_authorization_header(scope)`` condition and the no-probe
    fast path."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
            session_manager_stateless,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    probe_client = MagicMock()
    probe_client.post = AsyncMock()

    scope = _passthrough_mode_scope(
        "tp_server",
        extra_headers=[(b"authorization", b"Bearer upstream-token")],
    )
    receive = AsyncMock(
        return_value={
            "type": "http.request",
            "body": b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}',
            "more_body": False,
        }
    )
    send = AsyncMock()
    user_auth = MagicMock()
    user_auth.user_id = None
    tp_server = _build_passthrough_mode_server("tp_server", MCPAuth.true_passthrough)

    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(user_auth, None, ["tp_server"], None, None, None),
        ),
        patch("litellm.proxy._experimental.mcp_server.server.set_auth_context"),
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
            "litellm.proxy._experimental.mcp_server.server._check_passthrough_upstream_auth",
            new_callable=AsyncMock,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.get_async_httpx_client",
            return_value=probe_client,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_name",
            return_value=tp_server,
        ),
        patch.object(
            session_manager_stateless,
            "handle_request",
            new_callable=AsyncMock,
        ) as mock_handle_request,
        patch.object(session_manager_stateless, "_server_instances", {}),
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    assert mock_handle_request.await_count == 1
    probe_client.post.assert_not_awaited()
