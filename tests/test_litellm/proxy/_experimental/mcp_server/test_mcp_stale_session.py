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

    with patch(
        "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
        new_callable=AsyncMock,
        return_value=(user_auth, None, ["repro_oauth_server"], None, None, None),
    ), patch(
        "litellm.proxy._experimental.mcp_server.server.set_auth_context",
    ), patch(
        "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
        True,
    ), patch(
        "litellm.proxy._experimental.mcp_server.server._handle_stale_mcp_session",
        new_callable=AsyncMock,
        return_value=False,
    ), patch(
        "litellm.proxy._experimental.mcp_server.server._get_user_oauth_extra_headers_from_db",
        new_callable=AsyncMock,
        return_value=None,
    ) as mock_get_stored_token, patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_name",
        return_value=oauth_server,
    ), patch.object(
        session_manager,
        "handle_request",
        new_callable=AsyncMock,
    ) as mock_handle_request:
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

    with patch(
        "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
        new_callable=AsyncMock,
        return_value=(user_auth, None, ["repro_oauth_server"], None, None, None),
    ), patch(
        "litellm.proxy._experimental.mcp_server.server.set_auth_context",
    ), patch(
        "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
        True,
    ), patch(
        "litellm.proxy._experimental.mcp_server.server._handle_stale_mcp_session",
        new_callable=AsyncMock,
        return_value=False,
    ), patch(
        "litellm.proxy._experimental.mcp_server.server._get_user_oauth_extra_headers_from_db",
        new_callable=AsyncMock,
        return_value={"Authorization": "Bearer cached-token"},
    ) as mock_get_stored_token, patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_name",
        return_value=oauth_server,
    ), patch.object(
        session_manager,
        "handle_request",
        new_callable=AsyncMock,
    ) as mock_handle_request:
        await handle_streamable_http_mcp(scope, receive, send)

    assert mock_get_stored_token.await_count == 1
    assert mock_handle_request.await_count == 1


@pytest.mark.asyncio
async def test_oauth_bootstrap_returns_401_without_mocking_extract_mcp_auth_context():
    """
    Regression: an unauthenticated POST to /mcp/{server} where the server is
    OAuth2-configured must return 401 + WWW-Authenticate, not 500.

    This test deliberately does NOT mock extract_mcp_auth_context. It registers
    a real OAuth2 server in the manager and lets the request flow through the
    pre-check helper. Without the pre-check, the empty Authorization header
    would reach strict API-key validation and raise a ProxyException that the
    catch-all coerces to 500.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP server not available")

    global_mcp_server_manager.registry.clear()
    public_server = MCPServer(
        server_id="bootstrap_test_server",
        name="bootstrap_test_server",
        server_name="bootstrap_test_server",
        alias="bootstrap_test_server",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="real-client-id",
        client_secret=None,
        authorization_url="https://idp.example/authorize",
        token_url="https://idp.example/token",
    )
    global_mcp_server_manager.registry[public_server.server_id] = public_server

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp/bootstrap_test_server",
        "headers": [
            (b"content-type", b"application/json"),
        ],
    }
    receive = AsyncMock()
    send = AsyncMock()

    try:
        with patch(
            "litellm.proxy._experimental.mcp_server.server.IPAddressUtils.get_mcp_client_ip",
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_streamable_http_mcp(scope, receive, send)
    finally:
        global_mcp_server_manager.registry.clear()

    exc = exc_info.value
    assert exc.status_code == 401
    assert "www-authenticate" in exc.headers
    assert (
        "/.well-known/oauth-authorization-server/bootstrap_test_server"
        in exc.headers["www-authenticate"]
    )


@pytest.mark.asyncio
async def test_oauth_bootstrap_skips_when_authorization_header_present():
    """
    Pre-check must NOT fire when the client sends any Authorization header —
    let the existing auth flow + 401 logic decide.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP server not available")

    global_mcp_server_manager.registry.clear()
    public_server = MCPServer(
        server_id="bootstrap_test_server",
        name="bootstrap_test_server",
        server_name="bootstrap_test_server",
        alias="bootstrap_test_server",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="real-client-id",
        client_secret=None,
        authorization_url="https://idp.example/authorize",
        token_url="https://idp.example/token",
    )
    global_mcp_server_manager.registry[public_server.server_id] = public_server

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp/bootstrap_test_server",
        "headers": [
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer sk-some-litellm-key"),
        ],
    }
    receive = AsyncMock()
    send = AsyncMock()

    sentinel_extract = AsyncMock(
        side_effect=RuntimeError("downstream_reached_as_expected")
    )

    try:
        with patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            sentinel_extract,
        ), patch(
            "litellm.proxy._experimental.mcp_server.server.IPAddressUtils.get_mcp_client_ip",
            return_value=None,
        ):
            await handle_streamable_http_mcp(scope, receive, send)
    finally:
        global_mcp_server_manager.registry.clear()

    assert sentinel_extract.await_count == 1


@pytest.mark.asyncio
async def test_oauth_bootstrap_skips_when_path_does_not_resolve_to_named_server():
    """
    Pre-check no-op when path is /mcp (root) without a server name — flows
    into the existing extract_mcp_auth_context path.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
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

    sentinel_extract = AsyncMock(
        side_effect=RuntimeError("downstream_reached_as_expected")
    )

    with patch(
        "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
        sentinel_extract,
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    assert sentinel_extract.await_count == 1


@pytest.mark.asyncio
async def test_oauth_bootstrap_skips_for_non_oauth_server():
    """
    Pre-check no-op for path-resolved server whose auth_type != oauth2.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            handle_streamable_http_mcp,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP server not available")

    global_mcp_server_manager.registry.clear()
    api_key_server = MCPServer(
        server_id="api_key_server",
        name="api_key_server",
        server_name="api_key_server",
        alias="api_key_server",
        transport=MCPTransport.http,
        auth_type=MCPAuth.api_key,
    )
    global_mcp_server_manager.registry[api_key_server.server_id] = api_key_server

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp/api_key_server",
        "headers": [
            (b"content-type", b"application/json"),
        ],
    }
    receive = AsyncMock()
    send = AsyncMock()

    sentinel_extract = AsyncMock(
        side_effect=RuntimeError("downstream_reached_as_expected")
    )

    try:
        with patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            sentinel_extract,
        ), patch(
            "litellm.proxy._experimental.mcp_server.server.IPAddressUtils.get_mcp_client_ip",
            return_value=None,
        ):
            await handle_streamable_http_mcp(scope, receive, send)
    finally:
        global_mcp_server_manager.registry.clear()

    assert sentinel_extract.await_count == 1


@pytest.mark.asyncio
async def test_oauth_bootstrap_returns_401_via_sse_handler():
    """
    Regression: the SSE handler must propagate the 401 raised by the pre-check
    helper instead of swallowing it via its bare `except Exception`. Mirror
    of the StreamableHTTP test against handle_sse_mcp.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import handle_sse_mcp
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP server not available")

    global_mcp_server_manager.registry.clear()
    public_server = MCPServer(
        server_id="bootstrap_sse_server",
        name="bootstrap_sse_server",
        server_name="bootstrap_sse_server",
        alias="bootstrap_sse_server",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        client_id="real-client-id",
        client_secret=None,
        authorization_url="https://idp.example/authorize",
        token_url="https://idp.example/token",
    )
    global_mcp_server_manager.registry[public_server.server_id] = public_server

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp/bootstrap_sse_server",
        "headers": [
            (b"content-type", b"application/json"),
        ],
    }
    receive = AsyncMock()
    send = AsyncMock()

    try:
        with patch(
            "litellm.proxy._experimental.mcp_server.server.IPAddressUtils.get_mcp_client_ip",
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handle_sse_mcp(scope, receive, send)
    finally:
        global_mcp_server_manager.registry.clear()

    exc = exc_info.value
    assert exc.status_code == 401
    assert "www-authenticate" in exc.headers
    assert (
        "/.well-known/oauth-authorization-server/bootstrap_sse_server"
        in exc.headers["www-authenticate"]
    )
