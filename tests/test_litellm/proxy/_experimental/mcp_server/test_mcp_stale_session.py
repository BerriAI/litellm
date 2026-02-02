"""
Tests for MCP stale session ID handling (Fixes #20292).

When VSCode reconnects to LiteLLM's MCP endpoint after a reload, it sends a stale
`mcp-session-id` header. The session manager returns a 404 because the old session
was cleaned up. This test verifies that stale session IDs are detected and stripped
so a new session is created automatically.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestStripStaleMcpSessionHeader:
    """Unit tests for the _strip_stale_mcp_session_header helper."""

    def test_strips_stale_session_id(self):
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                _strip_stale_mcp_session_header,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        scope = {
            "headers": [
                (b"content-type", b"application/json"),
                (b"mcp-session-id", b"stale-id"),
            ],
        }
        mgr = MagicMock()
        mgr._server_instances = {}  # no active sessions

        _strip_stale_mcp_session_header(scope, mgr)

        header_names = [k for k, _ in scope["headers"]]
        assert b"mcp-session-id" not in header_names

    def test_preserves_valid_session_id(self):
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                _strip_stale_mcp_session_header,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        scope = {
            "headers": [
                (b"content-type", b"application/json"),
                (b"mcp-session-id", b"valid-id"),
            ],
        }
        mgr = MagicMock()
        mgr._server_instances = {"valid-id": MagicMock()}

        _strip_stale_mcp_session_header(scope, mgr)

        header_names = [k for k, _ in scope["headers"]]
        assert b"mcp-session-id" in header_names

    def test_no_op_when_no_session_header(self):
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                _strip_stale_mcp_session_header,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        scope = {
            "headers": [
                (b"content-type", b"application/json"),
            ],
        }
        mgr = MagicMock()
        mgr._server_instances = {}

        _strip_stale_mcp_session_header(scope, mgr)

        assert len(scope["headers"]) == 1

    def test_no_op_when_server_instances_missing(self):
        """If _server_instances attr doesn't exist, don't crash."""
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                _strip_stale_mcp_session_header,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        scope = {
            "headers": [
                (b"mcp-session-id", b"some-id"),
            ],
        }
        mgr = MagicMock(spec=[])  # no attributes

        _strip_stale_mcp_session_header(scope, mgr)

        # Should keep the header since we can't verify
        header_names = [k for k, _ in scope["headers"]]
        assert b"mcp-session-id" in header_names


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

    with patch(
        "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
        new_callable=AsyncMock,
        return_value=(MagicMock(), None, None, None, None, None),
    ), patch(
        "litellm.proxy._experimental.mcp_server.server.set_auth_context",
    ), patch(
        "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
        True,
    ), patch.object(
        session_manager,
        "handle_request",
        side_effect=mock_handle_request,
    ), patch.object(
        session_manager,
        "_server_instances",
        {},  # Empty dict = no active sessions
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    # Verify the mcp-session-id header was stripped
    header_names = [k for k, v in captured_scope.get("headers", [])]
    assert b"mcp-session-id" not in header_names, (
        "Stale mcp-session-id header should have been stripped from the scope"
    )


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

    with patch(
        "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
        new_callable=AsyncMock,
        return_value=(MagicMock(), None, None, None, None, None),
    ), patch(
        "litellm.proxy._experimental.mcp_server.server.set_auth_context",
    ), patch(
        "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
        True,
    ), patch.object(
        session_manager,
        "handle_request",
        side_effect=mock_handle_request,
    ), patch.object(
        session_manager,
        "_server_instances",
        mock_instances,
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    # Verify the mcp-session-id header was preserved
    header_names = [k for k, v in captured_scope.get("headers", [])]
    assert b"mcp-session-id" in header_names, (
        "Valid mcp-session-id header should have been preserved"
    )


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

    with patch(
        "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
        new_callable=AsyncMock,
        return_value=(MagicMock(), None, None, None, None, None),
    ), patch(
        "litellm.proxy._experimental.mcp_server.server.set_auth_context",
    ), patch(
        "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
        True,
    ), patch.object(
        session_manager,
        "handle_request",
        side_effect=mock_handle_request,
    ), patch.object(
        session_manager,
        "_server_instances",
        {},
    ):
        await handle_streamable_http_mcp(scope, receive, send)

    # Verify headers are unchanged (no mcp-session-id was added or anything weird)
    header_names = [k for k, v in captured_scope.get("headers", [])]
    assert b"mcp-session-id" not in header_names
    assert b"content-type" in header_names
