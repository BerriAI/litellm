import pytest
import uuid
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from contextlib import asynccontextmanager
from litellm.proxy._experimental.mcp_server.server import (
    handle_sse_mcp_endpoint,
    handle_sse_post_messages,
    _captured_session_id_var,
    _session_id_auth_storage,
)
from litellm.proxy._types import UserAPIKeyAuth


@pytest.mark.asyncio
async def test_session_id_capture_and_binding():
    """
    Test that session_id is correctly captured during SSE connection
    and that subsequent POST messages are correctly bound to that session.
    """
    session_id = uuid.uuid4()

    # Mock SSE transport
    mock_sse = MagicMock()
    mock_sse.handle_post_message = AsyncMock()

    # Mock connect_sse to simulate the SDK behavior
    @asynccontextmanager
    async def mock_connect_sse(scope, receive, send):
        # Directly set the ContextVar to simulate our capturing dict's behavior
        _captured_session_id_var.set(session_id)
        yield (AsyncMock(), AsyncMock())

    mock_sse.connect_sse = MagicMock(side_effect=mock_connect_sse)
    mock_sse._read_stream_writers = {session_id: MagicMock()}

    mock_auth_context = (
        UserAPIKeyAuth(api_key="session-key", user_id="user1"),
        "auth-header",
        ["server1"],
        {},
        {},
        {},
    )

    mock_scope = {"type": "http", "method": "GET", "path": "/mcp/sse", "headers": []}
    mock_receive = AsyncMock()
    mock_send = AsyncMock()

    # We'll use an event to coordinate with the mocked server.run
    run_started = asyncio.Event()
    finish_run = asyncio.Event()

    async def mock_run(*args, **kwargs):
        run_started.set()
        await finish_run.wait()

    # 1. Test Session ID Capture and POST binding during active session
    with (
        patch("litellm.proxy._experimental.mcp_server.server.sse", mock_sse),
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            AsyncMock(return_value=mock_auth_context),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._gateway_initialize_instructions_request_scope",
            MagicMock(
                return_value=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock())
            ),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.server",
            MagicMock(run=mock_run, create_initialization_options=MagicMock()),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
            True,
        ),
    ):
        # Start the endpoint in the background
        task = asyncio.create_task(
            handle_sse_mcp_endpoint(mock_scope, mock_receive, mock_send)
        )

        # Wait for it to start and set the auth
        await asyncio.wait_for(run_started.wait(), timeout=2.0)

        # Verify it was stored in the global storage
        assert session_id in _session_id_auth_storage
        stored_auth = _session_id_auth_storage[session_id]
        assert stored_auth.user_api_key_auth.api_key == "session-key"

        # 2. Test POST Message Binding (Success)
        post_auth_context = (
            UserAPIKeyAuth(api_key="session-key", user_id="user1"),
            "auth-header",
            ["server1"],
            {},
            {},
            {},
        )

        post_scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/messages",
            "query_string": f"session_id={session_id.hex}".encode(),
            "headers": [],
        }

        with patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            AsyncMock(return_value=post_auth_context),
        ):
            # Should NOT raise HTTPException
            await handle_sse_post_messages(post_scope, mock_receive, mock_send)

        # 3. Test POST Message Binding (Mismatch - Security Fix)
        wrong_post_auth_context = (
            UserAPIKeyAuth(api_key="wrong-key", user_id="user2"),
            "auth-header",
            ["server1"],
            {},
            {},
            {},
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            AsyncMock(return_value=wrong_post_auth_context),
        ):
            with pytest.raises(HTTPException) as exc:
                await handle_sse_post_messages(post_scope, mock_receive, mock_send)
            assert exc.value.status_code == 403
            assert "Authentication mismatch" in exc.value.detail

        # 3b. Test POST Message Binding (No Auth - Security Fix)
        no_post_auth_context = (
            None,  # user_api_key_auth is None
            None,
            ["server1"],
            {},
            {},
            {},
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            AsyncMock(return_value=no_post_auth_context),
        ):
            with pytest.raises(HTTPException) as exc:
                await handle_sse_post_messages(post_scope, mock_receive, mock_send)
            assert exc.value.status_code == 403
            assert "Authentication mismatch" in exc.value.detail

        # Finish the run
        finish_run.set()
        await task

    # 4. Verify cleanup after session ends
    assert session_id not in _session_id_auth_storage


@pytest.mark.asyncio
async def test_anonymous_session_still_works():
    """
    Test that if a session was started without auth (anonymous),
    subsequent POST messages without auth are still allowed.
    """
    session_id = uuid.uuid4()
    mock_sse = MagicMock()
    mock_sse.handle_post_message = AsyncMock()

    @asynccontextmanager
    async def mock_connect_sse(scope, receive, send):
        _captured_session_id_var.set(session_id)
        yield (AsyncMock(), AsyncMock())

    mock_sse.connect_sse = MagicMock(side_effect=mock_connect_sse)
    mock_sse._read_stream_writers = {session_id: MagicMock()}

    # SSE session started WITHOUT auth
    anon_auth_context = (None, None, ["server1"], {}, {}, {})
    mock_scope = {"type": "http", "method": "GET", "path": "/mcp/sse", "headers": []}

    finish_run = asyncio.Event()

    async def mock_run(*args, **kwargs):
        await finish_run.wait()

    with (
        patch("litellm.proxy._experimental.mcp_server.server.sse", mock_sse),
        patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            AsyncMock(return_value=anon_auth_context),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server.server",
            MagicMock(run=mock_run, create_initialization_options=MagicMock()),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._gateway_initialize_instructions_request_scope",
            MagicMock(
                return_value=MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock())
            ),
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.server._SESSION_MANAGERS_INITIALIZED",
            True,
        ),
    ):
        task = asyncio.create_task(
            handle_sse_mcp_endpoint(mock_scope, AsyncMock(), AsyncMock())
        )
        await asyncio.sleep(0.1)

        # POST without auth
        post_scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/messages",
            "query_string": f"session_id={session_id.hex}".encode(),
            "headers": [],
        }

        with patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            AsyncMock(return_value=anon_auth_context),
        ):
            # Should NOT raise HTTPException
            await handle_sse_post_messages(post_scope, AsyncMock(), AsyncMock())

        finish_run.set()
        await task
