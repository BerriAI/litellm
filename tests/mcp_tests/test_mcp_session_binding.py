import pytest
import uuid
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from contextlib import asynccontextmanager
from litellm.proxy._experimental.mcp_server.server import (
    handle_sse_mcp_endpoint,
    handle_sse_post_messages,
    _captured_session_id_container_var,
    _session_id_auth_storage,
    _session_obj_auth_storage,
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

    # Track the streams yielded by mock_connect_sse so mock_run can verify
    # it receives the same objects — proving server.run() is called INSIDE
    # the async-with block, not after it exits (which would close streams).
    yielded_streams = {}

    # Mock connect_sse to simulate the SDK behavior
    @asynccontextmanager
    async def mock_connect_sse(scope, receive, send):
        # Directly set the ContextVar to simulate our capturing dict's behavior
        container = _captured_session_id_container_var.get()
        if container is not None:
            container["session_id"] = session_id
        read_stream = AsyncMock()
        write_stream = AsyncMock()
        yielded_streams["read"] = read_stream
        yielded_streams["write"] = write_stream
        yielded_streams["closed"] = False
        yield (read_stream, write_stream)
        yielded_streams["closed"] = True

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
        # Assert that server.run() is called INSIDE the connect_sse context
        # manager — i.e. the streams are still open.  If server.run() were
        # called AFTER the async-with exits, yielded_streams["closed"] would
        # be True and the first arg would NOT match the yielded read stream.
        assert (
            yielded_streams.get("closed") is False
        ), "server.run() called after connect_sse context exited — streams are closed"
        assert (
            len(args) >= 2
        ), "server.run() must receive (read_stream, write_stream, options)"
        assert (
            args[0] is yielded_streams["read"]
        ), "server.run() received a different read_stream than connect_sse yielded"
        assert (
            args[1] is yielded_streams["write"]
        ), "server.run() received a different write_stream than connect_sse yielded"
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
        await asyncio.wait_for(run_started.wait(), timeout=10.0)

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
            # Reset mock_send calls
            mock_send.reset_mock()
            await handle_sse_post_messages(post_scope, mock_receive, mock_send)

            # Verify 403 was sent
            assert mock_send.call_count >= 1
            # Check the status code in the http.response.start message
            status_sent = False
            for call in mock_send.call_args_list:
                msg = call[0][0]
                if msg.get("type") == "http.response.start":
                    assert msg.get("status") == 403
                    status_sent = True
            assert status_sent, "Should have sent a 403 status"

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
            # Reset mock_send calls
            mock_send.reset_mock()
            await handle_sse_post_messages(post_scope, mock_receive, mock_send)

            # Verify 403 was sent
            assert mock_send.call_count >= 1
            status_sent = False
            for call in mock_send.call_args_list:
                msg = call[0][0]
                if msg.get("type") == "http.response.start":
                    assert msg.get("status") == 403
                    status_sent = True
            assert status_sent, "Should have sent a 403 status"

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

    yielded_streams = {}

    @asynccontextmanager
    async def mock_connect_sse(scope, receive, send):
        container = _captured_session_id_container_var.get()
        if container is not None:
            container["session_id"] = session_id
        read_stream = AsyncMock()
        write_stream = AsyncMock()
        yielded_streams["read"] = read_stream
        yielded_streams["write"] = write_stream
        yielded_streams["closed"] = False
        yield (read_stream, write_stream)
        yielded_streams["closed"] = True

    mock_sse.connect_sse = MagicMock(side_effect=mock_connect_sse)
    mock_sse._read_stream_writers = {session_id: MagicMock()}

    # SSE session started WITHOUT auth
    anon_auth_context = (None, None, ["server1"], {}, {}, {})
    mock_scope = {"type": "http", "method": "GET", "path": "/mcp/sse", "headers": []}

    finish_run = asyncio.Event()

    async def mock_run(*args, **kwargs):
        assert (
            yielded_streams.get("closed") is False
        ), "server.run() called after connect_sse context exited — streams are closed"
        assert (
            len(args) >= 2
        ), "server.run() must receive (read_stream, write_stream, options)"
        assert (
            args[0] is yielded_streams["read"]
        ), "server.run() received a different read_stream than connect_sse yielded"
        assert (
            args[1] is yielded_streams["write"]
        ), "server.run() received a different write_stream than connect_sse yielded"
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


@pytest.mark.asyncio
async def test_session_obj_auth_storage_lazy_cache_and_fallback():
    """
    Verify that get_or_extract_auth_context:
    1. Lazily caches auth in _session_obj_auth_storage when ContextVar is set.
    2. Falls back to _session_obj_auth_storage when ContextVar is lost.

    This proves the robust id(session)-based lookup works without relying
    on the private _read_stream attribute.
    """
    from litellm.proxy._experimental.mcp_server.server import (
        get_or_extract_auth_context,
        set_auth_context,
    )

    # Create a fake session object (any object with a stable id)
    fake_session = MagicMock(name="FakeServerSession")
    session_id_key = id(fake_session)

    # Create a fake request_ctx
    fake_request_ctx_value = MagicMock()
    fake_request_ctx_value.session = fake_session

    test_auth = UserAPIKeyAuth(api_key="lazy-cache-key", user_id="lazy-user")

    # Ensure _session_obj_auth_storage starts clean for this session
    _session_obj_auth_storage.pop(session_id_key, None)

    # --- Step 1: ContextVar has auth, session is available ---
    # get_or_extract_auth_context should lazily cache auth.
    with patch(
        "mcp.server.lowlevel.server.request_ctx",
    ) as mock_ctx_var:
        mock_ctx_var.get.return_value = fake_request_ctx_value

        # Set the ContextVar-based auth
        set_auth_context(
            user_api_key_auth=test_auth,
            mcp_auth_header="auth-hdr",
            mcp_servers=["srv1"],
            mcp_server_auth_headers={},
            oauth2_headers={},
            raw_headers={},
            client_ip="127.0.0.1",
        )

        result = await get_or_extract_auth_context()
        assert result[0] is not None
        assert result[0].api_key == "lazy-cache-key"

        # Verify lazy cache was populated
        assert session_id_key in _session_obj_auth_storage
        cached = _session_obj_auth_storage[session_id_key]
        assert cached.user_api_key_auth.api_key == "lazy-cache-key"

    # --- Step 2: ContextVar is cleared (simulates SDK sub-task) ---
    # get_or_extract_auth_context should recover from _session_obj_auth_storage.
    set_auth_context(
        user_api_key_auth=None,
        mcp_auth_header=None,
        mcp_servers=None,
        mcp_server_auth_headers=None,
        oauth2_headers=None,
        raw_headers=None,
        client_ip=None,
    )

    with patch(
        "mcp.server.lowlevel.server.request_ctx",
    ) as mock_ctx_var:
        mock_ctx_var.get.return_value = fake_request_ctx_value

        result = await get_or_extract_auth_context()
        assert (
            result[0] is not None
        ), "Fallback to _session_obj_auth_storage failed — auth was lost"
        assert result[0].api_key == "lazy-cache-key"

    # Cleanup
    _session_obj_auth_storage.pop(session_id_key, None)
