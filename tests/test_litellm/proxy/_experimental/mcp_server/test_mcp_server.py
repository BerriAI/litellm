import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import UserAPIKeyAuth


@pytest.mark.asyncio
async def test_mcp_server_tool_call_body_contains_request_data():
    """Test that proxy_server_request body contains name and arguments"""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            mcp_server_tool_call,
            set_auth_context,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    # Setup test data
    tool_name = "test_tool"
    tool_arguments = {"param1": "value1", "param2": 123}

    # Mock user auth
    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="test_user")
    set_auth_context(user_api_key_auth)

    # Mock the add_litellm_data_to_request function to capture the data
    captured_data = {}

    async def mock_add_litellm_data_to_request(
        data, request, user_api_key_dict, proxy_config
    ):
        captured_data.update(data)
        # Simulate the proxy_server_request creation
        captured_data["proxy_server_request"] = {
            "url": str(request.url),
            "method": request.method,
            "headers": {},
            "body": data.copy(),  # This is what we want to test
        }
        return captured_data

    # Mock the call_mcp_tool function to avoid actual tool execution
    async def mock_call_mcp_tool(*args, **kwargs):
        return [{"type": "text", "text": "mocked response"}]

    with patch(
        "litellm.proxy.litellm_pre_call_utils.add_litellm_data_to_request",
        mock_add_litellm_data_to_request,
    ):
        with patch(
            "litellm.proxy._experimental.mcp_server.server.call_mcp_tool",
            mock_call_mcp_tool,
        ):
            with patch(
                "litellm.proxy.proxy_server.proxy_config",
                MagicMock(),
            ):
                # Call the function
                await mcp_server_tool_call(tool_name, tool_arguments)

    # Verify the body contains the expected data
    assert "proxy_server_request" in captured_data
    assert "body" in captured_data["proxy_server_request"]

    body = captured_data["proxy_server_request"]["body"]
    assert body["name"] == tool_name
    assert body["arguments"] == tool_arguments


@pytest.mark.asyncio
async def test_get_tools_from_mcp_servers_continues_when_one_server_fails():
    """Test that _get_tools_from_mcp_servers continues when one server fails"""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
            set_auth_context,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    # Mock user auth
    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="test_user")
    set_auth_context(user_api_key_auth)

    # Mock servers
    working_server = MagicMock()
    working_server.name = "working_server"
    working_server.alias = "working"

    failing_server = MagicMock()
    failing_server.name = "failing_server"
    failing_server.alias = "failing"

    # Mock global_mcp_server_manager
    mock_manager = MagicMock()
    mock_manager.get_allowed_mcp_servers = AsyncMock(
        return_value=["working_server", "failing_server"]
    )
    mock_manager.get_mcp_server_by_id = lambda server_id: (
        working_server if server_id == "working_server" else failing_server
    )

    async def mock_get_tools_from_server(server, mcp_auth_header=None):
        if server.name == "working_server":
            # Working server returns tools
            tool1 = MagicMock()
            tool1.name = "working_tool_1"
            tool1.description = "Working tool 1"
            tool1.inputSchema = {}
            return [tool1]
        else:
            # Failing server raises an exception
            raise Exception("Server connection failed")

    mock_manager._get_tools_from_server = mock_get_tools_from_server

    with patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        mock_manager,
    ):
        with patch(
            "litellm.proxy._experimental.mcp_server.server.verbose_logger",
        ) as mock_logger:
            # Test with server-specific auth headers
            mcp_server_auth_headers = {
                "working": "Bearer working-token",
                "failing": "Bearer failing-token",
            }

            result = await _get_tools_from_mcp_servers(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=None,
                mcp_servers=None,
                mcp_server_auth_headers=mcp_server_auth_headers,
            )

            # Verify that tools from the working server are returned
            assert len(result) == 1
            assert result[0].name == "working_tool_1"

            # Verify failure logging
            mock_logger.exception.assert_any_call(
                "Error getting tools from server failing_server: Server connection failed"
            )

            # Verify success logging
            mock_logger.info.assert_any_call(
                "Successfully fetched 1 tools total from all MCP servers"
            )


@pytest.mark.asyncio
async def test_get_tools_from_mcp_servers_handles_all_servers_failing():
    """Test that _get_tools_from_mcp_servers handles all servers failing gracefully"""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
            set_auth_context,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    # Mock user auth
    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="test_user")
    set_auth_context(user_api_key_auth)

    # Mock servers
    failing_server1 = MagicMock()
    failing_server1.name = "failing_server1"
    failing_server1.alias = "failing1"

    failing_server2 = MagicMock()
    failing_server2.name = "failing_server2"
    failing_server2.alias = "failing2"

    # Mock global_mcp_server_manager
    mock_manager = MagicMock()
    mock_manager.get_allowed_mcp_servers = AsyncMock(
        return_value=["failing_server1", "failing_server2"]
    )
    mock_manager.get_mcp_server_by_id = lambda server_id: (
        failing_server1 if server_id == "failing_server1" else failing_server2
    )

    async def mock_get_tools_from_server(server, mcp_auth_header=None):
        # All servers fail
        raise Exception(f"Server {server.name} connection failed")

    mock_manager._get_tools_from_server = mock_get_tools_from_server

    with patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        mock_manager,
    ):
        with patch(
            "litellm.proxy._experimental.mcp_server.server.verbose_logger",
        ) as mock_logger:
            # Test with server-specific auth headers
            mcp_server_auth_headers = {
                "failing1": "Bearer failing1-token",
                "failing2": "Bearer failing2-token",
            }

            result = await _get_tools_from_mcp_servers(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=None,
                mcp_servers=None,
                mcp_server_auth_headers=mcp_server_auth_headers,
            )

            # Verify that empty list is returned
            assert len(result) == 0

            # Verify failure logging for both servers
            mock_logger.exception.assert_any_call(
                "Error getting tools from server failing_server1: Server failing_server1 connection failed"
            )
            mock_logger.exception.assert_any_call(
                "Error getting tools from server failing_server2: Server failing_server2 connection failed"
            )

            # Verify total logging
            mock_logger.info.assert_any_call(
                "Successfully fetched 0 tools total from all MCP servers"
            )


@pytest.mark.asyncio
async def test_mcp_server_tool_call_body_with_none_arguments():
    """Test that proxy_server_request body handles None arguments correctly"""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            mcp_server_tool_call,
            set_auth_context,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    # Setup test data
    tool_name = "test_tool_no_args"
    tool_arguments = None

    # Mock user auth
    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="test_user")
    set_auth_context(user_api_key_auth)

    # Mock the add_litellm_data_to_request function to capture the data
    captured_data = {}

    async def mock_add_litellm_data_to_request(
        data, request, user_api_key_dict, proxy_config
    ):
        captured_data.update(data)
        captured_data["proxy_server_request"] = {
            "url": str(request.url),
            "method": request.method,
            "headers": {},
            "body": data.copy(),
        }
        return captured_data

    # Mock the call_mcp_tool function
    async def mock_call_mcp_tool(*args, **kwargs):
        return [{"type": "text", "text": "mocked response"}]

    with patch(
        "litellm.proxy.litellm_pre_call_utils.add_litellm_data_to_request",
        mock_add_litellm_data_to_request,
    ):
        with patch(
            "litellm.proxy._experimental.mcp_server.server.call_mcp_tool",
            mock_call_mcp_tool,
        ):
            with patch(
                "litellm.proxy.proxy_server.proxy_config",
                MagicMock(),
            ):
                # Call the function
                await mcp_server_tool_call(tool_name, tool_arguments)

    # Verify the body contains the expected data
    assert "proxy_server_request" in captured_data
    assert "body" in captured_data["proxy_server_request"]

    body = captured_data["proxy_server_request"]["body"]
    assert body["name"] == tool_name
    assert body["arguments"] == tool_arguments  # Should be None


@pytest.mark.asyncio
async def test_concurrent_initialize_session_managers():
    """Test that concurrent calls to initialize_session_managers don't cause race conditions."""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _INITIALIZATION_LOCK,
            _SESSION_MANAGERS_INITIALIZED,
            initialize_session_managers,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    # Import the module to reset state
    import litellm.proxy._experimental.mcp_server.server as mcp_server

    # Reset state before test
    original_initialized = mcp_server._SESSION_MANAGERS_INITIALIZED
    original_session_cm = mcp_server._session_manager_cm
    original_sse_session_cm = mcp_server._sse_session_manager_cm

    try:
        mcp_server._SESSION_MANAGERS_INITIALIZED = False
        mcp_server._session_manager_cm = None
        mcp_server._sse_session_manager_cm = None

        # Mock the session managers to avoid actual MCP initialization
        with patch(
            "litellm.proxy._experimental.mcp_server.server.session_manager"
        ) as mock_session_manager, patch(
            "litellm.proxy._experimental.mcp_server.server.sse_session_manager"
        ) as mock_sse_session_manager, patch(
            "litellm.proxy._experimental.mcp_server.server.verbose_logger"
        ):
            # Mock the run() method to return a mock context manager
            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock()
            mock_cm.__aexit__ = AsyncMock()

            mock_session_manager.run.return_value = mock_cm
            mock_sse_session_manager.run.return_value = mock_cm

            # Create multiple concurrent tasks that call initialize_session_managers
            async def init_task():
                await initialize_session_managers()
                return "success"

            # Run 10 concurrent initialization attempts
            tasks = [init_task() for _ in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All tasks should complete successfully (no exceptions)
            assert all(
                result == "success" for result in results
            ), f"Some tasks failed: {results}"

            # session_manager.run() should only be called once due to the lock
            assert (
                mock_session_manager.run.call_count == 1
            ), f"Expected 1 call to session_manager.run(), got {mock_session_manager.run.call_count}"
            assert (
                mock_sse_session_manager.run.call_count == 1
            ), f"Expected 1 call to sse_session_manager.run(), got {mock_sse_session_manager.run.call_count}"

            # The context managers should only be entered once each
            assert (
                mock_cm.__aenter__.call_count == 2
            ), f"Expected 2 calls to __aenter__ (one for each session manager), got {mock_cm.__aenter__.call_count}"

            # State should be properly set
            assert mcp_server._SESSION_MANAGERS_INITIALIZED is True

    finally:
        # Restore original state
        mcp_server._SESSION_MANAGERS_INITIALIZED = original_initialized
        mcp_server._session_manager_cm = original_session_cm
        mcp_server._sse_session_manager_cm = original_sse_session_cm


@pytest.mark.asyncio
async def test_mcp_routing_with_conflicting_alias_and_group_name():
    """
    Tests (GH #14536) where an MCP server alias (e.g., "group/id")
    conflicts with an access group name (e.g., "group").
    """
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._experimental.mcp_server.server import (
            _get_mcp_servers_in_path,
            _get_tools_from_mcp_servers,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP server not available")

    global_mcp_server_manager.registry.clear()

    # Create two in-memory servers
    specific_server = MCPServer(
        server_id="specific_server_id",
        name="custom_solutions/user_123",
        alias="custom_solutions/user_123",
        transport=MCPTransport.http,
    )
    other_server = MCPServer(
        server_id="other_server_in_group_id",
        name="custom_solutions/another_user_456",
        alias="custom_solutions/another_user_456",
        transport=MCPTransport.http,
    )
    global_mcp_server_manager.registry[specific_server.server_id] = specific_server
    global_mcp_server_manager.registry[other_server.server_id] = other_server

    user_key = UserAPIKeyAuth(api_key="sk-test", team_id="team_custom_solutions")

    # Define the request path that triggers the bug
    test_path = "/mcp/custom_solutions/user_123/chat/completions"

    # This mock will be our "spy" to see which servers are ultimately contacted
    mock_get_tools_spy = AsyncMock(return_value=[])

    # Mock the function that checks DB for an access group named "custom_solutions"
    mock_db_lookup = AsyncMock(
        return_value=[specific_server.server_id, other_server.server_id]
    )

    mock_get_allowed = AsyncMock(
        return_value=[specific_server.server_id, other_server.server_id]
    )

    with patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_allowed_mcp_servers",
        mock_get_allowed,
    ), patch(
        "litellm.proxy._experimental.mcp_server.server.MCPRequestHandler._get_mcp_servers_from_access_groups",
        mock_db_lookup,
    ), patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager._get_tools_from_server",
        mock_get_tools_spy,
    ):
        mcp_servers_from_path = _get_mcp_servers_in_path(test_path)

        await _get_tools_from_mcp_servers(
            user_api_key_auth=user_key,
            mcp_servers=mcp_servers_from_path,
            mcp_auth_header=None,
        )

    # Get the list of actual server objects that the orchestrator tried to contact
    called_servers = [
        call.kwargs["server"] for call in mock_get_tools_spy.call_args_list
    ]

    assert len(called_servers) == 1, "Should have resolved to exactly one server."
    assert (
        called_servers[0].server_id == specific_server.server_id
    ), "Should have contacted the specific server alias, not the group."
