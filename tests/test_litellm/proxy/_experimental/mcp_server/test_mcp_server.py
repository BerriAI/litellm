import asyncio
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from mcp import ReadResourceResult, Resource
from mcp.types import Prompt, ResourceTemplate, TextResourceContents

from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    MCPTransport,
    UserAPIKeyAuth,
)
from litellm.types.mcp_server.mcp_server_manager import MCPServer


@pytest.fixture(autouse=True)
def cleanup_mcp_global_state():
    """Clean up MCP global state before and after each test.

    This fixture ensures test isolation when running with pytest-xdist
    parallel execution. Without this, global_mcp_server_manager state
    can leak between tests causing mock assertion failures.
    """
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        # Clear before test
        global_mcp_server_manager.registry.clear()
        global_mcp_server_manager.tool_name_to_mcp_server_name_mapping.clear()
        yield
        # Clear after test
        global_mcp_server_manager.registry.clear()
        global_mcp_server_manager.tool_name_to_mcp_server_name_mapping.clear()
    except ImportError:
        # MCP not available, skip cleanup
        yield


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


def test_prepare_mcp_server_headers_case_insensitive_extra_headers():
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _prepare_mcp_server_headers,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    server = MCPServer(
        server_id="server-case",
        name="server",
        transport=MCPTransport.http,
        extra_headers=["Authorization"],
    )

    server_auth_header, extra_headers = _prepare_mcp_server_headers(
        server=server,
        mcp_server_auth_headers=None,
        mcp_auth_header=None,
        oauth2_headers=None,
        raw_headers={"authorization": "Bearer token"},
    )

    assert server_auth_header is None
    assert extra_headers == {"Authorization": "Bearer token"}


@pytest.mark.asyncio
async def test_get_prompts_from_mcp_servers_success():
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_prompts_from_mcp_servers,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="test_user")

    server_a = MagicMock(name="server_a_obj")
    server_a.name = "server_a"
    server_a.alias = "server_a"
    server_a.server_name = "server_a"
    server_a.server_id = "a"
    server_a.auth_type = None
    server_a.extra_headers = None

    server_b = MagicMock(name="server_b_obj")
    server_b.name = "server_b"
    server_b.alias = "server_b"
    server_b.server_name = "server_b"
    server_b.server_id = "b"
    server_b.auth_type = None
    server_b.extra_headers = None

    with patch(
        "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
        AsyncMock(return_value=[server_a, server_b]),
    ) as mock_allowed, patch(
        "litellm.proxy._experimental.mcp_server.server._prepare_mcp_server_headers",
        return_value=(None, None),
    ) as mock_headers, patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
    ) as mock_manager:
        mock_manager.get_prompts_from_server = AsyncMock(
            side_effect=[
                [Prompt(name="hello", description="hi")],
                [Prompt(name="howdy", description="hey")],
            ]
        )

        prompts = await _get_prompts_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
        )

    mock_allowed.assert_awaited_once()
    assert mock_headers.call_count == 2
    assert mock_manager.get_prompts_from_server.await_count == 2
    assert {prompt.name for prompt in prompts} == {"hello", "howdy"}


@pytest.mark.asyncio
async def test_get_resources_from_mcp_servers_success():
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_resources_from_mcp_servers,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="user")

    server_a = MagicMock(name="server_a_obj")
    server_a.name = "server_a"
    server_a.alias = "server_a"
    server_a.server_name = "server_a"
    server_a.server_id = "a"
    server_a.auth_type = None
    server_a.extra_headers = None

    server_b = MagicMock(name="server_b_obj")
    server_b.name = "server_b"
    server_b.alias = "server_b"
    server_b.server_name = "server_b"
    server_b.server_id = "b"
    server_b.auth_type = None
    server_b.extra_headers = None

    with patch(
        "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
        AsyncMock(return_value=[server_a, server_b]),
    ) as mock_allowed, patch(
        "litellm.proxy._experimental.mcp_server.server._prepare_mcp_server_headers",
        return_value=(None, None),
    ) as mock_headers, patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
    ) as mock_manager:
        mock_manager.get_resources_from_server = AsyncMock(
            side_effect=[
                [
                    Resource(
                        name="resource_a",
                        uri="https://example.com/a",
                    )
                ],
                [
                    Resource(
                        name="resource_b",
                        uri="https://example.com/b",
                    )
                ],
            ]
        )

        resources = await _get_resources_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
        )

    mock_allowed.assert_awaited_once()
    assert mock_headers.call_count == 2
    assert mock_manager.get_resources_from_server.await_count == 2
    assert {resource.name for resource in resources} == {
        "resource_a",
        "resource_b",
    }


@pytest.mark.asyncio
async def test_get_resource_templates_from_mcp_servers_success():
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_resource_templates_from_mcp_servers,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="user")

    server = MagicMock(name="server_obj")
    server.name = "server"
    server.alias = "server"
    server.server_name = "server"
    server.server_id = "server-id"
    server.auth_type = None
    server.extra_headers = None

    with patch(
        "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
        AsyncMock(return_value=[server]),
    ) as mock_allowed, patch(
        "litellm.proxy._experimental.mcp_server.server._prepare_mcp_server_headers",
        return_value=(None, None),
    ) as mock_headers, patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
    ) as mock_manager:
        mock_manager.get_resource_templates_from_server = AsyncMock(
            return_value=[
                ResourceTemplate(
                    name="template",
                    description="desc",
                    uriTemplate="https://example.com/resource/{id}",
                )
            ]
        )

        templates = await _get_resource_templates_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
        )

    mock_allowed.assert_awaited_once()
    mock_headers.assert_called_once()
    mock_manager.get_resource_templates_from_server.assert_awaited_once()
    assert [template.name for template in templates] == ["template"]


@pytest.mark.asyncio
async def test_mcp_get_prompt_success():
    try:
        from litellm.proxy._experimental.mcp_server.server import mcp_get_prompt
    except ImportError:
        pytest.skip("MCP server not available")

    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="test_user")

    server = MagicMock()
    server.name = "server_a"

    prompt_result = MagicMock(name="prompt_result")

    with patch(
        "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
        AsyncMock(return_value=[server]),
    ) as mock_allowed, patch(
        "litellm.proxy._experimental.mcp_server.server._prepare_mcp_server_headers",
        return_value=({"Authorization": "token"}, {"X-Test": "1"}),
    ) as mock_headers, patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
    ) as mock_manager:
        mock_manager.get_prompt_from_server = AsyncMock(return_value=prompt_result)

        result = await mcp_get_prompt(
            name="hello",
            arguments={"foo": "bar"},
            user_api_key_auth=user_api_key_auth,
        )

    mock_allowed.assert_awaited_once()
    mock_headers.assert_called_once_with(
        server=server,
        mcp_server_auth_headers=None,
        mcp_auth_header=None,
        oauth2_headers=None,
        raw_headers=None,
    )
    mock_manager.get_prompt_from_server.assert_awaited_once_with(
        server=server,
        prompt_name="hello",
        arguments={"foo": "bar"},
        mcp_auth_header={"Authorization": "token"},
        extra_headers={"X-Test": "1"},
        raw_headers=None,
    )
    assert result is prompt_result


@pytest.mark.asyncio
async def test_mcp_read_resource_success():
    try:
        from litellm.proxy._experimental.mcp_server.server import mcp_read_resource
    except ImportError:
        pytest.skip("MCP server not available")

    user_api_key_auth = UserAPIKeyAuth(api_key="key", user_id="user")

    server = MagicMock()
    server.name = "server"

    read_result = ReadResourceResult(
        contents=[
            TextResourceContents(
                uri="https://example.com/resource",
                text="hello world",
                mimeType="text/plain",
            )
        ]
    )

    with patch(
        "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
        AsyncMock(return_value=[server]),
    ) as mock_allowed, patch(
        "litellm.proxy._experimental.mcp_server.server._prepare_mcp_server_headers",
        return_value=({"Authorization": "token"}, {"X-Test": "1"}),
    ) as mock_headers, patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
    ) as mock_manager:
        mock_manager.read_resource_from_server = AsyncMock(return_value=read_result)

        result = await mcp_read_resource(
            url="https://example.com/resource",
            user_api_key_auth=user_api_key_auth,
        )

    mock_allowed.assert_awaited_once()
    mock_headers.assert_called_once_with(
        server=server,
        mcp_server_auth_headers=None,
        mcp_auth_header=None,
        oauth2_headers=None,
        raw_headers=None,
    )
    mock_manager.read_resource_from_server.assert_awaited_once_with(
        server=server,
        url="https://example.com/resource",
        mcp_auth_header={"Authorization": "token"},
        extra_headers={"X-Test": "1"},
        raw_headers=None,
    )
    assert result is read_result


@pytest.mark.asyncio
async def test_mcp_read_resource_multiple_servers_error():
    try:
        from litellm.proxy._experimental.mcp_server.server import mcp_read_resource
    except ImportError:
        pytest.skip("MCP server not available")

    user_api_key_auth = UserAPIKeyAuth(api_key="key", user_id="user")

    server_a = MagicMock()
    server_b = MagicMock()
    server_a.name = "server_a"
    server_b.name = "server_b"

    with patch(
        "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
        AsyncMock(return_value=[server_a, server_b]),
    ) as mock_allowed:
        with pytest.raises(HTTPException) as exc_info:
            await mcp_read_resource(
                url="https://example.com/resource",
                user_api_key_auth=user_api_key_auth,
            )

    mock_allowed.assert_awaited_once()
    assert exc_info.value.status_code == 400
    assert "Multiple MCP servers" in str(exc_info.value.detail)


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
    working_server.allowed_tools = None
    working_server.disallowed_tools = None
    working_server.server_id = "working_server"
    working_server.server_name = "working_server"
    working_server.auth_type = None
    working_server.extra_headers = None

    failing_server = MagicMock()
    failing_server.name = "failing_server"
    failing_server.alias = "failing"
    failing_server.allowed_tools = None
    failing_server.disallowed_tools = None
    failing_server.server_id = "failing_server"
    failing_server.server_name = "failing_server"
    failing_server.auth_type = None
    failing_server.extra_headers = None

    # Mock global_mcp_server_manager
    mock_manager = MagicMock()
    mock_manager.get_allowed_mcp_servers = AsyncMock(
        return_value=["working_server", "failing_server"]
    )
    mock_manager.get_mcp_server_by_id = lambda server_id: (
        working_server if server_id == "working_server" else failing_server
    )
    # Mock filter_server_ids_by_ip to return server_ids unchanged (no IP filtering)
    mock_manager.filter_server_ids_by_ip = lambda server_ids, client_ip: server_ids

    async def mock_get_tools_from_server(
        server,
        mcp_auth_header=None,
        extra_headers=None,
        add_prefix=True,
        raw_headers=None,
    ):
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
                mcp_servers=["working_server", "failing_server"],
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
    failing_server1.allowed_tools = None
    failing_server1.disallowed_tools = None
    failing_server1.server_id = "failing_server1"
    failing_server1.server_name = "failing_server1"
    failing_server1.auth_type = None
    failing_server1.extra_headers = None

    failing_server2 = MagicMock()
    failing_server2.name = "failing_server2"
    failing_server2.alias = "failing2"
    failing_server2.allowed_tools = None
    failing_server2.disallowed_tools = None
    failing_server2.server_id = "failing_server2"
    failing_server2.server_name = "failing_server2"
    failing_server2.auth_type = None
    failing_server2.extra_headers = None

    # Mock global_mcp_server_manager
    mock_manager = MagicMock()
    mock_manager.get_allowed_mcp_servers = AsyncMock(
        return_value=["failing_server1", "failing_server2"]
    )
    mock_manager.get_mcp_server_by_id = lambda server_id: (
        failing_server1 if server_id == "failing_server1" else failing_server2
    )
    # Mock filter_server_ids_by_ip to return server_ids unchanged (no IP filtering)
    mock_manager.filter_server_ids_by_ip = lambda server_ids, client_ip: server_ids

    async def mock_get_tools_from_server(
        server,
        mcp_auth_header=None,
        extra_headers=None,
        add_prefix=True,
        raw_headers=None,
    ):
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
async def test_streamable_http_session_manager_is_stateless():
    """
    Test that the StreamableHTTPSessionManager is initialized with stateless=True.

    Regression test for GitHub issue #20242 / PR #19809.
    When stateless=False, the mcp library rejects non-initialize requests
    that lack an mcp-session-id header, breaking clients like MCP Inspector,
    curl, and any HTTP client without automatic session management.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import session_manager
    except ImportError:
        pytest.skip("MCP server not available")

    # The session manager must be stateless to avoid requiring mcp-session-id
    # on every request. This was regressed by PR #19809 (stateless=True -> False).
    assert session_manager.stateless is True, (
        "StreamableHTTPSessionManager must be initialized with stateless=True. "
        "stateless=False breaks MCP clients that don't manage session IDs. "
        "See: https://github.com/BerriAI/litellm/issues/20242"
    )


@pytest.mark.asyncio
@pytest.mark.no_parallel
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


@pytest.mark.asyncio
@pytest.mark.no_parallel
async def test_oauth2_headers_passed_to_mcp_client():
    """Test that OAuth2 headers are properly passed through to the MCP client for OAuth2 servers like github_mcp"""
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
            set_auth_context,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp import MCPAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP server not available")

    # Clear the registry to avoid conflicts with other tests
    global_mcp_server_manager.registry.clear()

    # Create an OAuth2 MCP server similar to github_mcp configuration
    oauth2_server = MCPServer(
        server_id="github_mcp_server_id",
        name="github_mcp",
        alias="github_mcp",
        transport=MCPTransport.http,
        url="https://api.githubcopilot.com/mcp",
        auth_type=MCPAuth.oauth2,
        client_id="test_github_client_id",
        client_secret="test_github_client_secret",
        scopes=["public_repo", "user:email"],
        authorization_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
    )
    global_mcp_server_manager.registry[oauth2_server.server_id] = oauth2_server

    # Mock user auth
    user_api_key_auth = UserAPIKeyAuth(api_key="test_key", user_id="test_user")

    # Set up OAuth2 headers that would come from the client
    oauth2_headers = {"Authorization": "Bearer github_oauth_token_12345"}

    # Set auth context with OAuth2 headers
    set_auth_context(user_api_key_auth=user_api_key_auth, oauth2_headers=oauth2_headers)

    # This will capture the arguments passed to _create_mcp_client
    captured_client_args = {}

    async def mock_create_mcp_client(
        server,
        mcp_auth_header=None,
        extra_headers=None,
        stdio_env=None,
    ):
        # Capture the arguments for verification
        captured_client_args.update(
            {
                "server": server,
                "mcp_auth_header": mcp_auth_header,
                "extra_headers": extra_headers,
                "stdio_env": stdio_env,
            }
        )
        # Return a mock client that doesn't actually connect
        mock_client = MagicMock()
        return mock_client

    # Mock _fetch_tools_with_timeout to avoid actual network calls
    async def mock_fetch_tools_with_timeout(client, server_name):
        return []  # Return empty list of tools

    with patch.object(
        global_mcp_server_manager,
        "_create_mcp_client",
        side_effect=mock_create_mcp_client,
    ) as mock_create_client, patch.object(
        global_mcp_server_manager,
        "_fetch_tools_with_timeout",
        side_effect=mock_fetch_tools_with_timeout,
    ), patch(
        "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
        AsyncMock(return_value=[oauth2_server]),
    ):
        # Call _get_tools_from_mcp_servers which should eventually call _create_mcp_client
        await _get_tools_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=None,  # Will use all allowed servers
            oauth2_headers=oauth2_headers,
        )

    # Verify that _create_mcp_client was called
    assert (
        mock_create_client.call_count == 1
    ), "Expected _create_mcp_client to be called once"

    # Verify the server passed to _create_mcp_client is the OAuth2 server
    assert captured_client_args["server"].server_id == oauth2_server.server_id
    assert captured_client_args["server"].auth_type == MCPAuth.oauth2

    # Most importantly: verify that OAuth2 headers were passed as extra_headers
    assert (
        captured_client_args["extra_headers"] is not None
    ), "Expected extra_headers to be passed for OAuth2 server"
    assert (
        captured_client_args["extra_headers"] == oauth2_headers
    ), f"Expected OAuth2 headers to be passed as extra_headers, got {captured_client_args['extra_headers']}"

    # Verify the Authorization header specifically
    assert "Authorization" in captured_client_args["extra_headers"]
    assert (
        captured_client_args["extra_headers"]["Authorization"]
        == "Bearer github_oauth_token_12345"
    )


@pytest.mark.asyncio
async def test_list_tools_single_server_unprefixed_names():
    """When only one MCP server is allowed, list tools should return unprefixed names."""
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

    # One allowed server
    server = MagicMock()
    server.server_id = "server1"
    server.name = "Zapier MCP"
    server.alias = "zapier"
    server.allowed_tools = None
    server.disallowed_tools = None
    server.server_name = "server1"
    server.auth_type = None
    server.extra_headers = None

    # Mock manager: allow just one server and return a tool based on add_prefix flag
    mock_manager = MagicMock()
    mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["server1"])
    mock_manager.get_mcp_server_by_id = MagicMock(return_value=server)
    # Mock filter_server_ids_by_ip to return server_ids unchanged (no IP filtering)
    mock_manager.filter_server_ids_by_ip = lambda server_ids, client_ip: server_ids

    async def mock_get_tools_from_server(
        server,
        mcp_auth_header=None,
        extra_headers=None,
        add_prefix=False,
        raw_headers=None,
    ):
        tool = MagicMock()
        tool.name = f"{server.alias}-toolA" if add_prefix else "toolA"
        tool.description = "desc"
        tool.inputSchema = {}
        return [tool]

    mock_manager._get_tools_from_server = mock_get_tools_from_server

    with patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        mock_manager,
    ):
        tools = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
        )

    # Should be unprefixed since only one server is allowed
    assert len(tools) == 1
    assert tools[0].name == "toolA"


@pytest.mark.asyncio
async def test_list_tools_multiple_servers_prefixed_names():
    """When multiple MCP servers are allowed, list tools should return prefixed names."""
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

    # Two allowed servers
    server1 = MagicMock()
    server1.server_id = "server1"
    server1.name = "Zapier MCP"
    server1.alias = "zapier"
    server1.allowed_tools = None
    server1.disallowed_tools = None
    server1.server_name = "server1"
    server1.auth_type = None
    server1.extra_headers = None

    server2 = MagicMock()
    server2.server_id = "server2"
    server2.name = "Jira MCP"
    server2.alias = "jira"
    server2.allowed_tools = None
    server2.disallowed_tools = None
    server2.server_name = "server2"
    server2.auth_type = None
    server2.extra_headers = None

    # Mock manager
    mock_manager = MagicMock()
    mock_manager.get_allowed_mcp_servers = AsyncMock(
        return_value=["server1", "server2"]
    )
    mock_manager.get_mcp_server_by_id = lambda server_id: (
        server1 if server_id == "server1" else server2
    )
    # Mock filter_server_ids_by_ip to return server_ids unchanged (no IP filtering)
    mock_manager.filter_server_ids_by_ip = lambda server_ids, client_ip: server_ids

    async def mock_get_tools_from_server(
        server,
        mcp_auth_header=None,
        extra_headers=None,
        add_prefix=True,
        raw_headers=None,
    ):
        tool = MagicMock()
        # When multiple servers, add_prefix should be True -> prefixed names
        tool.name = f"{server.alias}-toolA" if add_prefix else "toolA"
        tool.description = "desc"
        tool.inputSchema = {}
        return [tool]

    mock_manager._get_tools_from_server = mock_get_tools_from_server

    with patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        mock_manager,
    ):
        tools = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
        )

    # Should be prefixed since multiple servers are allowed
    names = sorted([t.name for t in tools])
    assert names == ["jira-toolA", "zapier-toolA"]


@pytest.mark.asyncio
async def test_mcp_manager_allows_public_servers_without_permissions():
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP server not available")

    manager = MCPServerManager()
    public_server = MCPServer(
        server_id="public",
        name="public",
        transport=MCPTransport.http,
        allow_all_keys=True,
    )
    manager.registry = {public_server.server_id: public_server}

    with patch(
        "litellm.proxy.management_endpoints.common_utils._user_has_admin_view",
        return_value=False,
    ), patch(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPRequestHandler.get_allowed_mcp_servers",
        AsyncMock(return_value=[]),
    ):
        allowed = await manager.get_allowed_mcp_servers(UserAPIKeyAuth())

    assert allowed == ["public"]


@pytest.mark.asyncio
async def test_mcp_manager_returns_public_when_permission_lookup_fails():
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP server not available")

    manager = MCPServerManager()
    public_server = MCPServer(
        server_id="public",
        name="public",
        transport=MCPTransport.http,
        allow_all_keys=True,
    )
    manager.registry = {public_server.server_id: public_server}

    with patch(
        "litellm.proxy.management_endpoints.common_utils._user_has_admin_view",
        return_value=False,
    ), patch(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPRequestHandler.get_allowed_mcp_servers",
        AsyncMock(side_effect=Exception("boom")),
    ):
        allowed = await manager.get_allowed_mcp_servers(UserAPIKeyAuth())

    assert allowed == ["public"]


@pytest.mark.asyncio
async def test_mcp_manager_merges_public_and_restricted_servers():
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            MCPServerManager,
        )
        from litellm.proxy._types import MCPTransport
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP server not available")

    manager = MCPServerManager()
    public_server = MCPServer(
        server_id="public",
        name="public",
        transport=MCPTransport.http,
        allow_all_keys=True,
    )
    scoped_server = MCPServer(
        server_id="restricted",
        name="restricted",
        transport=MCPTransport.http,
    )
    manager.registry = {
        public_server.server_id: public_server,
        scoped_server.server_id: scoped_server,
    }

    with patch(
        "litellm.proxy.management_endpoints.common_utils._user_has_admin_view",
        return_value=False,
    ), patch(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.MCPRequestHandler.get_allowed_mcp_servers",
        AsyncMock(return_value=["restricted"]),
    ):
        allowed = await manager.get_allowed_mcp_servers(UserAPIKeyAuth())

    assert set(allowed) == {"public", "restricted"}


@pytest.mark.asyncio
async def test_call_mcp_tool_user_unauthorized_access():
    """Test that a user cannot call a tool from a server they don't have access to"""
    from fastapi import HTTPException

    from litellm.proxy._experimental.mcp_server.server import call_mcp_tool
    from litellm.proxy._types import UserAPIKeyAuth

    # Create a mock user without access to the server
    mock_user_auth = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id="team-basic",
        object_permission_id="key-permission-123",
    )

    # Mock global_mcp_server_manager.get_mcp_server_by_id to return servers
    # only for allowed servers, not "restricted_server" (the server the user is trying to access)
    allowed_server_obj = MagicMock()
    allowed_server_obj.name = "allowed_server"
    allowed_server_obj.server_name = "allowed_server"
    allowed_server_obj.server_id = "allowed_server"
    allowed_server_obj.alias = "allowed_server"
    allowed_server_obj.allowed_tools = None
    allowed_server_obj.disallowed_tools = None
    allowed_server_obj.auth_type = None
    allowed_server_obj.extra_headers = None

    another_server_obj = MagicMock()
    another_server_obj.name = "another_server"
    another_server_obj.server_name = "another_server"
    another_server_obj.server_id = "another_server"
    another_server_obj.alias = "another_server"
    another_server_obj.allowed_tools = None
    another_server_obj.disallowed_tools = None
    another_server_obj.auth_type = None
    another_server_obj.extra_headers = None

    def mock_get_server_by_id(server_id):
        if server_id == "allowed_server":
            return allowed_server_obj
        elif server_id == "another_server":
            return another_server_obj
        return None

    with patch(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler.get_allowed_mcp_servers",
        AsyncMock(return_value=["allowed_server", "another_server"]),
    ), patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_mcp_server_by_id",
        side_effect=mock_get_server_by_id,
    ):
        # Try to call a tool from "restricted_server" - should raise HTTPException with 403 status
        with pytest.raises(HTTPException) as exc_info:
            await call_mcp_tool(
                name="restricted_server-send_email",
                arguments={
                    "to": "test@example.com",
                    "subject": "Test",
                    "body": "Test",
                },
                user_api_key_auth=mock_user_auth,
                mcp_auth_header="Bearer test_token",
            )

        # Verify the exception details
        assert exc_info.value.status_code == 403
        assert "User not allowed to call this tool" in exc_info.value.detail


@pytest.mark.asyncio
async def test_list_tools_filters_by_key_team_permissions():
    """Test that list_tools filters tools based on key/team mcp_tool_permissions"""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
            set_auth_context,
        )
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
    except ImportError:
        pytest.skip("MCP server not available")

    # Create object permission with tool-level restrictions
    object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="perm_123",
        mcp_tool_permissions={
            "server1": ["tool1", "tool2"],  # Only allow tool1 and tool2
        },
    )

    user_api_key_auth = UserAPIKeyAuth(
        api_key="test_key",
        user_id="test_user",
        object_permission=object_permission,
    )
    set_auth_context(user_api_key_auth)

    # Mock server
    server = MagicMock()
    server.server_id = "server1"
    server.name = "Test Server"
    server.alias = "test"
    server.allowed_tools = None
    server.disallowed_tools = None
    server.server_name = "server1"
    server.auth_type = None
    server.extra_headers = None

    # Mock manager
    mock_manager = MagicMock()
    mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["server1"])
    mock_manager.get_mcp_server_by_id = lambda server_id: server
    # Mock filter_server_ids_by_ip to return server_ids unchanged (no IP filtering)
    mock_manager.filter_server_ids_by_ip = lambda server_ids, client_ip: server_ids

    async def mock_get_tools_from_server(
        server,
        mcp_auth_header=None,
        extra_headers=None,
        add_prefix=False,
        raw_headers=None,
    ):
        # Return 4 tools, but only 2 should be allowed
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.description = "Tool 1"
        tool1.inputSchema = {}

        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2.description = "Tool 2"
        tool2.inputSchema = {}

        tool3 = MagicMock()
        tool3.name = "tool3"
        tool3.description = "Tool 3 - not allowed"
        tool3.inputSchema = {}

        tool4 = MagicMock()
        tool4.name = "tool4"
        tool4.description = "Tool 4 - not allowed"
        tool4.inputSchema = {}

        return [tool1, tool2, tool3, tool4]

    mock_manager._get_tools_from_server = mock_get_tools_from_server

    with patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        mock_manager,
    ):
        tools = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
        )

    # Should only return tool1 and tool2
    assert len(tools) == 2
    tool_names = sorted([t.name for t in tools])
    assert tool_names == ["tool1", "tool2"]


@pytest.mark.asyncio
async def test_list_tools_with_team_tool_permissions_inheritance():
    """Test that list_tools correctly applies key/team tool permissions inheritance logic"""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
            set_auth_context,
        )
        from litellm.proxy._types import (
            LiteLLM_ObjectPermissionTable,
            LiteLLM_TeamTable,
            UserAPIKeyAuth,
        )
    except ImportError:
        pytest.skip("MCP server not available")

    # Team allows tool1, tool2, tool3
    team_object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="team_perm_123",
        mcp_tool_permissions={
            "server1": ["tool1", "tool2", "tool3"],
        },
    )

    # Key allows tool2, tool3, tool4 - intersection should be tool2, tool3
    key_object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="key_perm_456",
        mcp_tool_permissions={
            "server1": ["tool2", "tool3", "tool4"],
        },
    )

    user_api_key_auth = UserAPIKeyAuth(
        api_key="test_key",
        user_id="test_user",
        team_id="team_123",
        object_permission=key_object_permission,
    )
    set_auth_context(user_api_key_auth)

    # Mock server
    server = MagicMock()
    server.server_id = "server1"
    server.name = "Test Server"
    server.alias = "test"
    server.allowed_tools = None
    server.disallowed_tools = None
    server.server_name = "server1"
    server.auth_type = None
    server.extra_headers = None

    # Mock manager
    mock_manager = MagicMock()
    mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["server1"])
    mock_manager.get_mcp_server_by_id = lambda server_id: server
    # Mock filter_server_ids_by_ip to return server_ids unchanged (no IP filtering)
    mock_manager.filter_server_ids_by_ip = lambda server_ids, client_ip: server_ids

    async def mock_get_tools_from_server(
        server,
        mcp_auth_header=None,
        extra_headers=None,
        add_prefix=False,
        raw_headers=None,
    ):
        # Return 4 tools
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.description = "Tool 1"
        tool1.inputSchema = {}

        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2.description = "Tool 2"
        tool2.inputSchema = {}

        tool3 = MagicMock()
        tool3.name = "tool3"
        tool3.description = "Tool 3"
        tool3.inputSchema = {}

        tool4 = MagicMock()
        tool4.name = "tool4"
        tool4.description = "Tool 4"
        tool4.inputSchema = {}

        return [tool1, tool2, tool3, tool4]

    mock_manager._get_tools_from_server = mock_get_tools_from_server

    with patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        mock_manager,
    ):
        # Mock the team object permission retrieval
        with patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_team_object_permission",
            AsyncMock(return_value=team_object_permission),
        ):
            tools = await _get_tools_from_mcp_servers(
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=None,
                mcp_servers=None,
                mcp_server_auth_headers=None,
            )

    # Should only return tool2 and tool3 (intersection of key and team permissions)
    assert len(tools) == 2
    tool_names = sorted([t.name for t in tools])
    assert tool_names == ["tool2", "tool3"]


@pytest.mark.asyncio
async def test_list_tools_with_no_tool_permissions_shows_all():
    """Test that list_tools shows all tools when no mcp_tool_permissions are set"""
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
            set_auth_context,
        )
        from litellm.proxy._types import UserAPIKeyAuth
    except ImportError:
        pytest.skip("MCP server not available")

    # No tool-level restrictions
    user_api_key_auth = UserAPIKeyAuth(
        api_key="test_key",
        user_id="test_user",
        object_permission=None,
    )
    set_auth_context(user_api_key_auth)

    # Mock server
    server = MagicMock()
    server.server_id = "server1"
    server.name = "Test Server"
    server.alias = "test"
    server.allowed_tools = None
    server.disallowed_tools = None
    server.server_name = "server1"
    server.auth_type = None
    server.extra_headers = None

    # Mock manager
    mock_manager = MagicMock()
    mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["server1"])
    mock_manager.get_mcp_server_by_id = lambda server_id: server
    # Mock filter_server_ids_by_ip to return server_ids unchanged (no IP filtering)
    mock_manager.filter_server_ids_by_ip = lambda server_ids, client_ip: server_ids

    async def mock_get_tools_from_server(
        server,
        mcp_auth_header=None,
        extra_headers=None,
        add_prefix=False,
        raw_headers=None,
    ):
        # Return 3 tools
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.description = "Tool 1"
        tool1.inputSchema = {}

        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2.description = "Tool 2"
        tool2.inputSchema = {}

        tool3 = MagicMock()
        tool3.name = "tool3"
        tool3.description = "Tool 3"
        tool3.inputSchema = {}

        return [tool1, tool2, tool3]

    mock_manager._get_tools_from_server = mock_get_tools_from_server

    with patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        mock_manager,
    ):
        tools = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
        )

    # Should return all tools when no restrictions
    assert len(tools) == 3
    tool_names = sorted([t.name for t in tools])
    assert tool_names == ["tool1", "tool2", "tool3"]


@pytest.mark.asyncio
async def test_list_tools_strips_prefix_when_matching_permissions():
    """
    Test that tool permission filtering correctly strips prefixes from tool names.

    Tools from MCP servers are prefixed (e.g., "GITMCP-fetch_litellm_documentation"),
    but allowed tools in DB are stored without prefix (e.g., "fetch_litellm_documentation").
    The filtering should strip the prefix before comparing.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
            set_auth_context,
        )
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
    except ImportError:
        pytest.skip("MCP server not available")

    # Create object permission with tool-level restrictions (WITHOUT prefix)
    object_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="perm_123",
        mcp_tool_permissions={
            "gitmcp_server": [
                "fetch_litellm_documentation",  # No prefix in DB
                "search_litellm_code",  # No prefix in DB
            ],
        },
    )

    user_api_key_auth = UserAPIKeyAuth(
        api_key="test_key",
        user_id="test_user",
        object_permission=object_permission,
    )
    set_auth_context(user_api_key_auth)

    # Mock server
    server = MagicMock()
    server.server_id = "gitmcp_server"
    server.name = "GITMCP"
    server.alias = "gitmcp"
    server.allowed_tools = None
    server.disallowed_tools = None

    # Mock manager
    mock_manager = MagicMock()
    mock_manager.get_allowed_mcp_servers = AsyncMock(return_value=["gitmcp_server"])
    mock_manager.get_mcp_server_by_id = MagicMock(return_value=server)
    # Mock filter_server_ids_by_ip to return server_ids unchanged (no IP filtering)
    mock_manager.filter_server_ids_by_ip = lambda server_ids, client_ip: server_ids

    async def mock_get_tools_from_server(
        server,
        mcp_auth_header=None,
        extra_headers=None,
        add_prefix=True,
        raw_headers=None,
    ):
        # Return tools WITH prefix (as they come from MCP server)
        tool1 = MagicMock()
        tool1.name = "GITMCP-fetch_litellm_documentation"  # Prefixed
        tool1.description = "Fetch docs"
        tool1.inputSchema = {}

        tool2 = MagicMock()
        tool2.name = (
            "GITMCP-search_litellm_documentation"  # Prefixed, not in allowed list
        )
        tool2.description = "Search docs"
        tool2.inputSchema = {}

        tool3 = MagicMock()
        tool3.name = "GITMCP-search_litellm_code"  # Prefixed
        tool3.description = "Search code"
        tool3.inputSchema = {}

        tool4 = MagicMock()
        tool4.name = "GITMCP-fetch_generic_url_content"  # Prefixed, not in allowed list
        tool4.description = "Fetch URL"
        tool4.inputSchema = {}

        return [tool1, tool2, tool3, tool4]

    mock_manager._get_tools_from_server = mock_get_tools_from_server

    with patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
        mock_manager,
    ):
        tools = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
        )

    # Should only return the 2 tools that match (after stripping prefix)
    assert len(tools) == 2
    tool_names = sorted([t.name for t in tools])
    # Tools still have prefixes in the output, but were filtered correctly
    assert tool_names == [
        "GITMCP-fetch_litellm_documentation",
        "GITMCP-search_litellm_code",
    ]


def test_filter_tools_by_allowed_tools():
    """Test that filter_tools_by_allowed_tools filters tools correctly"""
    from mcp.types import Tool

    from litellm.proxy._experimental.mcp_server.server import (
        filter_tools_by_allowed_tools,
    )
    from litellm.types.mcp import MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    mcp_server = MCPServer(
        server_id="my_api_mcp",
        name="my_api_mcp",
        alias="my_api_mcp",
        transport=MCPTransport.http,
        allowed_tools=["getpetbyid", "my_api_mcp-findpetsbystatus"],
        disallowed_tools=None,
    )
    tools_to_return = [
        Tool(
            name="my_api_mcp-getpetbyid",
            title=None,
            description="Find pet by ID",
            inputSchema={
                "type": "object",
                "properties": {"petId": {"type": "integer", "description": ""}},
                "required": ["petId"],
            },
            outputSchema=None,
            annotations=None,
        ),
        Tool(
            name="my_api_mcp-findpetsbystatus",
            title=None,
            description="Finds Pets by status",
            inputSchema={
                "type": "object",
                "properties": {"status": {"type": "string", "description": ""}},
                "required": ["status"],
            },
            outputSchema=None,
            annotations=None,
        ),
        Tool(
            name="my_api_mcp-addpet",
            title=None,
            description="Add a new pet to the store",
            inputSchema={
                "type": "object",
                "properties": {
                    "body": {
                        "type": "object",
                        "description": "Request body",
                        "properties": {
                            "name": {"type": "string"},
                            "status": {"type": "string"},
                        },
                    }
                },
                "required": ["body"],
            },
            outputSchema=None,
            annotations=None,
        ),
    ]

    filtered_tools = filter_tools_by_allowed_tools(tools_to_return, mcp_server)

    assert len(filtered_tools) == 2
    assert filtered_tools[0].name == "my_api_mcp-getpetbyid"
    assert filtered_tools[1].name == "my_api_mcp-findpetsbystatus"


def _make_db_mcp_server(server_id: str, updated_at: datetime) -> LiteLLM_MCPServerTable:
    return LiteLLM_MCPServerTable(
        server_id=server_id,
        server_name="server",
        alias="server",
        url="https://example.com",
        transport=MCPTransport.http,
        created_at=updated_at,
        updated_at=updated_at,
        mcp_info={},
    )


class TestMCPServerManagerReload:
    @pytest.mark.asyncio
    async def test_reuses_existing_server_when_updated_at_matches(self):
        try:
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                MCPServerManager,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        manager = MCPServerManager()
        timestamp = datetime.utcnow()
        existing_server = MCPServer(
            server_id="server-1",
            name="server",
            transport=MCPTransport.http,
            updated_at=timestamp,
        )
        manager.registry = {existing_server.server_id: existing_server}

        db_row = _make_db_mcp_server("server-1", timestamp)

        with patch(
            "litellm.proxy._experimental.mcp_server.db.get_all_mcp_servers",
            new=AsyncMock(return_value=[db_row]),
        ) as mock_get_all, patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=object(),
        ), patch.object(
            manager, "build_mcp_server_from_table", AsyncMock()
        ) as mock_build:
            await manager.reload_servers_from_database()

        mock_get_all.assert_awaited_once()
        mock_build.assert_not_awaited()
        assert manager.registry["server-1"] is existing_server

    @pytest.mark.asyncio
    async def test_rebuilds_server_when_updated_at_changes(self):
        try:
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                MCPServerManager,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        manager = MCPServerManager()
        timestamp = datetime.utcnow()
        existing_server = MCPServer(
            server_id="server-1",
            name="server",
            transport=MCPTransport.http,
            updated_at=timestamp,
        )
        manager.registry = {existing_server.server_id: existing_server}

        new_timestamp = timestamp + timedelta(minutes=5)
        db_row = _make_db_mcp_server("server-1", new_timestamp)
        rebuilt_server = MCPServer(
            server_id="server-1",
            name="server",
            transport=MCPTransport.http,
            updated_at=new_timestamp,
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.db.get_all_mcp_servers",
            new=AsyncMock(return_value=[db_row]),
        ) as mock_get_all, patch(
            "litellm.proxy.management_endpoints.mcp_management_endpoints.get_prisma_client_or_throw",
            return_value=object(),
        ), patch.object(
            manager,
            "build_mcp_server_from_table",
            AsyncMock(return_value=rebuilt_server),
        ) as mock_build:
            await manager.reload_servers_from_database()

        mock_get_all.assert_awaited_once()
        mock_build.assert_awaited_once_with(db_row)
        assert manager.registry["server-1"] is rebuilt_server


@pytest.mark.asyncio
async def test_call_mcp_tool_logs_failure_via_post_call_failure_hook():
    """
    Regression test for 6267f168...:
    Ensure proxy-side `call_mcp_tool` logs failures via `proxy_logging_obj.post_call_failure_hook`.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            call_mcp_tool,
            global_mcp_server_manager,
        )
        from litellm.proxy._types import MCPTransport, UserAPIKeyAuth
        from litellm.types.mcp_server.mcp_server_manager import MCPServer
    except ImportError:
        pytest.skip("MCP server not available")

    mock_server = MCPServer(
        server_id="server-123",
        name="test_server",
        alias="test_server",
        server_name="test_server",
        url="https://test-server.com/mcp",
        transport=MCPTransport.http,
        mcp_info={"server_name": "test_server"},
    )

    proxy_logging_mock = MagicMock()
    proxy_logging_mock.post_call_failure_hook = AsyncMock()

    user_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user")

    with patch.object(
        global_mcp_server_manager,
        "get_allowed_mcp_servers",
        new_callable=AsyncMock,
        return_value=[mock_server.server_id],
    ), patch.object(
        global_mcp_server_manager,
        "get_mcp_server_by_id",
        return_value=mock_server,
    ), patch(
        "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers_from_mcp_server_names",
        new_callable=AsyncMock,
        return_value=[mock_server],
    ), patch(
        "litellm.proxy._experimental.mcp_server.server.execute_mcp_tool",
        new_callable=AsyncMock,
        side_effect=Exception("boom"),
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj",
        proxy_logging_mock,
    ):
        with pytest.raises(Exception):
            await call_mcp_tool(
                name="test_server-any_tool",
                arguments={"x": 1},
                user_api_key_auth=user_auth,
                litellm_call_id="cid",
            )

    proxy_logging_mock.post_call_failure_hook.assert_awaited_once()
    assert (
        proxy_logging_mock.post_call_failure_hook.await_args.kwargs.get("route")
        == "/mcp/call_tool"
    )


@pytest.mark.asyncio
async def test_get_tools_from_mcp_servers_logs_list_tools_to_spendlogs_when_enabled():
    """
    Regression test for 872e5b98...:
    Ensure list-tools logging path calls `async_success_handler` when enabled.
    """
    try:
        from litellm.proxy._experimental.mcp_server.server import (
            _get_tools_from_mcp_servers,
        )
        from litellm.proxy._types import UserAPIKeyAuth
    except ImportError:
        pytest.skip("MCP server not available")

    user_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user")

    server_a = MagicMock(name="server_a_obj")
    server_a.name = "server_a"
    server_a.alias = "server_a"
    server_a.server_name = "server_a"
    server_a.server_id = "a"
    server_a.auth_type = None
    server_a.extra_headers = None

    tool_1 = MagicMock()
    tool_1.name = "server_a-tool_1"

    dummy_logging_obj = MagicMock()
    dummy_logging_obj.model_call_details = {"metadata": {"spend_logs_metadata": {}}}
    dummy_logging_obj.async_success_handler = AsyncMock()

    with patch(
        "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
        new=AsyncMock(return_value=[server_a]),
    ), patch(
        "litellm.proxy._experimental.mcp_server.server._prepare_mcp_server_headers",
        return_value=(None, None),
    ), patch(
        "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager",
    ) as mock_manager, patch(
        "litellm.proxy._experimental.mcp_server.server.filter_tools_by_allowed_tools",
        side_effect=lambda tools, _server: tools,
    ), patch(
        "litellm.proxy._experimental.mcp_server.server.filter_tools_by_key_team_permissions",
        new=AsyncMock(side_effect=lambda tools, **_: tools),
    ), patch(
        "litellm.proxy._experimental.mcp_server.server.function_setup",
        return_value=(dummy_logging_obj, None),
    ):
        mock_manager._get_tools_from_server = AsyncMock(return_value=[tool_1])

        tools = await _get_tools_from_mcp_servers(
            user_api_key_auth=user_auth,
            mcp_auth_header=None,
            mcp_servers=["server_a"],
            mcp_server_auth_headers=None,
            log_list_tools_to_spendlogs=True,
            list_tools_log_source="mcp_protocol",
        )

    assert tools == [tool_1]
    dummy_logging_obj.async_success_handler.assert_awaited_once()
    assert dummy_logging_obj.async_success_handler.await_args.kwargs["result"] == [tool_1]

    spend_meta = dummy_logging_obj.model_call_details["metadata"]["spend_logs_metadata"]
    assert spend_meta["tool_count_total"] == 1
    assert spend_meta["allowed_server_count"] == 1
    assert spend_meta["per_server_tool_counts"]["server_a"] == 1
