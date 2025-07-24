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
