import pytest
from unittest.mock import AsyncMock, MagicMock

from litellm import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.mcp_tool_preprocessor import _PROXY_MCPToolPreprocessor


@pytest.mark.asyncio
async def test_mcp_tool_preprocessor_normalizes_missing_server_url():
    """Test that MCP tools with missing server_url get normalized to 'litellm_proxy'"""
    
    # Initialize the hook
    hook = _PROXY_MCPToolPreprocessor()
    
    # Mock dependencies
    user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
    cache = MagicMock(spec=DualCache)
    
    # Test data with MCP tool missing server_url
    data = {
        "model": "gpt-4",
        "tools": [
            {
                "type": "mcp",
                "name": "test_tool",
                "description": "A test tool",
                "require_approval": "never"
                # Note: server_url is missing
            }
        ],
        "messages": [{"role": "user", "content": "Test message"}]
    }
    
    # Call the hook
    result = await hook.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="completion"
    )
    
    # Verify the result
    assert result is not None
    assert "tools" in result
    assert len(result["tools"]) == 1
    
    tool = result["tools"][0]
    assert tool["type"] == "mcp"
    assert tool["server_url"] == "litellm_proxy"  # Should be normalized
    assert tool["name"] == "test_tool"
    assert tool["require_approval"] == "never"


@pytest.mark.asyncio
async def test_mcp_tool_preprocessor_preserves_existing_server_url():
    """Test that MCP tools with existing server_url are preserved"""
    
    # Initialize the hook
    hook = _PROXY_MCPToolPreprocessor()
    
    # Mock dependencies
    user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
    cache = MagicMock(spec=DualCache)
    
    # Test data with MCP tool having existing server_url
    data = {
        "model": "gpt-4",
        "tools": [
            {
                "type": "mcp",
                "name": "external_tool",
                "description": "An external tool",
                "server_url": "https://external-server.com/mcp",
                "require_approval": "always"
            }
        ],
        "messages": [{"role": "user", "content": "Test message"}]
    }
    
    # Call the hook
    result = await hook.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="completion"
    )
    
    # Verify the result
    assert result is not None
    assert "tools" in result
    assert len(result["tools"]) == 1
    
    tool = result["tools"][0]
    assert tool["type"] == "mcp"
    assert tool["server_url"] == "https://external-server.com/mcp"  # Should be preserved
    assert tool["name"] == "external_tool"
    assert tool["require_approval"] == "always"


@pytest.mark.asyncio
async def test_mcp_tool_preprocessor_mixed_tools():
    """Test that both MCP and non-MCP tools are handled correctly"""
    
    # Initialize the hook
    hook = _PROXY_MCPToolPreprocessor()
    
    # Mock dependencies
    user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
    cache = MagicMock(spec=DualCache)
    
    # Test data with mixed tool types
    data = {
        "model": "gpt-4",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info"
                }
            },
            {
                "type": "mcp",
                "name": "mcp_tool",
                "description": "MCP tool without server_url"
                # Missing server_url
            }
        ],
        "messages": [{"role": "user", "content": "Test message"}]
    }
    
    # Call the hook
    result = await hook.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="completion"
    )
    
    # Verify the result
    assert result is not None
    assert "tools" in result
    assert len(result["tools"]) == 2
    
    # Function tool should be unchanged
    function_tool = result["tools"][0]
    assert function_tool["type"] == "function"
    assert function_tool["function"]["name"] == "get_weather"
    
    # MCP tool should have server_url normalized
    mcp_tool = result["tools"][1]
    assert mcp_tool["type"] == "mcp"
    assert mcp_tool["server_url"] == "litellm_proxy"
    assert mcp_tool["name"] == "mcp_tool"


@pytest.mark.asyncio
async def test_mcp_tool_preprocessor_no_tools():
    """Test that requests without tools are handled correctly"""
    
    # Initialize the hook
    hook = _PROXY_MCPToolPreprocessor()
    
    # Mock dependencies
    user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
    cache = MagicMock(spec=DualCache)
    
    # Test data without tools
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Test message"}]
    }
    
    # Call the hook
    result = await hook.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="completion"
    )
    
    # Verify the result
    assert result == data  # Should return original data unchanged


@pytest.mark.asyncio
async def test_mcp_tool_preprocessor_empty_server_url():
    """Test that MCP tools with empty server_url get normalized"""
    
    # Initialize the hook
    hook = _PROXY_MCPToolPreprocessor()
    
    # Mock dependencies
    user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)
    cache = MagicMock(spec=DualCache)
    
    # Test data with MCP tool having empty server_url
    data = {
        "model": "gpt-4",
        "tools": [
            {
                "type": "mcp",
                "name": "test_tool",
                "description": "A test tool",
                "server_url": "",  # Empty string
                "require_approval": "never"
            }
        ],
        "messages": [{"role": "user", "content": "Test message"}]
    }
    
    # Call the hook
    result = await hook.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="completion"
    )
    
    # Verify the result
    assert result is not None
    assert "tools" in result
    assert len(result["tools"]) == 1
    
    tool = result["tools"][0]
    assert tool["type"] == "mcp"
    assert tool["server_url"] == "litellm_proxy"  # Should be normalized from empty string
    assert tool["name"] == "test_tool"
    assert tool["require_approval"] == "never"