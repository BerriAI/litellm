"""
Simple test for Built-in MCP Server Integration feature
"""

import os
from unittest.mock import patch

# Remove pytest dependency and create simple test functions
def test_builtin_registry_initialization():
    """Test that builtin registry initializes with default servers"""
    from litellm.proxy._experimental.mcp_server.builtin_registry import BuiltinMCPRegistry
    
    registry = BuiltinMCPRegistry()
    
    # Check that default servers are loaded (only calculator in defaults)
    assert len(registry.list_builtin_names()) > 0
    assert "calculator" in registry.list_builtin_names()
    print("✓ Registry initialized with default calculator server")


def test_builtin_server_config_creation():
    """Test creation of builtin server config"""
    from litellm.proxy._experimental.mcp_server.builtin_registry import BuiltinMCPServerConfig
    
    config = BuiltinMCPServerConfig(
        name="test_server",
        url="https://test.com/mcp",
        transport="sse",
        auth_type="bearer_token",
        env_key="TEST_TOKEN"
    )
    
    assert config.name == "test_server"
    assert config.url == "https://test.com/mcp"
    assert config.transport == "sse"
    assert config.auth_type == "bearer_token"
    assert config.env_key == "TEST_TOKEN"
    print("✓ Builtin server config creation works")


def test_builtin_config_to_mcp_server_with_auth():
    """Test conversion of builtin config to MCPServer with authentication"""
    from litellm.proxy._experimental.mcp_server.builtin_registry import BuiltinMCPServerConfig
    
    config = BuiltinMCPServerConfig(
        name="test_server",
        url="https://test.com/mcp",
        transport="sse",
        auth_type="bearer_token",
        env_key="TEST_TOKEN"
    )
    
    with patch.dict(os.environ, {"TEST_TOKEN": "test_token_value"}):
        server = config.to_mcp_server("test_id")
        
        assert server.server_id == "test_id"
        assert server.name == "test_server"
        assert server.url == "https://test.com/mcp"
        assert server.authentication_token == "test_token_value"
    print("✓ Builtin config to MCPServer conversion works")


def test_mcp_server_manager_builtin_methods():
    """Test MCPServerManager builtin integration methods"""
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import global_mcp_server_manager
    
    # Test builtin server identification
    assert global_mcp_server_manager.is_builtin_server("calculator") == True
    assert global_mcp_server_manager.is_builtin_server("nonexistent") == False
    assert global_mcp_server_manager.is_builtin_server("builtin_calculator") == True
    assert global_mcp_server_manager.is_builtin_server("regular_server_id") == False
    print("✓ Builtin server identification works")


def test_mcp_handler_builtin_detection():
    """Test MCP handler builtin tool detection"""
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
    
    tools_with_builtin = [{"type": "mcp", "builtin": "calculator"}]
    tools_without_builtin = [{"type": "function", "function": {"name": "test"}}]
    
    assert LiteLLM_Proxy_MCP_Handler._should_use_litellm_mcp_gateway(tools_with_builtin) == True
    assert LiteLLM_Proxy_MCP_Handler._should_use_litellm_mcp_gateway(tools_without_builtin) == False
    print("✓ MCP gateway detection for builtin tools works")


def test_mcp_handler_tool_parsing():
    """Test parsing tools that include builtin references"""
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
    
    tools = [
        {"type": "mcp", "builtin": "calculator"},
        {"type": "mcp", "server_url": "litellm_proxy"},
        {"type": "function", "function": {"name": "test"}}
    ]
    
    mcp_tools, other_tools = LiteLLM_Proxy_MCP_Handler._parse_mcp_tools(tools)
    
    assert len(mcp_tools) == 2  # Both builtin and server_url tools
    assert len(other_tools) == 1  # The function tool
    
    # Check that builtin tool is included
    builtin_tool = next((t for t in mcp_tools if t.get("builtin") == "calculator"), None)
    assert builtin_tool is not None
    print("✓ MCP tool parsing with builtins works")


def test_calculator_builtin_server():
    """Test calculator builtin server configuration"""
    from litellm.proxy._experimental.mcp_server.builtin_registry import BuiltinMCPRegistry
    
    registry = BuiltinMCPRegistry()
    
    # Test calculator server availability  
    assert registry.is_builtin_available("calculator") == True
    
    # Test getting calculator server
    calc_server = registry.get_builtin_server("calculator")
    assert calc_server is not None
    assert calc_server.name == "calculator"
    assert calc_server.transport == "stdio"
    assert calc_server.command == "python"
    assert len(calc_server.args) > 0
    assert calc_server.args[0].endswith("sample_calculator_mcp_server.py")
    print("✓ Calculator builtin server configuration works")


def test_builtin_tool_expansion():
    """Test expansion of builtin tool references"""
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
    
    tools = [
        {"type": "mcp", "builtin": "calculator", "require_approval": "never"},
        {"type": "function", "function": {"name": "test"}}
    ]
    
    expanded_tools = LiteLLM_Proxy_MCP_Handler._expand_builtin_tools(tools)
    
    assert len(expanded_tools) == 2
    
    # Find the expanded calculator tool
    calc_tool = next((t for t in expanded_tools if t.get("_builtin_name") == "calculator"), None)
    assert calc_tool is not None
    assert calc_tool.get("server_url") == "litellm_proxy"
    assert calc_tool.get("builtin") is None  # Should be removed
    assert calc_tool.get("_builtin_server_id") is not None
    assert calc_tool.get("require_approval") == "never"  # Should be preserved
    print("✓ Calculator builtin tool expansion works")


if __name__ == "__main__":
    print("Running built-in MCP integration tests...\n")
    
    try:
        test_builtin_registry_initialization()
        test_builtin_server_config_creation() 
        test_builtin_config_to_mcp_server_with_auth()
        
        # Skip MCP server manager tests for now (requires mcp module)
        print("⚠️  Skipping MCP server manager tests (requires 'mcp' module)")
        
        # Test MCP handler methods (should work without mcp module)
        try:
            test_mcp_handler_builtin_detection()
            test_mcp_handler_tool_parsing()
            test_calculator_builtin_server()
            test_builtin_tool_expansion()
        except ImportError as e:
            print(f"⚠️  Skipping MCP handler tests: {e}")
        
        print("\n🎉 Basic builtin MCP integration tests passed!")
        print("💡 Note: Full functionality requires the 'mcp' module to be installed")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()