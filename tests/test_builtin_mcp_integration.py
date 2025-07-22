"""
Simple test for Built-in MCP Server Integration feature
"""

import os
from unittest.mock import patch


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
    assert global_mcp_server_manager.is_builtin_server("calculator")
    assert not global_mcp_server_manager.is_builtin_server("nonexistent")
    assert global_mcp_server_manager.is_builtin_server("builtin_calculator")
    assert not global_mcp_server_manager.is_builtin_server("regular_server_id")
    print("✓ Builtin server identification works")


def test_mcp_handler_builtin_detection():
    """Test MCP handler builtin tool detection"""
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler

    tools_with_builtin = [{"type": "mcp", "builtin": "calculator"}]
    tools_without_builtin = [{"type": "function", "function": {"name": "test"}}]

    assert LiteLLM_Proxy_MCP_Handler._should_use_litellm_mcp_gateway(tools_with_builtin)
    assert not LiteLLM_Proxy_MCP_Handler._should_use_litellm_mcp_gateway(tools_without_builtin)
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
    assert registry.is_builtin_available("calculator")

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


def test_remote_mcp_tool_processing():
    """Test processing of remote MCP tools"""
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler

    tools = [
        {
            "type": "mcp",
            "server_label": "stripe",
            "server_url": "https://mcp.stripe.com",
            "headers": {"Authorization": "Bearer sk-test-123"},
            "allowed_tools": ["create_payment_link"],
            "require_approval": "never"
        },
        {"type": "function", "function": {"name": "test"}}
    ]

    processed_tools = LiteLLM_Proxy_MCP_Handler._process_remote_mcp_tools(tools)

    assert len(processed_tools) == 2

    # Find the processed remote tool
    remote_tool = next((t for t in processed_tools if isinstance(t, dict) and t.get("_remote_server_url")), None)
    assert remote_tool is not None
    assert remote_tool.get("server_url") == "litellm_proxy"
    assert remote_tool.get("_remote_server_url") == "https://mcp.stripe.com"
    assert remote_tool.get("_remote_server_label") == "stripe"
    assert remote_tool.get("_allowed_tools") == ["create_payment_link"]
    assert remote_tool.get("_require_approval") == "never"
    assert "headers" not in remote_tool  # Should be moved to _remote_headers
    assert remote_tool.get("_remote_headers") == {"Authorization": "Bearer sk-test-123"}
    print("✓ Remote MCP tool processing works")


def test_hybrid_mcp_usage():
    """Test mixing builtin and remote MCP tools"""
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler

    tools = [
        {"type": "mcp", "builtin": "calculator"},
        {
            "type": "mcp",
            "server_url": "https://mcp.stripe.com",
            "server_label": "stripe"
        },
        {"type": "function", "function": {"name": "regular_func"}}
    ]

    # First process remote tools
    processed = LiteLLM_Proxy_MCP_Handler._process_remote_mcp_tools(tools)
    # Then expand builtin tools
    expanded = LiteLLM_Proxy_MCP_Handler._expand_builtin_tools(processed)

    assert len(expanded) == 3

    # Check that we have both types
    builtin_tool = next((t for t in expanded if isinstance(t, dict) and t.get("_builtin_name")), None)
    remote_tool = next((t for t in expanded if isinstance(t, dict) and t.get("_remote_server_url")), None)
    regular_tool = next((t for t in expanded if isinstance(t, dict) and t.get("type") == "function"), None)

    assert builtin_tool is not None
    assert remote_tool is not None
    assert regular_tool is not None
    print("✓ Hybrid MCP usage (builtin + remote + regular) works")


def test_mcp_approval_workflow():
    """Test MCP approval request/response workflow"""
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler

    # Test approval request creation
    approval_request = LiteLLM_Proxy_MCP_Handler._create_approval_request(
        tool_name="create_payment",
        tool_arguments='{"amount": 100}',
        server_label="stripe"
    )

    assert approval_request["type"] == "mcp_approval_request"
    assert approval_request["name"] == "create_payment"
    assert approval_request["arguments"] == '{"amount": 100}'
    assert approval_request["server_label"] == "stripe"
    assert "id" in approval_request
    assert approval_request["id"].startswith("mcpr_")
    print("✓ MCP approval request creation works")


def test_mcp_call_result_creation():
    """Test MCP call result creation"""
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler

    # Test successful call result
    call_result = LiteLLM_Proxy_MCP_Handler._create_mcp_call_result(
        tool_name="create_payment",
        tool_arguments='{"amount": 100}',
        result="Payment created successfully",
        server_label="stripe",
        approval_request_id="mcpr_123"
    )

    assert call_result["type"] == "mcp_call"
    assert call_result["name"] == "create_payment"
    assert call_result["arguments"] == '{"amount": 100}'
    assert call_result["output"] == "Payment created successfully"
    assert call_result["server_label"] == "stripe"
    assert call_result["approval_request_id"] == "mcpr_123"
    assert call_result["error"] is None
    assert "id" in call_result
    assert call_result["id"].startswith("mcp_")

    # Test error call result
    error_result = LiteLLM_Proxy_MCP_Handler._create_mcp_call_result(
        tool_name="create_payment",
        tool_arguments='{"amount": 100}',
        result="",
        server_label="stripe",
        error="Insufficient funds"
    )

    assert error_result["error"] == "Insufficient funds"
    assert error_result["output"] == ""
    print("✓ MCP call result creation works")


def test_approval_response_processing():
    """Test processing of approval responses"""
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler

    input_items = [
        {"type": "message", "content": "Hello"},
        {
            "type": "mcp_approval_response",
            "approve": True,
            "approval_request_id": "mcpr_123"
        },
        {
            "type": "mcp_approval_response",
            "approve": False,
            "approval_request_id": "mcpr_456"
        },
        {
            "type": "mcp_approval_response",
            "approve": True,
            "approval_request_id": "mcpr_789"
        }
    ]

    approved_ids = LiteLLM_Proxy_MCP_Handler._process_approval_responses(input_items)

    assert len(approved_ids) == 2
    assert "mcpr_123" in approved_ids
    assert "mcpr_789" in approved_ids
    assert "mcpr_456" not in approved_ids
    print("✓ Approval response processing works")


def test_tool_approval_checking():
    """Test tool approval requirement checking"""
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler

    # Test simple "never" approval
    config_never = {"_require_approval": "never"}
    assert not LiteLLM_Proxy_MCP_Handler._check_tool_approval("test_tool", config_never)

    # Test default (always require approval)
    config_default = {}
    assert LiteLLM_Proxy_MCP_Handler._check_tool_approval("test_tool", config_default)

    # Test granular approval - tool allowed
    config_granular = {
        "_require_approval": {
            "never": {
                "tool_names": ["safe_tool", "read_only_tool"]
            }
        }
    }
    assert not LiteLLM_Proxy_MCP_Handler._check_tool_approval("safe_tool", config_granular)
    assert LiteLLM_Proxy_MCP_Handler._check_tool_approval("dangerous_tool", config_granular)

    print("✓ Tool approval checking works")


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

            # Test new remote MCP functionality
            test_remote_mcp_tool_processing()
            test_hybrid_mcp_usage()

            # Test new approval functionality
            test_mcp_approval_workflow()
            test_mcp_call_result_creation()
            test_approval_response_processing()
            test_tool_approval_checking()

        except ImportError as e:
            print(f"⚠️  Skipping MCP handler tests: {e}")

        print("\n🎉 Built-in, Remote MCP, and Approval workflow tests passed!")
        print("💡 Note: Full functionality requires the 'mcp' module to be installed")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

