"""
Tests for MCP metadata preservation.

This module tests that tool metadata is preserved when creating prefixed tools,
which is critical for ChatGPT UI widget rendering.
"""


import pytest

# Add the parent directory to the path so we can import litellm

from mcp.types import Tool as MCPTool

from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy._types import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


class TestMCPMetadataPreservation:
    """Test that metadata is preserved when creating prefixed tools"""

    def test_create_prefixed_tools_preserves_metadata(self):
        """Test that _create_prefixed_tools preserves metadata and _meta fields"""
        manager = MCPServerManager()

        # Create a mock server
        mock_server = MCPServer(
            server_id="test-server-1",
            name="test_server",
            alias="test",
            server_name="Test Server",
            url="https://test-server.com/mcp",
            transport=MCPTransport.http,
        )

        # Create a tool with metadata
        tool_with_metadata = MCPTool(
            name="hello_widget",
            description="Display a greeting widget",
            input_schema={"type": "object", "properties": {}},
            meta={
                "openai/outputTemplate": "ui://widget/hello.html",
                "openai/widgetDescription": "A greeting widget",
                "openai/toolInvocation/invoking": "Preparing greeting...",
            },
        )

        # Create prefixed tools
        prefixed_tools = manager._create_prefixed_tools(
            [tool_with_metadata], mock_server, add_prefix=True
        )

        # Verify
        assert len(prefixed_tools) == 1
        prefixed_tool = prefixed_tools[0]

        # Check that name is prefixed
        assert prefixed_tool.name == "test-hello_widget"

        # Check that _meta (the SDK `meta` field) is preserved
        assert prefixed_tool.meta == {
            "openai/outputTemplate": "ui://widget/hello.html",
            "openai/widgetDescription": "A greeting widget",
            "openai/toolInvocation/invoking": "Preparing greeting...",
        }

        # Check that other fields are preserved
        assert prefixed_tool.description == "Display a greeting widget"
        assert prefixed_tool.input_schema== {"type": "object", "properties": {}}


if __name__ == "__main__":
    pytest.main([__file__])
