import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._experimental.mcp_server.tool_registry import MCPToolRegistry


# Test handler function
def example_handler(input_data):
    return {"result": input_data}


def test_register_and_get_tool():
    registry = MCPToolRegistry()

    # Test registering a tool
    registry.register_tool(
        name="test_tool",
        description="A test tool",
        input_schema={"type": "object", "properties": {"test": {"type": "string"}}},
        handler=example_handler,
    )

    # Test getting the registered tool
    tool = registry.get_tool("test_tool")
    assert tool is not None
    assert tool.name == "test_tool"
    assert tool.description == "A test tool"
    assert callable(tool.handler)

    # Test getting non-existent tool
    assert registry.get_tool("non_existent") is None


def test_list_tools():
    registry = MCPToolRegistry()

    # Register multiple tools
    registry.register_tool(
        name="tool1", description="Tool 1", input_schema={}, handler=example_handler
    )
    registry.register_tool(
        name="tool2", description="Tool 2", input_schema={}, handler=example_handler
    )

    # Test listing tools
    tools = registry.list_tools()
    assert len(tools) == 2
    assert {tool.name for tool in tools} == {"tool1", "tool2"}


def test_load_tools_from_config():
    registry = MCPToolRegistry()

    # Test valid config
    valid_config = [
        {
            "name": "config_tool",
            "description": "A tool from config",
            "input_schema": {"type": "object"},
            "handler": "test_tool_registry.example_handler",
        }
    ]

    registry.load_tools_from_config(valid_config)
    assert "config_tool" in registry.tools
    assert registry.tools["config_tool"].name == "config_tool"
    assert registry.tools["config_tool"].description == "A tool from config"
    assert callable(registry.tools["config_tool"].handler)


def test_tool_execution():
    registry = MCPToolRegistry()

    # Register a tool
    registry.register_tool(
        name="echo",
        description="Echo the input",
        input_schema={"type": "object"},
        handler=example_handler,
    )

    # Get and execute the tool
    tool = registry.get_tool("echo")
    assert tool is not None

    test_input = {"message": "hello"}
    result = tool.handler(test_input)
    assert result == {"result": test_input}
