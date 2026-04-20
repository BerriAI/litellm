"""
Unit tests for ``litellm/proxy/_experimental/mcp_server/utils.py`` helpers.

Focuses on ``get_tool_name`` which normalises tool payloads across the three
shapes the proxy sees in practice.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from mcp.types import Tool as MCPTool

from litellm.proxy._experimental.mcp_server.utils import get_tool_name


def test_get_tool_name_openai_wrapper():
    tool = {
        "type": "function",
        "function": {"name": "foo", "description": "bar"},
    }
    assert get_tool_name(tool) == "foo"


def test_get_tool_name_flat_dict():
    tool = {"name": "foo", "description": "bar"}
    assert get_tool_name(tool) == "foo"


def test_get_tool_name_mcptool_object():
    tool = MCPTool(name="foo", description="bar", inputSchema={"type": "object"})
    assert get_tool_name(tool) == "foo"


def test_get_tool_name_missing_returns_empty_string():
    assert get_tool_name({}) == ""
    assert get_tool_name({"function": {"description": "no name"}}) == ""

    class _Nameless:
        pass

    assert get_tool_name(_Nameless()) == ""


def test_get_tool_name_openai_wrapper_missing_name_falls_back_to_flat():
    # Wrapper present but no name -> fall through to flat-dict lookup.
    tool = {
        "type": "function",
        "function": {"description": "nope"},
        "name": "outer",
    }
    assert get_tool_name(tool) == "outer"
