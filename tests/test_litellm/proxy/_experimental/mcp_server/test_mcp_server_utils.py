"""
Unit tests for ``litellm/proxy/_experimental/mcp_server/utils.py`` helpers.

Focuses on ``get_tool_name_and_description`` which normalises tool payloads
across the three shapes the proxy sees in practice.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from mcp.types import Tool as MCPTool

from litellm.proxy._experimental.mcp_server.utils import (
    get_tool_name_and_description,
)


def test_pair_openai_wrapper_returns_function_block_fields():
    tool = {
        "type": "function",
        "function": {"name": "foo", "description": "bar"},
    }
    assert get_tool_name_and_description(tool) == ("foo", "bar")


def test_pair_flat_dict_returns_top_level_fields():
    tool = {"name": "foo", "description": "bar"}
    assert get_tool_name_and_description(tool) == ("foo", "bar")


def test_pair_mcptool_object_returns_attributes():
    tool = MCPTool(name="foo", description="bar", inputSchema={"type": "object"})
    assert get_tool_name_and_description(tool) == ("foo", "bar")


def test_pair_missing_fields_return_empty_strings():
    assert get_tool_name_and_description({}) == ("", "")

    class _Nameless:
        pass

    assert get_tool_name_and_description(_Nameless()) == ("", "")


def test_pair_description_falls_back_to_name():
    """Matches the historical ``_extract_tool_info`` fallback so the
    semantic router never embeds an empty string."""
    # Dict with no description.
    assert get_tool_name_and_description({"name": "foo"}) == ("foo", "foo")
    # Dict with empty description.
    assert get_tool_name_and_description({"name": "foo", "description": ""}) == (
        "foo",
        "foo",
    )
    # MCPTool with no description.
    tool = MCPTool(name="foo", description=None, inputSchema={"type": "object"})
    assert get_tool_name_and_description(tool) == ("foo", "foo")


def test_pair_openai_wrapper_missing_name_falls_back_to_outer_dict():
    """Malformed wrapper where ``function`` block has no name - fall
    through to the outer dict so both name and description are read
    together from the same source."""
    tool = {
        "type": "function",
        "function": {"description": "nope"},
        "name": "outer",
        "description": "outer-desc",
    }
    assert get_tool_name_and_description(tool) == ("outer", "outer-desc")
