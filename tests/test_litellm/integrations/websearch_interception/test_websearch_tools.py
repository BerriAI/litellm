"""Unit tests for web search tool shape detection."""

from typing import Any, Dict

import pytest

from litellm.integrations.websearch_interception.tools import (
    is_web_search_tool,
    is_web_search_tool_chat_completion,
)


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "function", "function": {"name": "litellm_web_search"}},
        {"type": "function", "function": {"name": "web_search"}},
    ],
)
def test_recognized_function_tool_shapes_are_detected(tool: Dict[str, Any]):
    """Regression: {"type": "function", "function": {"name": "web_search"}} (the shape sent by
    Anthropic-style clients over Chat Completions) must be recognized, not just the LiteLLM
    standard name. Before the fix this fell through to provider tool mapping and raised
    "Missing required parameter: name".
    """
    assert is_web_search_tool(tool) is True
    assert is_web_search_tool_chat_completion(tool) is True


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "function", "function": {"name": "web_search_helper"}},
        {"type": "function", "function": {"name": "search"}},
        {"type": "function", "function": None},
        {"type": "function"},
    ],
)
def test_unrelated_or_malformed_function_tools_are_not_detected(tool: Dict[str, Any]):
    assert is_web_search_tool(tool) is False
    assert is_web_search_tool_chat_completion(tool) is False


def test_user_defined_web_search_function_with_schema_is_not_hijacked():
    """A user's own tool may happen to be named "web_search" but carry a real schema;
    only the bare {"name": "web_search"} shape is the conventional web-search marker.
    """
    tool = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search a private index",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    assert is_web_search_tool(tool) is False
    assert is_web_search_tool_chat_completion(tool) is False
