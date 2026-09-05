"""Unit tests for web search tool shape detection."""

from typing import Any

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
def test_openai_function_web_search_shapes_are_detected(tool: dict[str, Any]):
    """Recognize both LiteLLM and conventional web_search function names."""
    assert is_web_search_tool(tool) is True
    assert is_web_search_tool_chat_completion(tool) is True


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "function", "function": {"name": "web_search_helper"}},
        {"type": "function", "function": {"name": "search"}},
        {"type": "function", "function": None},
    ],
)
def test_unrelated_openai_function_tools_are_not_detected(tool: dict[str, Any]):
    """Do not classify similarly named user tools as web search."""
    assert is_web_search_tool(tool) is False
    assert is_web_search_tool_chat_completion(tool) is False


@pytest.mark.parametrize(
    "tool",
    [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search a private index",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ],
)
def test_user_defined_web_search_functions_are_not_detected(tool: dict[str, Any]):
    """Preserve user-defined schemas even when their name is web_search."""
    assert is_web_search_tool(tool) is False
    assert is_web_search_tool_chat_completion(tool) is False
