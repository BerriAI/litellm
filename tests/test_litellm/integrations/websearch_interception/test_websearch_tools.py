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
    ],
)
def test_openai_function_litellm_web_search_is_detected(tool: dict[str, Any]):
    """Recognize the LiteLLM standard web_search function name."""
    assert is_web_search_tool(tool) is True
    assert is_web_search_tool_chat_completion(tool) is True


def test_conventional_web_search_requires_opt_in():
    """Name-only web_search stays a user tool unless explicitly opted in."""
    tool = {"type": "function", "function": {"name": "web_search"}}
    assert is_web_search_tool(tool) is False
    assert is_web_search_tool_chat_completion(tool) is False
    assert is_web_search_tool(tool, recognize_conventional_name=True) is True
    assert is_web_search_tool_chat_completion(tool, recognize_conventional_name=True) is True


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
