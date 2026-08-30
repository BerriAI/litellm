"""Unit tests for web search tool shape detection."""

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
def test_openai_function_web_search_shapes_are_detected(tool):
    """Recognize both LiteLLM and conventional web_search function names."""
    assert is_web_search_tool(tool) is True
    assert is_web_search_tool_chat_completion(tool) is True


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "function", "function": {"name": "web_search_helper"}},
        {"type": "function", "function": {"name": "search"}},
    ],
)
def test_unrelated_openai_function_tools_are_not_detected(tool):
    """Do not classify similarly named user tools as web search."""
    assert is_web_search_tool(tool) is False
    assert is_web_search_tool_chat_completion(tool) is False
