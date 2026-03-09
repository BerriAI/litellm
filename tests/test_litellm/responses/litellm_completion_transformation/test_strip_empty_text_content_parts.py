"""
Unit tests for _strip_empty_text_content_parts helper in handler.py.

The helper filters {"type": "text", "text": ""} parts from message content lists
before the request reaches strict OpenAI-compatible endpoints that reject them
(e.g. Kimi-K2.5, gpt-oss-120b on Azure AI).
"""

from litellm.responses.litellm_completion_transformation.handler import (
    _strip_empty_text_content_parts,
)


def test_strips_empty_text_part_adjacent_to_tool_calls():
    """
    The primary real-world case: an assistant message with a tool_calls array and
    an empty text content part produced by the Responses API -> chat/completions
    transformation.
    """
    request = {
        "model": "azure_ai/kimi-k2.5",
        "messages": [
            {"role": "user", "content": "Use a tool"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},
                    {
                        "type": "tool_calls",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "my_tool", "arguments": "{}"},
                            }
                        ],
                    },
                ],
            },
        ],
    }

    result = _strip_empty_text_content_parts(request)

    assistant_content = result["messages"][1]["content"]
    assert isinstance(assistant_content, list)
    # The empty text part must be gone
    text_parts = [
        p for p in assistant_content if isinstance(p, dict) and p.get("type") == "text"
    ]
    assert text_parts == [], f"Expected no text parts, got: {text_parts}"
    # The non-text part must still be present
    assert len(assistant_content) == 1
    assert assistant_content[0]["type"] == "tool_calls"


def test_preserves_nonempty_text_parts():
    """Non-empty text parts must not be removed."""
    request = {
        "model": "azure_ai/kimi-k2.5",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},
                    {"type": "text", "text": "Here is the result:"},
                    {"type": "text", "text": ""},
                ],
            }
        ],
    }

    result = _strip_empty_text_content_parts(request)

    content = result["messages"][0]["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0] == {"type": "text", "text": "Here is the result:"}


def test_falls_back_to_empty_string_when_all_parts_are_empty():
    """
    If filtering removes every part, content must fall back to "" rather than
    leaving an empty list (which also breaks strict endpoints).
    """
    request = {
        "model": "azure_ai/kimi-k2.5",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},
                    {"type": "text", "text": ""},
                ],
            }
        ],
    }

    result = _strip_empty_text_content_parts(request)

    assert result["messages"][0]["content"] == ""


def test_no_copy_when_no_changes_needed():
    """
    When there are no empty text parts, the original dict must be returned
    unchanged (not a copy) to avoid unnecessary allocations.
    """
    request = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ],
    }

    result = _strip_empty_text_content_parts(request)

    assert result is request


def test_non_text_parts_preserved():
    """image_url and other non-text parts must survive the filter."""
    request = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ""},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.png"},
                    },
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ],
    }

    result = _strip_empty_text_content_parts(request)

    content = result["messages"][0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "image_url"
    assert content[1] == {"type": "text", "text": "What is this?"}


def test_string_content_untouched():
    """Messages whose content is already a string must not be modified."""
    request = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello"},
        ],
    }

    result = _strip_empty_text_content_parts(request)

    assert result is request
    assert result["messages"][0]["content"] == "Hello"
