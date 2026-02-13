"""
Tests for preserving cache_control in Responses API content transformation.

Anthropic prompt caching requires a cache_control directive on content blocks.
When transforming Responses API input items to Chat Completion messages, the
cache_control field must be preserved on text content items.
"""

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def test_cache_control_preserved_on_text_content():
    """cache_control on a text content item should be preserved in the
    transformed Chat Completion content block."""
    content = [
        {
            "type": "input_text",
            "text": "You are a helpful assistant.",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    result = LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(
        content=content
    )

    assert len(result) == 1
    assert result[0]["text"] == "You are a helpful assistant."
    assert result[0]["cache_control"] == {"type": "ephemeral"}


def test_cache_control_absent_when_not_provided():
    """When cache_control is not on the input item, it should not appear
    in the output."""
    content = [
        {
            "type": "input_text",
            "text": "Hello",
        }
    ]

    result = LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(
        content=content
    )

    assert len(result) == 1
    assert result[0]["text"] == "Hello"
    assert "cache_control" not in result[0]


def test_cache_control_only_on_tagged_items():
    """In a mixed list, only the item with cache_control should have it."""
    content = [
        {"type": "input_text", "text": "first"},
        {
            "type": "input_text",
            "text": "second",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "input_text", "text": "third"},
    ]

    result = LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(
        content=content
    )

    assert len(result) == 3
    assert "cache_control" not in result[0]
    assert result[1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in result[2]


def test_cache_control_end_to_end_via_input_items():
    """cache_control should survive the full input-to-messages transformation."""
    input_items = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Cached system context",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        },
    ]

    messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=input_items,
        responses_api_request={},
    )

    # Find the user message
    user_msgs = [
        m for m in messages
        if (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) == "user"
    ]
    assert len(user_msgs) == 1

    content = user_msgs[0].get("content") if isinstance(user_msgs[0], dict) else getattr(user_msgs[0], "content", None)
    assert isinstance(content, list)
    text_blocks = [b for b in content if (b.get("type") if isinstance(b, dict) else getattr(b, "type", "")) == "text"]
    assert len(text_blocks) == 1
    assert text_blocks[0]["cache_control"] == {"type": "ephemeral"}
