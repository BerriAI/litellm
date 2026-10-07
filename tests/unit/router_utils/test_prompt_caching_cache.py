from typing import Final

from litellm.router_utils.prompt_caching_cache import PromptCachingCache


def test_extract_cacheable_prefix_with_string_content_and_message_level_cache_control():
    """
    Test that extract_cacheable_prefix correctly handles messages where:
    - content is a string (not a list of content blocks)
    - cache_control is a sibling key at the message level

    This is a valid message format per LiteLLM's ChatCompletionUserMessage type:
    {"role": "user", "content": "...", "cache_control": {"type": "ephemeral"}}

    Regression test for issue #19228.
    """
    messages_string_content: Final = [
        {"role": "system", "content": "You are a helpful assistant"},
        {
            "role": "user",
            "content": "This is a large message that should be cached",
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        },
    ]

    result: Final = PromptCachingCache.extract_cacheable_prefix(messages_string_content)

    assert len(result) == 2, f"Expected 2 messages, got {len(result)}"
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "This is a large message that should be cached"
    assert result[1].get("cache_control") == {"type": "ephemeral", "ttl": "5m"}


def test_extract_cacheable_prefix_with_string_content_no_cache_control():
    """
    Test that extract_cacheable_prefix returns empty list when:
    - content is a string
    - no cache_control is present
    """
    messages_no_cache: Final = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]

    result: Final = PromptCachingCache.extract_cacheable_prefix(messages_no_cache)

    assert len(result) == 0, f"Expected 0 messages, got {len(result)}"


def test_extract_cacheable_prefix_mixed_string_and_list_content():
    """
    Test that extract_cacheable_prefix handles messages with a mix of:
    - String content with message-level cache_control
    - List content with block-level cache_control

    The last cache_control (regardless of format) should determine the cacheable prefix.
    """
    messages_mixed: Final = [
        {"role": "system", "content": "You are a helpful assistant"},
        {
            "role": "user",
            "content": "First cached message",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Second cached message in list format",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "user", "content": "This should not be in the prefix"},
    ]

    result: Final = PromptCachingCache.extract_cacheable_prefix(messages_mixed)

    assert len(result) == 3, f"Expected 3 messages, got {len(result)}"
    assert result[0]["role"] == "system"
    assert result[1]["content"] == "First cached message"
    assert isinstance(result[2]["content"], list)
