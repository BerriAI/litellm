"""
Regression tests for https://github.com/BerriAI/litellm/issues/47046

Vertex reads cachedContent as a prefix of contents. When the cache_control
block is not a prefix of the conversation, the conversation was reordered
(earlier messages re-sent after the cached block) and a system message
outside the cached block was silently dropped under modify_params.
"""

from litellm.llms.vertex_ai.context_caching.transformation import (
    separate_cached_messages,
    transform_openai_messages_to_gemini_context_caching,
)

CACHED = {"cache_control": {"type": "ephemeral"}}


def _system(text="you are helpful"):
    return {"role": "system", "content": text}


def _user(text, cached=False):
    msg = {"role": "user", "content": text}
    if cached:
        msg["cache_control"] = CACHED["cache_control"]
    return msg


def _assistant(text, cached=False):
    msg = {"role": "assistant", "content": text}
    if cached:
        msg["cache_control"] = CACHED["cache_control"]
    return msg


def test_non_prefix_block_sends_conversation_uncached():
    # Exact repro from the issue: cache_control on a message that is not the
    # start of the conversation.
    messages = [
        _system(),
        _user("What's the weather in Tokyo today?"),
        _assistant("Sunny."),
        _user("And Nagoya tomorrow?", cached=True),
    ]
    cached, non_cached = separate_cached_messages(messages=messages)
    assert cached == []
    assert non_cached == messages


def test_non_system_prefix_sends_conversation_uncached():
    messages = [
        _user("Q1"),
        _assistant("A1"),
        _user("Q2", cached=True),
    ]
    cached, non_cached = separate_cached_messages(messages=messages)
    assert cached == []
    assert non_cached == messages


def test_leading_system_message_folded_into_cached_block():
    # A system message before the block can be preserved inside the cache's
    # system_instruction, so caching still applies with nothing reordered.
    messages = [
        _system(),
        _user("Q1", cached=True),
        _assistant("A1", cached=True),
        _user("Q2"),
    ]
    cached, non_cached = separate_cached_messages(messages=messages)
    assert cached == messages[:3]
    assert non_cached == [messages[3]]


def test_true_prefix_block_still_caches():
    messages = [
        _user("Q1", cached=True),
        _assistant("A1", cached=True),
        _user("Q2"),
    ]
    cached, non_cached = separate_cached_messages(messages=messages)
    assert cached == messages[:2]
    assert non_cached == [messages[2]]


def test_trailing_system_message_sends_conversation_uncached():
    # A system message after the cached block would be extracted and dropped
    # alongside cachedContent, so caching must be skipped.
    messages = [
        _user("Q1", cached=True),
        _system("late instruction"),
        _user("Q2"),
    ]
    cached, non_cached = separate_cached_messages(messages=messages)
    assert cached == []
    assert non_cached == messages


def test_no_cache_control_unchanged():
    messages = [_system(), _user("Q1"), _assistant("A1")]
    cached, non_cached = separate_cached_messages(messages=messages)
    assert cached == []
    assert non_cached == messages


def test_cache_body_preserves_folded_system_message():
    # The folded system message must land in the cache's system_instruction,
    # not in contents and not dropped.
    cached, _ = separate_cached_messages(
        messages=[_system(), _user("Q1", cached=True)]
    )
    body = transform_openai_messages_to_gemini_context_caching(
        model="gemini-2.0-flash",
        messages=cached,
        custom_llm_provider="gemini",
        cache_key="test-key",
        vertex_project=None,
        vertex_location=None,
    )
    # Preserved as system_instruction when the model supports it, otherwise
    # merged into contents - but never dropped.
    import json as _json

    assert "you are helpful" in _json.dumps(body)
