"""
Unit tests for litellm.compression.compress helpers.

get_protected_indices is the shared policy for which messages a compressor may
never rewrite. It is consumed by compress() and by the Headroom guardrail, so
the two agree on what "never compress this" means.
"""

from litellm.compression.compress import compress, get_protected_indices
from litellm.types.utils import CallTypes


def test_protects_system_last_user_and_last_assistant():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "newer question"},
        {"role": "assistant", "content": "newer answer"},
        {"role": "user", "content": "live instruction"},
    ]

    assert sorted(get_protected_indices(messages)) == [0, 4, 5]


def test_history_is_not_protected():
    messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "tool", "tool_call_id": "t1", "content": "old tool output"},
        {"role": "user", "content": "live instruction"},
    ]

    protected = sorted(get_protected_indices(messages))

    assert protected == [1, 3]
    # The tool row and the older user turn stay compressible; protection that
    # covered everything would make compression a no-op.
    assert 0 not in protected
    assert 2 not in protected


def test_every_system_row_is_protected():
    messages = [
        {"role": "system", "content": "first"},
        {"role": "user", "content": "q"},
        {"role": "system", "content": "second, injected mid conversation"},
        {"role": "user", "content": "live"},
    ]

    assert sorted(get_protected_indices(messages)) == [0, 2, 3]


def test_no_user_or_assistant_rows():
    assert sorted(get_protected_indices([{"role": "system", "content": "sys"}])) == [0]
    assert get_protected_indices([]) == ()


def test_rows_before_last_cache_control_breakpoint_are_protected():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old question"},
        {
            "role": "assistant",
            "content": "old answer",
            "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "large file body"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "cached turn", "cache_control": {"type": "ephemeral"}}],
        },
        {
            "role": "assistant",
            "content": "ack",
            "tool_calls": [{"id": "t2", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "t2", "content": "later tool output"},
        {"role": "user", "content": "live instruction"},
    ]

    protected = sorted(get_protected_indices(messages))

    assert protected == [0, 1, 2, 3, 4, 5, 7]
    assert 6 not in protected


def test_cache_control_directly_on_message_protects_prefix():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "tool", "tool_call_id": "before", "content": "large file body"},
        {"role": "user", "content": "old question"},
        {
            "role": "tool",
            "tool_call_id": "marked",
            "content": "cached tool",
            "cache_control": {"type": "ephemeral"},
        },
        {"role": "tool", "tool_call_id": "after", "content": "later tool output"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "live instruction"},
    ]

    protected = sorted(get_protected_indices(messages))

    assert 1 in protected
    assert 3 in protected
    assert 4 not in protected


def test_no_cache_control_leaves_history_compressible():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "tool", "tool_call_id": "t1", "content": "large file body"},
        {"role": "user", "content": "live instruction"},
    ]

    assert sorted(get_protected_indices(messages)) == [0, 2, 4]


def test_non_mapping_content_parts_are_not_cache_control():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": ["not", "a", "dict"]},
        {"role": "assistant", "content": "old answer"},
        {"role": "tool", "tool_call_id": "t1", "content": "plain string"},
        {"role": "user", "content": "live instruction"},
    ]

    protected = sorted(get_protected_indices(messages))

    assert protected == [0, 2, 4]
    assert 1 not in protected
    assert 3 not in protected


def test_mid_history_cache_control_part_is_protected():
    messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a large cached tool result", "cache_control": {"type": "ephemeral"}},
            ],
        },
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "live instruction"},
    ]

    assert sorted(get_protected_indices(messages)) == [0, 1, 2, 3, 4]


def test_cache_control_directly_on_message_is_protected():
    messages = [
        {"role": "user", "content": "old question", "cache_control": {"type": "ephemeral"}},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "live instruction"},
    ]

    assert sorted(get_protected_indices(messages)) == [0, 1, 2]


def test_cache_control_protection_does_not_duplicate_already_protected_rows():
    # The last user row is already protected by role; marking it too must not
    # produce a duplicate index.
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "live", "cache_control": {"type": "ephemeral"}},
    ]

    protected = get_protected_indices(messages)

    assert sorted(protected) == [0, 1]
    assert len(protected) == len(set(protected))


def test_content_that_is_not_a_list_of_mappings_is_not_treated_as_cache_control():
    # Defensive: a plain string content, or a list of non-dict items, must not
    # raise or be misread as carrying a breakpoint.
    messages = [
        {"role": "assistant", "content": "plain string content"},
        {"role": "user", "content": ["not", "a", "dict", "list"]},
        {"role": "user", "content": "live instruction"},
    ]

    assert sorted(get_protected_indices(messages)) == [0, 2]


def test_compress_keeps_part_level_cache_control_row_verbatim():
    stale_log = {"role": "user", "content": [{"type": "text", "text": "stale log line " * 2000}]}
    pinned = {
        "role": "user",
        "content": [
            {"type": "text", "text": "cached tool result " * 2000, "cache_control": {"type": "ephemeral"}},
        ],
    }
    messages = [
        pinned,
        {"role": "assistant", "content": "old answer"},
        stale_log,
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "live instruction"},
    ]

    result = compress(
        messages,
        model="gpt-4o",
        call_type=CallTypes.anthropic_messages,
        compression_trigger=1000,
        compression_target=500,
    )

    assert len(result["messages"]) == len(messages)
    assert result["messages"][0] == pinned
    assert result["messages"][2] != stale_log
    assert len(result["cache"]) >= 1


def test_trailing_cache_control_alone_does_not_protect_whole_conversation():
    """Claude Code marks the live turn every request. That trailing write marker
    must not pin the entire history as a cached prefix (#42939)."""
    messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "tool", "tool_call_id": "t1", "content": "large file body"},
        {"role": "user", "content": "live instruction", "cache_control": {"type": "ephemeral"}},
    ]

    protected = sorted(get_protected_indices(messages))

    assert protected == [1, 3]
    assert 0 not in protected
    assert 2 not in protected


def test_trailing_cache_control_defers_to_earlier_breakpoint():
    """System (or mid-history) breakpoint stays the prefix end; the trailing
    live-turn marker is ignored so mid-history after that earlier breakpoint
    remains compressible (#42939)."""
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}}
            ],
        },
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "tool", "tool_call_id": "t1", "content": "large file body"},
        {"role": "user", "content": "live instruction", "cache_control": {"type": "ephemeral"}},
    ]

    protected = sorted(get_protected_indices(messages))

    # System (0) from the earlier breakpoint + last assistant (2) + last user (4).
    assert protected == [0, 2, 4]
    assert 1 not in protected
    assert 3 not in protected
