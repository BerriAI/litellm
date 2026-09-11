"""
Unit tests for litellm.compression.compress helpers.

get_protected_indices is the shared policy for which messages a compressor may
never rewrite. It is consumed by compress() and by the Headroom guardrail, so
the two agree on what "never compress this" means.
"""

from litellm.compression.compress import get_protected_indices


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


def test_mid_history_cache_control_part_is_protected():
    # A large cached tool result from a few turns back, not the last user or
    # last assistant row -- exactly the row a provider prompt-cache pins to
    # exact bytes. Rewriting it (even leaving the marker on) changes those
    # bytes and turns the next request's cache read into a cache write.
    messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a large cached tool result"},
            ],
        },
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "live instruction"},
    ]
    messages[2]["content"][0]["cache_control"] = {"type": "ephemeral"}

    # index 3 = last assistant, index 4 = last user (both protected by role
    # regardless), index 2 = the cache_control-marked row itself.
    assert sorted(get_protected_indices(messages)) == [2, 3, 4]


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
