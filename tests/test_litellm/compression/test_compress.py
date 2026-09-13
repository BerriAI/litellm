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
