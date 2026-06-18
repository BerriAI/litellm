"""
Tests for normalize_system_role_in_anthropic_messages.

Issue #30705: Anthropic /v1/messages rejects requests that include
role="system" inside messages[].  OpenAI-style clients routinely send the
system prompt as the first message; the helper promotes those entries to the
top-level `system` parameter so the request stays valid.
"""

import pytest
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.llms.anthropic.common_utils import (
    normalize_system_role_in_anthropic_messages,
)


class TestNormalizeSystemRole:
    def test_no_system_messages_returns_input_unchanged(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        new_messages, new_system = normalize_system_role_in_anthropic_messages(
            messages, system=None
        )
        assert new_messages == messages
        assert new_system is None

    def test_promotes_string_system_message_to_top_level(self):
        messages = [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "hi"},
        ]
        new_messages, new_system = normalize_system_role_in_anthropic_messages(
            messages, system=None
        )
        assert new_system == [
            {"type": "text", "text": "you are a helpful assistant"}
        ]
        assert new_messages == [{"role": "user", "content": "hi"}]

    def test_promotes_content_block_system_message(self):
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "be concise",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": "hi"},
        ]
        new_messages, new_system = normalize_system_role_in_anthropic_messages(
            messages, system=None
        )
        assert new_system == [
            {
                "type": "text",
                "text": "be concise",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        assert new_messages == [{"role": "user", "content": "hi"}]

    def test_merges_with_existing_string_system(self):
        messages = [
            {"role": "system", "content": "second instruction"},
            {"role": "user", "content": "hi"},
        ]
        new_messages, new_system = normalize_system_role_in_anthropic_messages(
            messages, system="first instruction"
        )
        assert new_system == [
            {"type": "text", "text": "first instruction"},
            {"type": "text", "text": "second instruction"},
        ]
        assert new_messages == [{"role": "user", "content": "hi"}]

    def test_merges_with_existing_list_system(self):
        messages = [
            {"role": "system", "content": "second instruction"},
            {"role": "user", "content": "hi"},
        ]
        new_messages, new_system = normalize_system_role_in_anthropic_messages(
            messages,
            system=[{"type": "text", "text": "first instruction"}],
        )
        assert new_system == [
            {"type": "text", "text": "first instruction"},
            {"type": "text", "text": "second instruction"},
        ]
        assert new_messages == [{"role": "user", "content": "hi"}]

    def test_promotes_multiple_system_messages_in_order(self):
        messages = [
            {"role": "system", "content": "first"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "second"},
            {"role": "assistant", "content": "ok"},
        ]
        new_messages, new_system = normalize_system_role_in_anthropic_messages(
            messages, system=None
        )
        assert new_system == [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert new_messages == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]

    def test_does_not_mutate_input_messages(self):
        original = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]
        # Take a deep copy snapshot
        snapshot = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]
        normalize_system_role_in_anthropic_messages(original, system=None)
        assert original == snapshot

    def test_empty_string_system_message_skipped(self):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "hi"},
        ]
        new_messages, new_system = normalize_system_role_in_anthropic_messages(
            messages, system=None
        )
        assert new_messages == [{"role": "user", "content": "hi"}]
        # Empty system messages are dropped without producing any top-level
        # system blocks; the caller's original (None) is preserved.
        assert new_system is None

    def test_empty_messages_list(self):
        new_messages, new_system = normalize_system_role_in_anthropic_messages(
            [], system="existing"
        )
        assert new_messages == []
        assert new_system == "existing"

    def test_mixed_string_and_block_system_in_messages(self):
        messages = [
            {"role": "system", "content": "first"},
            {"role": "user", "content": "hi"},
            {
                "role": "system",
                "content": [{"type": "text", "text": "second"}],
            },
        ]
        new_messages, new_system = normalize_system_role_in_anthropic_messages(
            messages, system=None
        )
        assert new_system == [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert new_messages == [{"role": "user", "content": "hi"}]

    def test_invalid_message_shape_passed_through(self):
        messages = [
            "not a dict",
            {"role": "system", "content": "promoted"},
            None,
            {"role": "user", "content": "hi"},
        ]
        new_messages, new_system = normalize_system_role_in_anthropic_messages(
            messages, system=None
        )
        assert new_system == [{"type": "text", "text": "promoted"}]
        # Non-dict and None entries are passed through unchanged.
        assert new_messages[0] == "not a dict"
        assert new_messages[1] is None
        assert new_messages[2] == {"role": "user", "content": "hi"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))