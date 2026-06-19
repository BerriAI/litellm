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


class TestSyncPathNormalization:
    """
    Verify the sync entry point ``anthropic_messages_handler`` also normalizes
    role=system messages, not just the async ``anthropic_messages`` wrapper.

    Regression for the Greptile review on #30719: the async wrapper applied
    the normalizer before dispatching to the sync handler, but callers using
    ``litellm.messages.create`` go directly through the sync handler and were
    still hitting Anthropic's 400 on role=system messages.

    These tests assert the contract the handler enforces: any role=system
    entry in messages is lifted to the top-level system parameter before the
    request is built, regardless of which entry path was used.
    """

    def test_sync_handler_invokes_normalizer(self):
        """
        When a caller enters the sync ``anthropic_messages_handler`` with a
        role=system message and no ``_litellm_messages_presanitized`` flag,
        the handler must invoke ``normalize_system_role_in_anthropic_messages``
        before dispatching. We spy on the helper via monkeypatching.
        """
        from litellm.llms.anthropic.experimental_pass_through.messages import (
            handler as handler_module,
        )

        calls = []

        def spy_normalize(messages, system=None):
            calls.append((list(messages), system))
            # Delegate to the real helper so we exercise the same code path.
            from litellm.llms.anthropic.common_utils import (
                normalize_system_role_in_anthropic_messages as real_normalize,
            )
            return real_normalize(messages, system)

        # Patch the symbol the handler imported.
        original = handler_module.normalize_system_role_in_anthropic_messages
        handler_module.normalize_system_role_in_anthropic_messages = spy_normalize
        try:
            handler_module.anthropic_messages_handler(
                max_tokens=64,
                messages=[
                    {"role": "system", "content": "be concise"},
                    {"role": "user", "content": "hi"},
                ],
                model="claude-test",
            )
        except Exception:
            # We don't care about the dispatch outcome; we only want to
            # confirm normalization was invoked before the failure.
            pass
        finally:
            handler_module.normalize_system_role_in_anthropic_messages = original

        assert len(calls) >= 1, (
            "sync handler did not invoke normalize_system_role_in_anthropic_messages"
        )

    def test_sync_handler_skips_normalizer_when_presanitized_flag_set(self):
        """
        When the async wrapper dispatches to the sync handler it sets
        ``_litellm_messages_presanitized=True`` so the handler does NOT
        normalize again (would be wasted work since messages and system are
        already normalized and not reassigned before dispatch). This test
        pins that contract.
        """
        from litellm.llms.anthropic.experimental_pass_through.messages import (
            handler as handler_module,
        )

        calls = []

        def spy_normalize(messages, system=None):  # pragma: no cover - patched at runtime
            calls.append((list(messages), system))
            from litellm.llms.anthropic.common_utils import (
                normalize_system_role_in_anthropic_messages as real_normalize,
            )
            return real_normalize(messages, system)

        original = handler_module.normalize_system_role_in_anthropic_messages
        handler_module.normalize_system_role_in_anthropic_messages = spy_normalize
        try:
            handler_module.anthropic_messages_handler(
                max_tokens=64,
                messages=[{"role": "user", "content": "hi"}],
                model="claude-test",
                _litellm_messages_presanitized=True,
            )
        except Exception:
            pass
        finally:
            handler_module.normalize_system_role_in_anthropic_messages = original

        assert calls == [], (
            "sync handler re-invoked the normalizer even though "
            "_litellm_messages_presanitized=True was passed"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))