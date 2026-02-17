"""
Tests for completion() performance optimizations.

Covers:
1. LlmProvidersSet used instead of list comprehension for provider lookup
2. Deferred locals() - get_first_chars_messages works with minimal dict
"""

import litellm
from litellm.types.utils import LlmProviders, LlmProvidersSet


class TestLlmProvidersSetConsistency:
    """Verify LlmProvidersSet matches the enum values exactly."""

    def test_set_matches_enum(self):
        expected = {p.value for p in LlmProviders}
        assert LlmProvidersSet == expected

    def test_known_providers_in_set(self):
        for provider in ["openai", "anthropic", "azure", "bedrock", "vertex_ai"]:
            assert provider in LlmProvidersSet, f"{provider} not in LlmProvidersSet"

    def test_unknown_provider_not_in_set(self):
        assert "not_a_real_provider_xyz" not in LlmProvidersSet


class TestGetFirstCharsMessagesWithMinimalKwargs:
    """Verify get_first_chars_messages works when completion_kwargs only has 'messages'."""

    def test_with_messages_key(self):
        """get_first_chars_messages should work with just {"messages": ...}."""
        messages = [{"role": "user", "content": "hello"}]
        result = litellm.get_first_chars_messages(kwargs={"messages": messages})
        assert "hello" in result

    def test_with_empty_dict(self):
        """get_first_chars_messages should handle empty dict gracefully."""
        result = litellm.get_first_chars_messages(kwargs={})
        # Should return empty string or "None" — not crash
        assert isinstance(result, str)

    def test_truncates_long_messages(self):
        """get_first_chars_messages truncates to 100 chars."""
        messages = [{"role": "user", "content": "x" * 200}]
        result = litellm.get_first_chars_messages(kwargs={"messages": messages})
        assert len(result) <= 100
