"""
Unit tests for AnthropicCacheControlHook fixes:
  1. Negative string indices (e.g. "-1") are now parsed correctly (isdigit bug fix)
  2. Combined role + index filtering targets the Nth message of a given role
"""

import pytest
from unittest.mock import patch

from litellm.integrations.anthropic_cache_control_hook import (
    AnthropicCacheControlHook,
)
from litellm.types.llms.openai import ChatCompletionCachedContent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_messages():
    """
    Build a realistic multi-turn conversation:
      0: system
      1: user       (user #0)
      2: assistant   (assistant #0)
      3: user       (user #1)
      4: assistant   (assistant #1)
      5: user       (user #2)
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "First user message"},
        {"role": "assistant", "content": "First assistant reply"},
        {"role": "user", "content": "Second user message"},
        {"role": "assistant", "content": "Second assistant reply"},
        {"role": "user", "content": "Third user message"},
    ]


def _has_cache_control(msg):
    """Check whether a message was annotated with cache_control."""
    # String content: cache_control is a top-level key
    if "cache_control" in msg:
        return True
    # List content: cache_control is on the last content block
    content = msg.get("content")
    if isinstance(content, list):
        return any("cache_control" in item for item in content if isinstance(item, dict))
    return False


# ---------------------------------------------------------------------------
# 1. Bug fix: negative *string* indices ("-1") were rejected by isdigit()
# ---------------------------------------------------------------------------

class TestNegativeStringIndexParsing:
    """isdigit() returns False for '-1'; the fix uses try/except int()."""

    def test_negative_string_index_targets_last_message(self):
        msgs = _make_messages()
        point = {"location": "message", "index": "-1"}  # string, not int
        result = AnthropicCacheControlHook._process_message_injection(
            point=point, messages=msgs
        )
        assert _has_cache_control(result[-1]), "Last message should have cache_control"
        # No other message should be affected
        for m in result[:-1]:
            assert not _has_cache_control(m)

    def test_negative_string_index_minus_two(self):
        msgs = _make_messages()
        point = {"location": "message", "index": "-2"}
        result = AnthropicCacheControlHook._process_message_injection(
            point=point, messages=msgs
        )
        assert _has_cache_control(result[-2])
        for i, m in enumerate(result):
            if i != len(result) - 2:
                assert not _has_cache_control(m)

    def test_positive_string_index_still_works(self):
        msgs = _make_messages()
        point = {"location": "message", "index": "0"}
        result = AnthropicCacheControlHook._process_message_injection(
            point=point, messages=msgs
        )
        assert _has_cache_control(result[0])

    def test_non_numeric_string_index_is_ignored(self):
        msgs = _make_messages()
        point = {"location": "message", "index": "abc"}
        result = AnthropicCacheControlHook._process_message_injection(
            point=point, messages=msgs
        )
        # Nothing should be modified
        for m in result:
            assert not _has_cache_control(m)


# ---------------------------------------------------------------------------
# 2. New feature: combined role + index filtering (Case 1)
# ---------------------------------------------------------------------------

class TestRolePlusIndexFiltering:
    """When both role and index are set, index is relative to the role subset."""

    def test_last_assistant_message(self):
        """role=assistant, index=-1 should target the *last* assistant message."""
        msgs = _make_messages()
        point = {"location": "message", "role": "assistant", "index": -1}
        result = AnthropicCacheControlHook._process_message_injection(
            point=point, messages=msgs
        )
        # assistant messages are at indices 2 and 4; last = index 4
        assert _has_cache_control(result[4])
        for i, m in enumerate(result):
            if i != 4:
                assert not _has_cache_control(m)

    def test_first_user_message(self):
        """role=user, index=0 should target only the first user message."""
        msgs = _make_messages()
        point = {"location": "message", "role": "user", "index": 0}
        result = AnthropicCacheControlHook._process_message_injection(
            point=point, messages=msgs
        )
        # user messages are at indices 1, 3, 5; first = index 1
        assert _has_cache_control(result[1])
        for i, m in enumerate(result):
            if i != 1:
                assert not _has_cache_control(m)

    def test_last_user_message(self):
        """role=user, index=-1 should target only the last user message."""
        msgs = _make_messages()
        point = {"location": "message", "role": "user", "index": -1}
        result = AnthropicCacheControlHook._process_message_injection(
            point=point, messages=msgs
        )
        # user messages at 1, 3, 5; last = index 5
        assert _has_cache_control(result[5])
        for i, m in enumerate(result):
            if i != 5:
                assert not _has_cache_control(m)

    def test_second_assistant_message_via_index_1(self):
        """role=assistant, index=1 should target the second assistant message."""
        msgs = _make_messages()
        point = {"location": "message", "role": "assistant", "index": 1}
        result = AnthropicCacheControlHook._process_message_injection(
            point=point, messages=msgs
        )
        # assistant at 2, 4; second = index 4
        assert _has_cache_control(result[4])
        for i, m in enumerate(result):
            if i != 4:
                assert not _has_cache_control(m)

    def test_role_plus_string_negative_index(self):
        """Combined role + negative *string* index exercises both fixes together."""
        msgs = _make_messages()
        point = {"location": "message", "role": "user", "index": "-1"}
        result = AnthropicCacheControlHook._process_message_injection(
            point=point, messages=msgs
        )
        assert _has_cache_control(result[5])

    def test_role_plus_index_out_of_bounds_logs_warning(self):
        """Out-of-bounds index within the role subset should warn, not crash."""
        msgs = _make_messages()
        point = {"location": "message", "role": "assistant", "index": 10}
        with patch(
            "litellm.integrations.anthropic_cache_control_hook.verbose_logger"
        ) as mock_logger:
            result = AnthropicCacheControlHook._process_message_injection(
                point=point, messages=msgs
            )
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "out of bounds" in warning_msg.lower()
        # Nothing should be modified
        for m in result:
            assert not _has_cache_control(m)

    def test_role_with_no_matching_messages(self):
        """If the target role has zero messages, nothing happens (no crash)."""
        msgs = _make_messages()
        point = {"location": "message", "role": "tool", "index": 0}
        result = AnthropicCacheControlHook._process_message_injection(
            point=point, messages=msgs
        )
        for m in result:
            assert not _has_cache_control(m)

    def test_system_role_with_index_zero(self):
        """role=system, index=0 targets the single system message."""
        msgs = _make_messages()
        point = {"location": "message", "role": "system", "index": 0}
        result = AnthropicCacheControlHook._process_message_injection(
            point=point, messages=msgs
        )
        assert _has_cache_control(result[0])
        for m in result[1:]:
            assert not _has_cache_control(m)
