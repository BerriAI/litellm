"""
Tests for litellm/llms/base_llm/chat/transformation.py
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.base_llm.chat.transformation import BaseConfig


class TestIsThinkingEnabled:
    """Tests for BaseConfig.is_thinking_enabled()."""

    def setup_method(self):
        # Use a concrete subclass since BaseConfig is ABC
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        self.config = OpenAIGPTConfig()

    def test_thinking_enabled_with_type_enabled(self):
        """Standard case: thinking dict with type='enabled' returns True."""
        params = {"thinking": {"type": "enabled", "budget_tokens": 1000}}
        assert self.config.is_thinking_enabled(params) is True

    def test_thinking_disabled_with_type_disabled(self):
        """thinking dict with type='disabled' should return False."""
        params = {"thinking": {"type": "disabled"}}
        assert self.config.is_thinking_enabled(params) is False

    def test_thinking_key_is_none(self):
        """thinking=None must not crash with AttributeError.

        Previously, non_default_params.get("thinking", {}) would return None
        (since None is a valid value), and then calling .get("type") on None
        raises AttributeError. See: https://github.com/BerriAI/litellm/issues/28576
        """
        params = {"thinking": None}
        # Should not raise, should return False (no reasoning_effort either)
        assert self.config.is_thinking_enabled(params) is False

    def test_thinking_key_absent_no_reasoning_effort(self):
        """No thinking key and no reasoning_effort returns False."""
        params = {}
        assert self.config.is_thinking_enabled(params) is False

    def test_reasoning_effort_enables_thinking(self):
        """reasoning_effort being non-None should return True even without thinking key."""
        params = {"reasoning_effort": "high"}
        assert self.config.is_thinking_enabled(params) is True

    def test_reasoning_effort_none_value(self):
        """reasoning_effort=None is explicitly set but None, should return False."""
        params = {"reasoning_effort": None}
        assert self.config.is_thinking_enabled(params) is False

    def test_thinking_none_but_reasoning_effort_set(self):
        """thinking=None with reasoning_effort set should return True."""
        params = {"thinking": None, "reasoning_effort": "medium"}
        assert self.config.is_thinking_enabled(params) is True

    def test_thinking_is_string_not_dict(self):
        """Unexpected type for thinking (e.g. a string) should not crash."""
        params = {"thinking": "enabled"}
        # Should not raise — 'enabled' is not a dict so isinstance check is False
        assert self.config.is_thinking_enabled(params) is False

    def test_thinking_is_empty_dict(self):
        """Empty dict for thinking returns False (no type key)."""
        params = {"thinking": {}}
        assert self.config.is_thinking_enabled(params) is False
