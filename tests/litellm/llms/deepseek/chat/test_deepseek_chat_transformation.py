"""
Unit tests for DeepSeek chat transformation.

Tests the thinking and reasoning_effort parameter handling for DeepSeek models.
"""

import pytest
from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


class TestDeepSeekThinkingParams:
    """Test thinking and reasoning_effort parameter handling for DeepSeek."""

    def setup_method(self):
        self.config = DeepSeekChatConfig()
        self.model = "deepseek-reasoner"

    def test_get_supported_openai_params_includes_thinking(self):
        """Test that thinking and reasoning_effort are in supported params."""
        params = self.config.get_supported_openai_params(self.model)
        assert "thinking" in params
        assert "reasoning_effort" in params

    def test_map_thinking_enabled(self):
        """Test that thinking={"type": "enabled"} is passed through correctly."""
        non_default_params = {"thinking": {"type": "enabled"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_thinking_with_budget_tokens_strips_budget(self):
        """Test that budget_tokens is stripped from thinking param (DeepSeek doesn't support it)."""
        non_default_params = {"thinking": {"type": "enabled", "budget_tokens": 2048}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Should strip budget_tokens, only pass type
        assert result["thinking"] == {"type": "enabled"}
        assert "budget_tokens" not in result.get("thinking", {})

    def test_map_reasoning_effort_medium(self):
        """Test that reasoning_effort='medium' normalizes to high for V4 models."""
        non_default_params = {"reasoning_effort": "medium"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="deepseek-v4-pro",
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}
        assert result["reasoning_effort"] == "high"

    def test_map_reasoning_effort_low(self):
        """Test that reasoning_effort='low' normalizes to high for V4 models."""
        non_default_params = {"reasoning_effort": "low"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="deepseek-v4-pro",
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}
        assert result["reasoning_effort"] == "high"

    def test_map_reasoning_effort_high(self):
        """Test that reasoning_effort='high' passes through as high for V4 models."""
        non_default_params = {"reasoning_effort": "high"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="deepseek-v4-pro",
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}
        assert result["reasoning_effort"] == "high"

    def test_map_reasoning_effort_max(self):
        """Test that reasoning_effort='max' passes through as max for V4 models."""
        non_default_params = {"reasoning_effort": "max"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="deepseek-v4-pro",
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}
        assert result["reasoning_effort"] == "max"

    def test_map_reasoning_effort_xhigh_normalizes_to_max(self):
        """Test that reasoning_effort='xhigh' normalizes to max for V4 models."""
        non_default_params = {"reasoning_effort": "xhigh"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="deepseek-v4-pro",
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}
        assert result["reasoning_effort"] == "max"

    def test_map_reasoning_effort_not_forwarded_for_reasoner(self):
        """Test that reasoning_effort is not forwarded to deepseek-reasoner (R1 doesn't accept it)."""
        non_default_params = {"reasoning_effort": "max"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,  # deepseek-reasoner
            drop_params=False,
        )

        # thinking should still be enabled but reasoning_effort must NOT be forwarded
        assert result["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in result

    def test_map_reasoning_effort_none_is_noop_for_reasoner(self):
        """Test that reasoning_effort='none' is a no-op for deepseek-reasoner (always-on thinking)."""
        non_default_params = {"reasoning_effort": "none"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,  # deepseek-reasoner
            drop_params=False,
        )

        # deepseek-reasoner has always-on thinking, API rejects {"type": "disabled"}
        assert "thinking" not in result
        assert "reasoning_effort" not in result

    def test_map_reasoning_effort_none_disables_thinking_for_v4(self):
        """Test that reasoning_effort='none' sends thinking disabled for V4 opt-in models."""
        non_default_params = {"reasoning_effort": "none"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="deepseek-v4-pro",
            drop_params=False,
        )

        assert result["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in result

    def test_map_reasoning_effort_null_does_not_enable_thinking(self):
        """Test that reasoning_effort=None does not enable thinking."""
        non_default_params = {"reasoning_effort": None}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_thinking_takes_precedence_over_reasoning_effort(self):
        """Test that thinking param takes precedence when both are provided."""
        non_default_params = {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # thinking should be set, reasoning_effort should not override
        assert result["thinking"] == {"type": "enabled"}

    def test_invalid_thinking_type_ignored(self):
        """Test that invalid thinking type values are ignored."""
        non_default_params = {"thinking": {"type": "invalid"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_thinking_none_value_ignored(self):
        """Test that thinking=None is ignored."""
        non_default_params = {"thinking": None}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_map_thinking_disabled_is_noop_for_reasoner(self):
        """Test that thinking={"type": "disabled"} is a no-op for deepseek-reasoner (always-on thinking)."""
        non_default_params = {"thinking": {"type": "disabled"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,  # deepseek-reasoner
            drop_params=False,
        )

        # deepseek-reasoner rejects {"type": "disabled"} - should be silently dropped
        assert "thinking" not in result

    def test_map_thinking_disabled_passes_through_for_v4(self):
        """Test that thinking={"type": "disabled"} is passed through for V4 opt-in models."""
        non_default_params = {"thinking": {"type": "disabled"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="deepseek-v4-pro",
            drop_params=False,
        )

        assert result["thinking"] == {"type": "disabled"}

    def test_map_thinking_disabled_with_budget_tokens_strips_budget(self):
        """Test that budget_tokens is stripped even when thinking is disabled."""
        non_default_params = {"thinking": {"type": "disabled", "budget_tokens": 0}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="deepseek-v4-pro",
            drop_params=False,
        )

        assert result["thinking"] == {"type": "disabled"}
        assert "budget_tokens" not in result.get("thinking", {})

    def test_map_thinking_and_reasoning_effort_both_forwarded_for_v4(self):
        """Test that when both thinking and reasoning_effort are provided for V4, effort is not dropped."""
        non_default_params = {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "max",
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="deepseek-v4-pro",
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}
        assert result["reasoning_effort"] == "max"


class TestFillReasoningContent:
    """Test _fill_reasoning_content helper for multi-turn thinking-mode conversations."""

    def setup_method(self):
        self.config = DeepSeekChatConfig()

    def test_injects_placeholder_when_reasoning_content_missing(self):
        """Assistant message missing reasoning_content gets a space placeholder injected."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = self.config._fill_reasoning_content(messages)
        assert result[1]["reasoning_content"] == " "

    def test_promotes_reasoning_content_from_provider_specific_fields(self):
        """reasoning_content stored in provider_specific_fields is promoted to top level."""
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "hi",
                "provider_specific_fields": {"reasoning_content": "my reasoning"},
            },
        ]
        result = self.config._fill_reasoning_content(messages)
        assert result[1]["reasoning_content"] == "my reasoning"
        assert "reasoning_content" not in result[1].get("provider_specific_fields", {})

    def test_does_not_overwrite_existing_reasoning_content(self):
        """Assistant message that already has reasoning_content is left unchanged."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "reasoning_content": "already here"},
        ]
        result = self.config._fill_reasoning_content(messages)
        assert result[1]["reasoning_content"] == "already here"

    def test_non_assistant_messages_are_unchanged(self):
        """User and system messages are passed through untouched."""
        messages = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello"},
        ]
        result = self.config._fill_reasoning_content(messages)
        assert result == messages
