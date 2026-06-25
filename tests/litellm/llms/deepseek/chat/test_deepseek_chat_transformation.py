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
        """Test that reasoning_effort='medium' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "medium"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_low(self):
        """Test that reasoning_effort='low' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "low"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_high(self):
        """Test that reasoning_effort='high' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "high"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_none_does_not_enable_thinking(self):
        """Test that reasoning_effort='none' does not enable thinking."""
        non_default_params = {"reasoning_effort": "none"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

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


class TestDeepSeekV4ToolChoiceConstraint:
    """
    DeepSeek V4 (deepseek-v4-pro / deepseek-v4-flash, released 2026-04) is
    reasoner-only and rejects ``tool_choice="required"``, ``"any"``, or a
    dict form with HTTP 400 "deepseek-reasoner does not support this
    tool_choice", even when the caller did NOT explicitly send
    ``thinking={"type":"enabled"}`` (the server enables thinking mode by
    default for V4). Only ``"auto"`` and ``"none"`` are accepted in thinking
    mode. Verify the transformation converts unsupported values to ``"auto"``
    so SDKs that hardcode ``tool_choice="required"`` (e.g. OpenAI Agents SDK
    delegation paths) keep working on V4.
    """

    def setup_method(self):
        self.config = DeepSeekChatConfig()

    def test_v4_pro_tool_choice_required_converted_to_auto(self):
        """V4-pro + tool_choice='required' → 'auto' (model-name match, no thinking required)."""
        result = self.config.map_openai_params(
            non_default_params={"tool_choice": "required"},
            optional_params={"tool_choice": "required"},
            model="deepseek/deepseek-v4-pro",
            drop_params=False,
        )
        assert result["tool_choice"] == "auto"

    def test_v4_flash_tool_choice_any_converted_to_auto(self):
        """V4-flash + tool_choice='any' → 'auto'."""
        result = self.config.map_openai_params(
            non_default_params={"tool_choice": "any"},
            optional_params={"tool_choice": "any"},
            model="deepseek-v4-flash",
            drop_params=False,
        )
        assert result["tool_choice"] == "auto"

    def test_v4_pro_tool_choice_dict_form_converted_to_auto(self):
        """V4-pro + dict tool_choice (specific function) → 'auto'."""
        tc = {"type": "function", "function": {"name": "my_tool"}}
        result = self.config.map_openai_params(
            non_default_params={"tool_choice": tc},
            optional_params={"tool_choice": tc},
            model="deepseek/deepseek-v4-pro",
            drop_params=False,
        )
        assert result["tool_choice"] == "auto"

    def test_v4_pro_tool_choice_auto_unchanged(self):
        """V4-pro + tool_choice='auto' → unchanged (already valid)."""
        result = self.config.map_openai_params(
            non_default_params={"tool_choice": "auto"},
            optional_params={"tool_choice": "auto"},
            model="deepseek/deepseek-v4-pro",
            drop_params=False,
        )
        assert result["tool_choice"] == "auto"

    def test_v4_pro_tool_choice_none_unchanged(self):
        """V4-pro + tool_choice='none' → unchanged (already valid)."""
        result = self.config.map_openai_params(
            non_default_params={"tool_choice": "none"},
            optional_params={"tool_choice": "none"},
            model="deepseek/deepseek-v4-pro",
            drop_params=False,
        )
        assert result["tool_choice"] == "none"

    def test_non_v4_chat_model_tool_choice_required_unchanged(self):
        """Non-V4 deepseek-chat + tool_choice='required' → unchanged (V3 supports it)."""
        result = self.config.map_openai_params(
            non_default_params={"tool_choice": "required"},
            optional_params={"tool_choice": "required"},
            model="deepseek/deepseek-chat",
            drop_params=False,
        )
        assert result["tool_choice"] == "required"

    def test_thinking_enabled_explicit_also_converts(self):
        """Explicit thinking={'type':'enabled'} on any deepseek model converts tool_choice too."""
        result = self.config.map_openai_params(
            non_default_params={
                "thinking": {"type": "enabled"},
                "tool_choice": "required",
            },
            optional_params={
                "thinking": {"type": "enabled"},
                "tool_choice": "required",
            },
            model="deepseek-reasoner",
            drop_params=False,
        )
        assert result["tool_choice"] == "auto"
