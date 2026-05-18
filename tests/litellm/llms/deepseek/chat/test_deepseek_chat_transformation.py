"""
Unit tests for DeepSeek chat transformation.

Tests the thinking and reasoning_effort parameter handling for DeepSeek models.
"""

import os

# Use the bundled model_prices_and_context_window_backup.json instead of
# fetching from the live URL.  Required so the V4 capability flags added
# in this PR are visible to _supports_factory during tests.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import pytest
from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


class TestDeepSeekThinkingParams:
    """Test thinking and reasoning_effort parameter handling for DeepSeek."""

    def setup_method(self):
        self.config = DeepSeekChatConfig()
        self.model = "deepseek-reasoner"
        self.v4_model = "deepseek-v4-pro"

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

    # --- reasoning_effort on models without native reasoning_effort support
    # (e.g. deepseek-reasoner) ---

    def test_map_reasoning_effort_medium_non_v4(self):
        """Non-V4: reasoning_effort='medium' enables thinking but does NOT forward effort."""
        non_default_params = {"reasoning_effort": "medium"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in result

    def test_map_reasoning_effort_low_non_v4(self):
        """Non-V4: reasoning_effort='low' enables thinking but does NOT forward effort."""
        non_default_params = {"reasoning_effort": "low"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in result

    def test_map_reasoning_effort_high_non_v4(self):
        """Non-V4: reasoning_effort='high' enables thinking but does NOT forward effort."""
        non_default_params = {"reasoning_effort": "high"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in result

    # --- reasoning_effort on V4 models (native support per model_prices) ---

    def test_map_reasoning_effort_high_v4(self):
        """V4: reasoning_effort='high' enables thinking AND forwards effort."""
        non_default_params = {"reasoning_effort": "high"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.v4_model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}
        assert result["reasoning_effort"] == "high"

    def test_map_reasoning_effort_max_v4(self):
        """V4: reasoning_effort='max' is forwarded."""
        non_default_params = {"reasoning_effort": "max"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.v4_model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}
        assert result["reasoning_effort"] == "max"

    def test_map_reasoning_effort_xhigh_normalizes_to_max_v4(self):
        """V4: reasoning_effort='xhigh' normalizes to 'max'."""
        non_default_params = {"reasoning_effort": "xhigh"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.v4_model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}
        assert result["reasoning_effort"] == "max"

    def test_map_reasoning_effort_low_normalizes_to_high_v4(self):
        """V4: reasoning_effort='low' normalizes to 'high'."""
        non_default_params = {"reasoning_effort": "low"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.v4_model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}
        assert result["reasoning_effort"] == "high"

    def test_map_reasoning_effort_medium_normalizes_to_high_v4(self):
        """V4: reasoning_effort='medium' normalizes to 'high'."""
        non_default_params = {"reasoning_effort": "medium"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.v4_model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}
        assert result["reasoning_effort"] == "high"

    # --- Combined thinking + reasoning_effort ---

    def test_thinking_and_reasoning_effort_both_forwarded_v4(self):
        """V4: When both thinking and reasoning_effort are provided,
        both are forwarded — effort is NOT silently dropped."""
        non_default_params = {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "max",
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.v4_model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}
        assert result["reasoning_effort"] == "max"

    def test_thinking_takes_precedence_non_v4(self):
        """Non-V4: When both are provided, thinking is set but
        reasoning_effort is NOT forwarded (model doesn't support it)."""
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

        assert result["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in result

    # --- Edge cases ---

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

    # --- Capability gating via model_prices_and_context_window.json ---

    def test_supports_reasoning_effort_level_v4_pro(self):
        """V4-Pro has supports_high_reasoning_effort and supports_max_reasoning_effort."""
        assert (
            DeepSeekChatConfig._supports_reasoning_effort_level(
                "deepseek-v4-pro", "high"
            )
            is True
        )
        assert (
            DeepSeekChatConfig._supports_reasoning_effort_level(
                "deepseek-v4-pro", "max"
            )
            is True
        )

    def test_supports_reasoning_effort_level_v4_flash(self):
        """V4-Flash has the same reasoning_effort capabilities as V4-Pro."""
        assert (
            DeepSeekChatConfig._supports_reasoning_effort_level(
                "deepseek-v4-flash", "high"
            )
            is True
        )
        assert (
            DeepSeekChatConfig._supports_reasoning_effort_level(
                "deepseek-v4-flash", "max"
            )
            is True
        )

    def test_supports_reasoning_effort_level_v4_with_provider_prefix(self):
        """Provider-prefixed model names resolve to the same capability flags."""
        assert (
            DeepSeekChatConfig._supports_reasoning_effort_level(
                "deepseek/deepseek-v4-pro", "high"
            )
            is True
        )
        assert (
            DeepSeekChatConfig._supports_reasoning_effort_level(
                "deepseek/deepseek-v4-flash", "max"
            )
            is True
        )

    def test_supports_reasoning_effort_level_legacy_models_return_false(self):
        """Legacy DeepSeek models do NOT have supports_*_reasoning_effort
        flags in model_prices_and_context_window.json."""
        assert (
            DeepSeekChatConfig._supports_reasoning_effort_level(
                "deepseek-reasoner", "high"
            )
            is False
        )
        assert (
            DeepSeekChatConfig._supports_reasoning_effort_level(
                "deepseek-reasoner", "max"
            )
            is False
        )
        assert (
            DeepSeekChatConfig._supports_reasoning_effort_level(
                "deepseek-chat", "high"
            )
            is False
        )

    def test_supports_reasoning_effort_level_unknown_model_returns_false(self):
        """Unknown models return False (safe fallback)."""
        assert (
            DeepSeekChatConfig._supports_reasoning_effort_level(
                "deepseek-future-model", "high"
            )
            is False
        )
