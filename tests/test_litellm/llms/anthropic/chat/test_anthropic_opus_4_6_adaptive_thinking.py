"""
Tests for Claude Opus 4.6 adaptive thinking migration.

Opus 4.6 uses `thinking: {type: "adaptive"}` + `output_config: {effort: ...}`
instead of the deprecated `thinking: {type: "enabled", budget_tokens: N}`.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.constants import (
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
)
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.common_utils import AnthropicModelInfo


class TestOpus46AdaptiveThinking:
    """Tests for Opus 4.6 adaptive thinking parameter mapping."""

    def setup_method(self):
        self.config = AnthropicConfig()

    def test_reasoning_effort_maps_to_adaptive_thinking_for_opus_4_6(self):
        """reasoning_effort='high' produces thinking: {type: 'adaptive'} + output_config: {effort: 'high'}"""
        optional_params = {}
        non_default_params = {"reasoning_effort": "high"}
        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="claude-opus-4-6-20250924",
            drop_params=False,
        )
        assert result["thinking"] == {"type": "adaptive"}
        assert result["output_config"] == {"effort": "high"}

    def test_reasoning_effort_levels_for_opus_4_6(self):
        """'medium' and 'low' reasoning_effort map correctly for Opus 4.6."""
        for level in ("medium", "low"):
            optional_params = {}
            non_default_params = {"reasoning_effort": level}
            result = self.config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model="claude-opus-4-6-20250924",
                drop_params=False,
            )
            assert result["thinking"] == {"type": "adaptive"}
            assert result["output_config"] == {"effort": level}

    def test_reasoning_effort_minimal_maps_to_low_for_opus_4_6(self):
        """'minimal' is not in (low, medium, high) so should map to 'low' for Opus 4.6."""
        optional_params = {}
        non_default_params = {"reasoning_effort": "minimal"}
        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="claude-opus-4-6-20250924",
            drop_params=False,
        )
        assert result["thinking"] == {"type": "adaptive"}
        assert result["output_config"] == {"effort": "low"}

    def test_thinking_adaptive_passthrough_for_opus_4_6(self):
        """thinking: {type: 'adaptive'} passes through as-is for Opus 4.6."""
        optional_params = {}
        non_default_params = {"thinking": {"type": "adaptive"}}
        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="claude-opus-4-6-20250924",
            drop_params=False,
        )
        assert result["thinking"] == {"type": "adaptive"}
        assert "output_config" not in result

    def test_deprecated_thinking_enabled_converted_for_opus_4_6(self):
        """thinking: {type: 'enabled', budget_tokens: 4096} converts to adaptive + effort for Opus 4.6."""
        optional_params = {}
        non_default_params = {"thinking": {"type": "enabled", "budget_tokens": 4096}}
        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="claude-opus-4-6-20250924",
            drop_params=False,
        )
        assert result["thinking"] == {"type": "adaptive"}
        assert result["output_config"] == {"effort": "high"}

    def test_budget_tokens_to_effort_mapping(self):
        """Unit test for _budget_tokens_to_effort threshold mapping."""
        # At or below LOW threshold -> low
        assert (
            AnthropicConfig._budget_tokens_to_effort(
                DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET
            )
            == "low"
        )
        assert AnthropicConfig._budget_tokens_to_effort(512) == "low"

        # Above LOW but at or below MEDIUM threshold -> medium
        assert (
            AnthropicConfig._budget_tokens_to_effort(
                DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET
            )
            == "medium"
        )
        assert (
            AnthropicConfig._budget_tokens_to_effort(
                DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET + 1
            )
            == "medium"
        )

        # Above MEDIUM threshold -> high
        assert (
            AnthropicConfig._budget_tokens_to_effort(
                DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET + 1
            )
            == "high"
        )
        assert AnthropicConfig._budget_tokens_to_effort(10000) == "high"

    def test_opus_4_5_unchanged(self):
        """Opus 4.5 still uses type='enabled' with budget_tokens (not adaptive)."""
        optional_params = {}
        non_default_params = {"reasoning_effort": "high"}
        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="claude-opus-4-5-20250514",
            drop_params=False,
        )
        assert result["thinking"]["type"] == "enabled"
        assert "budget_tokens" in result["thinking"]
        assert result["output_config"] == {"effort": "high"}


class TestOpus46BetaHeader:
    """Tests for Opus 4.6 effort beta header behavior."""

    def setup_method(self):
        self.model_info = AnthropicModelInfo()

    def test_opus_4_6_no_effort_beta_header(self):
        """is_effort_used returns False for Opus 4.6 (effort is GA)."""
        optional_params = {"output_config": {"effort": "high"}}
        assert (
            self.model_info.is_effort_used(
                optional_params, model="claude-opus-4-6-20250924"
            )
            is False
        )

    def test_opus_4_5_still_gets_effort_beta_header(self):
        """is_effort_used returns True for Opus 4.5 with reasoning_effort."""
        optional_params = {"reasoning_effort": "high"}
        assert (
            self.model_info.is_effort_used(
                optional_params, model="claude-opus-4-5-20250514"
            )
            is True
        )


class TestIsThinkingEnabledAdaptive:
    """Tests for base class is_thinking_enabled recognizing adaptive type."""

    def setup_method(self):
        self.config = AnthropicConfig()

    def test_is_thinking_enabled_recognizes_adaptive(self):
        """Base class is_thinking_enabled recognizes type='adaptive'."""
        assert (
            self.config.is_thinking_enabled({"thinking": {"type": "adaptive"}}) is True
        )

    def test_is_thinking_enabled_still_recognizes_enabled(self):
        """Base class is_thinking_enabled still recognizes type='enabled'."""
        assert (
            self.config.is_thinking_enabled(
                {"thinking": {"type": "enabled", "budget_tokens": 4096}}
            )
            is True
        )

    def test_is_thinking_enabled_recognizes_reasoning_effort(self):
        """Base class is_thinking_enabled recognizes reasoning_effort."""
        assert self.config.is_thinking_enabled({"reasoning_effort": "high"}) is True


class TestMaxTokensNotAutoAdjusted:
    """Tests that adaptive thinking (no budget_tokens) doesn't trigger max_tokens auto-adjustment."""

    def setup_method(self):
        self.config = AnthropicConfig()

    def test_max_tokens_not_auto_adjusted_for_adaptive(self):
        """No budget_tokens means update_optional_params_with_thinking_tokens is a no-op."""
        non_default_params = {"thinking": {"type": "adaptive"}}
        optional_params = {"thinking": {"type": "adaptive"}}
        # Should not raise and should not add max_tokens
        self.config.update_optional_params_with_thinking_tokens(
            non_default_params=non_default_params,
            optional_params=optional_params,
        )
        assert "max_tokens" not in optional_params


class TestOpus46ResponseFormat:
    """Tests that Opus 4.6 uses native output_format for response_format."""

    def setup_method(self):
        self.config = AnthropicConfig()

    def test_opus_4_6_response_format_uses_output_format(self):
        """response_format uses native output_format path for Opus 4.6."""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
            },
        }
        optional_params = {}
        non_default_params = {"response_format": response_format}
        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="claude-opus-4-6-20250924",
            drop_params=False,
        )
        assert "output_format" in result


class TestOpus46ModelDetection:
    """Tests for _is_claude_opus_4_6 with various model name formats."""

    def setup_method(self):
        self.config = AnthropicConfig()

    def test_opus_4_6_model_detection(self):
        """_is_claude_opus_4_6 detects various Opus 4.6 model name formats."""
        assert self.config._is_claude_opus_4_6("claude-opus-4-6-20250924") is True
        assert self.config._is_claude_opus_4_6("claude-opus-4.6-20250924") is True
        assert self.config._is_claude_opus_4_6("anthropic/claude-opus-4-6") is True
        assert self.config._is_claude_opus_4_6("CLAUDE-OPUS-4-6") is True

        # Should NOT match other models
        assert self.config._is_claude_opus_4_6("claude-opus-4-5-20250514") is False
        assert self.config._is_claude_opus_4_6("claude-sonnet-4-5") is False
        assert self.config._is_claude_opus_4_6("claude-opus-4-1") is False


class TestBedrockConverseOpus46:
    """Tests for Bedrock Converse reasoning_effort handling with Opus 4.6."""

    def test_bedrock_converse_reasoning_effort_opus_4_6(self):
        """Bedrock Converse uses adaptive thinking for Opus 4.6."""
        from litellm.llms.bedrock.chat.converse_transformation import (
            AmazonConverseConfig,
        )

        config = AmazonConverseConfig()
        optional_params: dict = {}
        config._handle_reasoning_effort_parameter(
            model="us.anthropic.claude-opus-4-6-20250924-v1:0",
            reasoning_effort="high",
            optional_params=optional_params,
        )
        assert optional_params["thinking"] == {"type": "adaptive"}
