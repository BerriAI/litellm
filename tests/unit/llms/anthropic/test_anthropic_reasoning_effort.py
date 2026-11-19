"""
Tests for _map_reasoning_effort in AnthropicConfig.

Verifies that reasoning_effort=None returns None for all models,
including Claude Opus 4.6.
"""

from typing import Final

import pytest

from litellm.exceptions import BadRequestError
from litellm.llms.anthropic.chat.transformation import AnthropicConfig


@pytest.mark.parametrize(
    "model, expected",
    [
        ("claude-sonnet-5", {"type": "disabled"}),
        ("claude-opus-5", {"type": "disabled"}),
        ("claude-haiku-5-5", {"type": "disabled"}),
        ("claude-haiku-4-5", None),
        ("claude-sonnet-5-5", None),
        ("claude-opus-5-5", None),
        ("claude-fable-5", None),
        ("claude-mythos-5", None),
    ],
)
def test_none_disables_only_optional_default_thinking(model: str, expected: dict[str, str] | None) -> None:
    mapped: Final = AnthropicConfig().map_openai_params({"reasoning_effort": "none"}, {}, model, False)
    assert mapped.get("thinking") == expected
    assert "output_config" not in mapped


@pytest.mark.parametrize("effort", [None, "bogus"])
def test_default_thinking_preserves_absent_and_invalid_effort(effort: str | None) -> None:
    if effort is None:
        assert AnthropicConfig.map_reasoning_effort(effort, "claude-sonnet-5", "anthropic") is None
    else:
        with pytest.raises(BadRequestError, match="Invalid reasoning_effort"):
            AnthropicConfig().map_openai_params({"reasoning_effort": effort}, {}, "claude-sonnet-5", False)


class TestMapReasoningEffort:
    def test_none_returns_none_for_opus_4_6(self):
        """reasoning_effort=None should return None for Opus 4.6, not adaptive."""
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort=None, model="claude-opus-4-6", custom_llm_provider="anthropic"
        )
        assert result is None

    def test_none_returns_none_for_other_models(self):
        """reasoning_effort=None should return None for non-Opus models."""
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort=None, model="claude-4-sonnet-20250514", custom_llm_provider="anthropic"
        )
        assert result is None

    def test_opus_4_6_returns_adaptive_for_low(self):
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="low", model="claude-opus-4-6", custom_llm_provider="anthropic"
        )
        assert result["type"] == "adaptive"

    def test_opus_4_6_returns_adaptive_for_high(self):
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="high", model="claude-opus-4-6", custom_llm_provider="anthropic"
        )
        assert result["type"] == "adaptive"

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_adaptive_mapping_requests_summarized_display(self, effort):
        """Regression LIT-5714: adaptive thinking without ``display`` makes Anthropic
        return a blank thinking block, so reasoning_effort callers always got
        ``reasoning_content: ""``."""
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort=effort, model="claude-opus-4-6", custom_llm_provider="anthropic"
        )
        assert result["display"] == "summarized"

    def test_other_model_low_returns_enabled_with_budget(self):
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="low", model="claude-4-sonnet-20250514", custom_llm_provider="anthropic"
        )
        assert result["type"] == "enabled"
        assert "budget_tokens" in result

    def test_other_model_high_returns_enabled_with_budget(self):
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="high", model="claude-4-sonnet-20250514", custom_llm_provider="anthropic"
        )
        assert result["type"] == "enabled"
        assert "budget_tokens" in result

    def test_none_string_returns_none_for_opus_4_6(self):
        """reasoning_effort='none' should return None for Opus 4.6."""
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="none", model="claude-opus-4-6", custom_llm_provider="anthropic"
        )
        assert result is None

    def test_none_string_returns_none_for_other_models(self):
        """reasoning_effort='none' should return None for non-Opus models."""
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="none", model="claude-4-sonnet-20250514", custom_llm_provider="anthropic"
        )
        assert result is None
