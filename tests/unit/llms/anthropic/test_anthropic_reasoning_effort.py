"""
Tests for _map_reasoning_effort in AnthropicConfig.

Verifies that reasoning_effort=None returns None for all models,
including Claude Opus 4.6.
"""

import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.azure_ai.anthropic.transformation import AzureAnthropicConfig
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeConfig,
)
from litellm.llms.databricks.chat.transformation import DatabricksConfig
from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import (
    VertexAIAnthropicConfig,
)

from litellm.llms.anthropic.chat.transformation import AnthropicConfig


class TestMapReasoningEffort:
    def test_none_returns_none_for_opus_4_6(self):
        """reasoning_effort=None should return None for Opus 4.6, not adaptive."""
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort=None,
            model="claude-opus-4-6",
            custom_llm_provider="anthropic",
        )
        assert result is None

    def test_none_returns_none_for_other_models(self):
        """reasoning_effort=None should return None for non-Opus models."""
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort=None,
            model="claude-4-sonnet-20250514",
            custom_llm_provider="anthropic",
        )
        assert result is None

    def test_opus_4_6_returns_adaptive_for_low(self):
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="low",
            model="claude-opus-4-6",
            custom_llm_provider="anthropic",
        )
        assert result["type"] == "adaptive"

    def test_opus_4_6_returns_adaptive_for_high(self):
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="high",
            model="claude-opus-4-6",
            custom_llm_provider="anthropic",
        )
        assert result["type"] == "adaptive"

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_adaptive_mapping_requests_summarized_display(self, effort):
        """Regression LIT-5714: adaptive thinking without ``display`` makes Anthropic
        return a blank thinking block, so reasoning_effort callers always got
        ``reasoning_content: ""``."""
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort=effort,
            model="claude-opus-4-6",
            custom_llm_provider="anthropic",
        )
        assert result["display"] == "summarized"

    def test_other_model_low_returns_enabled_with_budget(self):
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="low",
            model="claude-4-sonnet-20250514",
            custom_llm_provider="anthropic",
        )
        assert result["type"] == "enabled"
        assert "budget_tokens" in result

    def test_other_model_high_returns_enabled_with_budget(self):
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="high",
            model="claude-4-sonnet-20250514",
            custom_llm_provider="anthropic",
        )
        assert result["type"] == "enabled"
        assert "budget_tokens" in result

    @pytest.mark.parametrize("model", ["claude-opus-4-6", "claude-haiku-5-5"])
    def test_none_string_disables_thinking(self, model: str) -> None:
        result = AnthropicConfig.map_reasoning_effort(
            reasoning_effort="none", model=model, custom_llm_provider="anthropic"
        )
        assert result == {"type": "disabled"}


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "claude-sonnet-5", "claude-opus-5", "claude-opus-4-6"])
@pytest.mark.parametrize("effort", ["none", {"effort": "none", "summary": "concise"}])
def test_explicit_none_disables_thinking_without_budget_or_effort(model: str, effort: object) -> None:
    result = AnthropicConfig().map_openai_params(
        non_default_params={"reasoning_effort": effort},
        optional_params={
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "max"},
        },
        model=model,
        drop_params=False,
    )
    assert result == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "claude-opus-4-6", "claude-fable-5"])
@pytest.mark.parametrize("params", [{}, {"reasoning_effort": None}])
def test_omitted_effort_keeps_provider_defaults(model: str, params: dict) -> None:
    result = AnthropicConfig().map_openai_params(params, {}, model, False)
    assert result == {}


@pytest.mark.parametrize(
    "model",
    ["claude-fable-5", "claude-mythos-5", "claude-opus-5-5", "claude-sonnet-5-5"],
)
def test_none_on_always_on_model_uses_unsupported_parameter_policy(model: str) -> None:
    with pytest.raises(litellm.UnsupportedParamsError, match="thinking cannot be disabled"):
        AnthropicConfig().map_openai_params({"reasoning_effort": "none"}, {}, model, False)
    assert AnthropicConfig().map_openai_params({"reasoning_effort": "none"}, {}, model, True) == {}


@pytest.mark.parametrize(
    "config_class,model",
    [
        (AnthropicConfig, "claude-haiku-5-5"),
        (VertexAIAnthropicConfig, "claude-haiku-5-5"),
        (AzureAnthropicConfig, "claude-haiku-5-5"),
        (AmazonAnthropicClaudeConfig, "anthropic.claude-haiku-5-5"),
        (AmazonConverseConfig, "anthropic.claude-haiku-5-5"),
        (DatabricksConfig, "databricks-claude-haiku-5-5"),
    ],
)
def test_none_reaches_shared_anthropic_routes(config_class: type[BaseConfig], model: str) -> None:
    config = config_class()
    result = config.map_openai_params({"reasoning_effort": "none"}, {}, model, False)
    assert result["thinking"] == {"type": "disabled"}
    assert "output_config" not in result
    assert "thinking" not in config.map_openai_params({}, {}, model, False)


@pytest.mark.parametrize(
    "config_class,model",
    [
        (AnthropicConfig, "claude-fable-5"),
        (AmazonConverseConfig, "anthropic.claude-fable-5"),
        (DatabricksConfig, "databricks-claude-fable-5"),
    ],
)
def test_shared_routes_reject_none_unless_dropping_unsupported_params(
    config_class: type[BaseConfig], model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_class()
    with pytest.raises(litellm.UnsupportedParamsError, match="thinking cannot be disabled"):
        config.map_openai_params({"reasoning_effort": "none"}, {}, model, False)
    assert "thinking" not in config.map_openai_params({"reasoning_effort": "none"}, {}, model, True)
    monkeypatch.setattr(litellm, "drop_params", True)
    assert "thinking" not in config.map_openai_params({"reasoning_effort": "none"}, {}, model, False)


def test_none_on_legacy_default_off_model_keeps_omission() -> None:
    assert AnthropicConfig.map_reasoning_effort("none", "claude-4-sonnet-20250514", "anthropic") is None
