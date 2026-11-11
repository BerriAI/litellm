"""
Tests for _map_reasoning_effort in AnthropicConfig.

Verifies that reasoning_effort=None returns None for all models,
including Claude Opus 4.6.
"""

from typing import Final

import pytest

import litellm
from litellm.exceptions import BadRequestError
from litellm.llms.anthropic.chat.transformation import AnthropicConfig


@pytest.mark.parametrize(
    "model, expected",
    [
        ("claude-sonnet-5", {"type": "disabled"}),
        ("claude-opus-5", {"type": "disabled"}),
        ("claude-haiku-5-5", {"type": "disabled"}),
        ("claude-haiku-4-5", None),
        ("claude-sonnet-5-5", {"type": "between_tools"}),
        ("claude-opus-5-5", None),
        ("claude-fable-5", None),
        ("claude-mythos-5", None),
    ],
)
def test_none_disables_only_optional_default_thinking(model: str, expected: dict[str, str] | None) -> None:
    mapped: Final = AnthropicConfig().map_openai_params({"reasoning_effort": "none"}, {}, model, True)
    assert mapped.get("thinking") == expected
    assert "output_config" not in mapped


@pytest.mark.parametrize("drop_params", [False, True])
@pytest.mark.parametrize("provider", ["anthropic", "bedrock", "vertex_ai", "azure_ai", "databricks"])
def test_always_on_none_requires_drop_params(provider: str, drop_params: bool) -> None:
    if not drop_params:
        with pytest.raises(BadRequestError, match=r"cannot disable thinking.*drop_params") as error:
            AnthropicConfig.map_reasoning_effort("none", "claude-fable-5-1", provider, drop_params=drop_params)
        assert error.value.status_code == 400
        return
    assert AnthropicConfig.map_reasoning_effort("none", "claude-fable-5-1", provider, drop_params=drop_params) is None


def test_global_drop_params_omits_always_on_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "drop_params", True)
    assert AnthropicConfig.map_reasoning_effort("none", "claude-fable-5-1", "anthropic") is None


@pytest.mark.parametrize("drop_params", [False, True])
@pytest.mark.parametrize("provider", ["anthropic", "bedrock", "vertex_ai", "azure_ai", "databricks"])
def test_none_uses_between_tools_without_adaptive_effort(provider: str, drop_params: bool) -> None:
    assert AnthropicConfig.map_reasoning_effort("none", "claude-sonnet-5-5", provider, drop_params=drop_params) == {
        "type": "between_tools"
    }
    mapped: Final = AnthropicConfig().map_openai_params(
        {"reasoning_effort": "none"},
        {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
        "claude-sonnet-5-5",
        drop_params,
    )
    assert mapped == {"thinking": {"type": "between_tools"}}


@pytest.mark.parametrize("drop_params", [False, True])
def test_provider_can_disable_between_tools_capability(monkeypatch: pytest.MonkeyPatch, drop_params: bool) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "azure_ai/claude-sonnet-5-5",
        {
            **litellm.model_cost["azure_ai/claude-sonnet-5-5"],
            "supports_between_tools_thinking": False,
        },
    )
    if drop_params:
        assert AnthropicConfig.map_reasoning_effort("none", "claude-sonnet-5-5", "azure_ai", drop_params=True) is None
    else:
        with pytest.raises(BadRequestError, match="cannot disable thinking"):
            AnthropicConfig.map_reasoning_effort("none", "claude-sonnet-5-5", "azure_ai")


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
