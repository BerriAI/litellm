import pytest

from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)


class TestMapThinkingParamBudgetZero:
    """
    Tests for the thinkingBudget=0 handling in _map_thinking_param.

    Gemini 2.5 Flash accepts thinkingBudget=0 to disable thinking.
    Gemini 2.5 Pro rejects it with 400 INVALID_ARGUMENT.
    When a router falls back from Flash (rate-limited) to Pro, the same
    thinking param must not produce thinkingBudget=0 in the Pro request body.
    """

    def test_pro_budget_zero_omits_thinking_budget(self):
        result = VertexGeminiConfig._map_thinking_param(
            thinking_param={"type": "enabled", "budget_tokens": 0},
            model="gemini-2.5-pro",
        )
        assert "thinkingBudget" not in result

    def test_flash_budget_zero_emits_thinking_budget(self):
        result = VertexGeminiConfig._map_thinking_param(
            thinking_param={"type": "enabled", "budget_tokens": 0},
            model="gemini-2.5-flash",
        )
        assert result.get("thinkingBudget") == 0

    def test_flash_nonzero_budget_emits_thinking_budget(self):
        result = VertexGeminiConfig._map_thinking_param(
            thinking_param={"type": "enabled", "budget_tokens": 1024},
            model="gemini-2.5-flash",
        )
        assert result.get("thinkingBudget") == 1024
        assert result.get("includeThoughts") is True

    def test_pro_nonzero_budget_emits_thinking_budget(self):
        result = VertexGeminiConfig._map_thinking_param(
            thinking_param={"type": "enabled", "budget_tokens": 1024},
            model="gemini-2.5-pro",
        )
        assert result.get("thinkingBudget") == 1024
        assert result.get("includeThoughts") is True

    def test_model_supports_thinking_budget_zero_flash(self):
        assert VertexGeminiConfig._model_supports_thinking_budget_zero("gemini-2.5-flash") is True
        assert VertexGeminiConfig._model_supports_thinking_budget_zero("gemini/gemini-2.5-flash") is True
        assert VertexGeminiConfig._model_supports_thinking_budget_zero("gemini-2.5-flash-lite") is True

    def test_model_supports_thinking_budget_zero_pro(self):
        assert VertexGeminiConfig._model_supports_thinking_budget_zero("gemini-2.5-pro") is False
        assert VertexGeminiConfig._model_supports_thinking_budget_zero("gemini/gemini-2.5-pro") is False

    def test_model_supports_thinking_budget_zero_none(self):
        assert VertexGeminiConfig._model_supports_thinking_budget_zero(None) is False
