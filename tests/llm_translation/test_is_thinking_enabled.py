"""
Tests for issue #26334:
is_thinking_enabled must recognise thinking.type="adaptive" (and any future
non-disabled type) so that forced tool_choice is never set alongside a thinking
parameter — Anthropic rejects that combination with HTTP 400.
"""
import pytest

from litellm.llms.base_llm.chat.transformation import BaseLLMHTTPHandler


# Use a concrete subclass to access the non-abstract method.
_BASE = BaseLLMHTTPHandler()


class TestIsThinkingEnabled:
    def test_type_enabled_returns_true(self):
        assert _BASE.is_thinking_enabled({"thinking": {"type": "enabled", "budget_tokens": 5000}})

    def test_type_adaptive_returns_true(self):
        """Claude Code sends thinking.type="adaptive"; must count as thinking-enabled."""
        assert _BASE.is_thinking_enabled({"thinking": {"type": "adaptive"}})

    def test_type_disabled_returns_false(self):
        assert not _BASE.is_thinking_enabled({"thinking": {"type": "disabled"}})

    def test_thinking_absent_returns_false(self):
        assert not _BASE.is_thinking_enabled({"model": "claude-3"})

    def test_thinking_none_value_returns_false(self):
        assert not _BASE.is_thinking_enabled({"thinking": None})

    def test_reasoning_effort_returns_true(self):
        assert _BASE.is_thinking_enabled({"reasoning_effort": "high"})

    def test_reasoning_effort_none_returns_false(self):
        assert not _BASE.is_thinking_enabled({"reasoning_effort": None})

    def test_unknown_future_type_returns_true(self):
        """Any unrecognised non-disabled type should be treated as enabled."""
        assert _BASE.is_thinking_enabled({"thinking": {"type": "extended"}})


class TestAnthropicTranslationNoForcedToolChoiceWithThinking:
    """
    When thinking is active, map_openai_params must NOT set tool_choice even
    when response_format is also requested (on non-output_format-supporting models).
    """

    def _build_params(self, thinking_type: str):
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig
        config = AnthropicConfig()
        non_default = {
            "thinking": {"type": thinking_type, "budget_tokens": 8000},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "report",
                    "schema": {
                        "type": "object",
                        "properties": {"msg": {"type": "string"}},
                        "required": ["msg"],
                    },
                },
            },
            "max_tokens": 1024,
        }
        optional_params = config.map_openai_params(
            non_default_params=non_default,
            optional_params={},
            model="claude-3-sonnet-20240229",  # non-output_format model
            drop_params=False,
        )
        return optional_params

    def test_type_enabled_no_forced_tool_choice(self):
        params = self._build_params("enabled")
        assert params.get("tool_choice") is None or params.get("tool_choice", {}).get("type") != "tool", (
            f"tool_choice must not be forced when thinking is enabled, got {params.get('tool_choice')}"
        )

    def test_type_adaptive_no_forced_tool_choice(self):
        """Core regression test for #26334."""
        params = self._build_params("adaptive")
        assert params.get("tool_choice") is None or params.get("tool_choice", {}).get("type") != "tool", (
            f"tool_choice must not be forced when thinking is adaptive, got {params.get('tool_choice')}"
        )
