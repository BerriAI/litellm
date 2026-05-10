"""
Tests for DeepSeekChatConfig.map_openai_params thinking/reasoning_effort handling.

Regression tests for https://github.com/BerriAI/litellm/issues/27453
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

import pytest

from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


@pytest.fixture
def config():
    return DeepSeekChatConfig()


class TestReasoningEffortNone:
    """reasoning_effort='none' should disable thinking."""

    def test_reasoning_effort_none_disables_thinking(self, config):
        result = config.map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={},
            model="deepseek-v4-pro",
            drop_params=False,
        )
        assert result["thinking"] == {"type": "disabled"}

    def test_reasoning_effort_high_enables_thinking(self, config):
        result = config.map_openai_params(
            non_default_params={"reasoning_effort": "high"},
            optional_params={},
            model="deepseek-v4-pro",
            drop_params=False,
        )
        assert result["thinking"] == {"type": "enabled"}

    def test_reasoning_effort_low_enables_thinking(self, config):
        result = config.map_openai_params(
            non_default_params={"reasoning_effort": "low"},
            optional_params={},
            model="deepseek-v4-pro",
            drop_params=False,
        )
        assert result["thinking"] == {"type": "enabled"}


class TestThinkingParam:
    """Direct thinking parameter should be passed through."""

    def test_thinking_enabled(self, config):
        result = config.map_openai_params(
            non_default_params={"thinking": {"type": "enabled", "budget_tokens": 5000}},
            optional_params={},
            model="deepseek-v4-pro",
            drop_params=False,
        )
        assert result["thinking"] == {"type": "enabled"}

    def test_thinking_disabled(self, config):
        result = config.map_openai_params(
            non_default_params={"thinking": {"type": "disabled"}},
            optional_params={},
            model="deepseek-v4-pro",
            drop_params=False,
        )
        assert result["thinking"] == {"type": "disabled"}

    def test_thinking_strips_budget_tokens(self, config):
        result = config.map_openai_params(
            non_default_params={"thinking": {"type": "enabled", "budget_tokens": 5000}},
            optional_params={},
            model="deepseek-v4-pro",
            drop_params=False,
        )
        assert "budget_tokens" not in result["thinking"]

    def test_no_thinking_params_leaves_thinking_unset(self, config):
        result = config.map_openai_params(
            non_default_params={"temperature": 0.5},
            optional_params={},
            model="deepseek-v4-pro",
            drop_params=False,
        )
        assert "thinking" not in result
