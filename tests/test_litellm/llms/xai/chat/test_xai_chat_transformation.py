"""
Tests for XAI Chat API transformation.

Source: litellm/llms/xai/chat/transformation.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../../../.."))

import pytest

from litellm.llms.xai.chat.transformation import XAIChatConfig


class TestXAIChatTransformation:
    def test_grok_3_mini_supports_reasoning_effort(self):
        config = XAIChatConfig()

        supported_params = config.get_supported_openai_params(model="grok-3-mini")

        assert "reasoning_effort" in supported_params

    @pytest.mark.parametrize(
        "model",
        [
            "grok-3",
            "grok-4",
            "grok-4-fast-reasoning",
            "grok-4-1-fast-reasoning",
            "grok-4-1-fast-non-reasoning",
        ],
    )
    def test_reasoning_effort_excluded_for_other_grok_models(self, model: str):
        config = XAIChatConfig()

        supported_params = config.get_supported_openai_params(model=model)

        assert "reasoning_effort" not in supported_params
