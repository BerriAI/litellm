import os
import sys


import pytest

# sys.path.insert(
#     0, os.path.abspath("../..")
# ) # noqa
# )  # Adds the parent directory to the system path

import litellm
from base_llm_unit_tests import BaseLLMChatTest
from litellm.llms.groq.chat.transformation import (
    GroqChatConfig,
    GroqChatCompletionStreamingHandler,
)

class TestGroq(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {
            "model": "groq/llama-3.3-70b-versatile",
        }

    def test_tool_call_with_empty_enum_property(self):
        pass

    @pytest.mark.parametrize(
        "model",
        ["groq/qwen/qwen3-32b", "groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b"],
    )
    def test_reasoning_effort_in_supported_params(self, model):
        """Test that reasoning_effort is in the list of supported parameters for Groq"""
        supported_params = GroqChatConfig().get_supported_openai_params(model=model)
        assert "reasoning_effort" in supported_params
