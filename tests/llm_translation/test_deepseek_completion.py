from base_llm_unit_tests import BaseLLMChatTest
import pytest


# Test implementations
@pytest.mark.skip(reason="Deepseek API is hanging")
class TestDeepSeekChatCompletion(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {
            "model": "deepseek/deepseek-chat",
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    def test_multilingual_requests(self):
        """
        DeepSeek API raises a 400 BadRequest error when the request contains invalid utf-8 sequences.

        Todo: if litellm.modify_params is True ensure it's a valid utf-8 sequence
        """
        pass
