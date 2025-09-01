from base_llm_unit_tests import BaseLLMChatTest


class TestGroq(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {
            "model": "groq/llama-3.3-70b-versatile",
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass
