from base_llm_unit_tests import BaseLLMChatTest


class TestGoogleAIStudioGemini(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {"model": "gemini/gemini-1.5-flash"}

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        from litellm.llms.prompt_templates.factory import (
            convert_to_gemini_tool_call_invoke,
        )

        result = convert_to_gemini_tool_call_invoke(tool_call_no_arguments)
        print(result)
