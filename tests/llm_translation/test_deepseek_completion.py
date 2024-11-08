from base_llm_unit_tests import BaseLLMChatTest


# Test implementation
class TestDeepSeekChatCompletion(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {
            "model": "deepseek/deepseek-chat",
        }
