from base_llm_unit_tests import BaseLLMChatTest
import pytest

import litellm


class TestBedrockTestSuite(BaseLLMChatTest):
    test_content_list_handling = None
    test_empty_tools = None
    test_function_calling_with_tool_response = None

    def get_base_completion_call_args(self) -> dict:
        litellm.turn_on_debug()
        return {
            "model": "bedrock/converse/us.meta.llama3-3-70b-instruct-v1:0",
        }
