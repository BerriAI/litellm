from base_llm_unit_tests import BaseLLMChatTest
import pytest

import litellm


class TestBedrockTestSuite(BaseLLMChatTest):
    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        pass

    def get_base_completion_call_args(self) -> dict:
        litellm._turn_on_debug()
        return {
            "model": "bedrock/converse/us.meta.llama3-3-70b-instruct-v1:0",
        }
