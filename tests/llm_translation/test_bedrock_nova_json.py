from base_llm_unit_tests import BaseLLMChatTest
import pytest

import litellm

class TestBedrockNovaJson(BaseLLMChatTest):
    test_function_calling_with_tool_response = None

    def get_base_completion_call_args(self) -> dict:
        litellm.turn_on_debug()
        return {
            "model": "bedrock/converse/us.amazon.nova-micro-v1:0",
        }

    # @pytest.fixture(autouse=True)
    # def skip_non_json_tests(self, request):
    #     if not "json" in request.function.__name__.lower():
    #         pytest.skip(
    #             f"Skipping non-JSON test: {request.function.__name__} does not contain 'json'"
    #         )
