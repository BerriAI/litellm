from base_llm_unit_tests import BaseLLMChatTest
import pytest

import litellm


class TestBedrockNovaJson(BaseLLMChatTest):
    test_content_list_handling = None
    test_developer_role_translation = None
    test_empty_tools = None
    test_function_calling_with_tool_response = None
    test_json_response_format_stream = None
    test_tool_call_with_empty_enum_property = None
    test_tool_call_with_property_type_array = None

    def get_base_completion_call_args(self) -> dict:
        litellm.turn_on_debug()
        return {
            "model": "bedrock/converse/us.amazon.nova-micro-v1:0",
        }

    def test_json_response_nested_pydantic_obj(self):
        pass

    def test_json_response_nested_json_schema(self):
        pass



    # @pytest.fixture(autouse=True)
    # def skip_non_json_tests(self, request):
    #     if not "json" in request.function.__name__.lower():
    #         pytest.skip(
    #             f"Skipping non-JSON test: {request.function.__name__} does not contain 'json'"
    #         )
