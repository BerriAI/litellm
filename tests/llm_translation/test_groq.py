


# sys.path.insert(
#     0, os.path.abspath("../..")
# ) # noqa
# )  # Adds the parent directory to the system path

from base_llm_unit_tests import BaseLLMChatTest


class TestGroq(BaseLLMChatTest):
    test_content_list_handling = None
    test_empty_tools = None
    test_web_search = None

    def get_base_completion_call_args(self) -> dict:
        return {
            "model": "groq/openai/gpt-oss-120b",
        }

    def test_tool_call_with_empty_enum_property(self):
        pass
