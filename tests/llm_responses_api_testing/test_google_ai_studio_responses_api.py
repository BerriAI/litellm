from base_responses_api import BaseResponsesAPITest


class TestGoogleAIStudioResponsesAPITest(BaseResponsesAPITest):
    test_basic_openai_responses_delete_endpoint = None
    test_basic_openai_responses_streaming_delete_endpoint = None
    test_basic_openai_responses_get_endpoint = None
    test_basic_openai_responses_cancel_endpoint = None

    def get_base_completion_call_args(self):
        # litellm.turn_on_debug()
        return {"model": "gemini/gemini-2.5-flash-lite"}
