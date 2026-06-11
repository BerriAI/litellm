from litellm.litellm_core_utils.prompt_templates.factory import (
    convert_to_gemini_tool_call_result,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def test_gemini_tool_result_preserves_file_id_only_file_block():
    """
    Responses input_file blocks without inline file_data should still keep their
    file_id when converted through Gemini tool-result handling.
    """

    messages = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output={
            "type": "function_call_output",
            "call_id": "call_file_ref",
            "output": [
                {
                    "type": "input_file",
                    "file_id": "files/report-123",
                    "filename": "report.pdf",
                }
            ],
        }
    )
    last_message_with_tool_calls = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_file_ref",
                "type": "function",
                "index": 0,
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ],
    }

    result = convert_to_gemini_tool_call_result(
        message=messages[0],
        last_message_with_tool_calls=last_message_with_tool_calls,
    )

    assert isinstance(result, dict)
    assert result["function_response"]["name"] == "read_file"
    assert result["function_response"]["response"] == {
        "type": "file",
        "file": {"file_id": "files/report-123", "filename": "report.pdf"},
    }
