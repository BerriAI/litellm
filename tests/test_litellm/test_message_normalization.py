from litellm.utils import cleanup_none_field_in_message


def test_cleanup_preserves_null_content_for_assistant_tool_calls():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }
        ],
    }

    result = cleanup_none_field_in_message(message)

    assert result == message
    assert "content" in result


def test_cleanup_still_removes_unneeded_null_fields():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": None,
        "function_call": None,
    }

    result = cleanup_none_field_in_message(message)

    assert result == {"role": "assistant"}
