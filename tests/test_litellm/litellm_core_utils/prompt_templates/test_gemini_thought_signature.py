def test_convert_gemini_tool_call_result_with_thought_signature():
    """
    Test that thought_signature from tool calls is propagated to the
    function_response part when converting tool results for Gemini.

    Gemini 3 thinking models require thoughtSignature fields on tool
    result parts to maintain the thinking chain.
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_gemini_tool_call_result,
    )
    from litellm.types.llms.openai import ChatCompletionToolMessage

    message = ChatCompletionToolMessage(
        role="tool",
        tool_call_id="call_abc",
        content='{"result": "ok"}',
    )
    last_message_with_tool_calls = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "get_data",
                    "arguments": "{}",
                    "provider_specific_fields": {
                        "thought_signature": "dGhvdWdodA==",
                    },
                },
            }
        ],
    }

    result = convert_to_gemini_tool_call_result(
        message=message,
        last_message_with_tool_calls=last_message_with_tool_calls,
        model="gemini-3-flash-preview",
    )
    assert isinstance(result, dict)
    assert "thoughtSignature" in result
    assert result["thoughtSignature"] == "dGhvdWdodA=="


def test_convert_gemini_tool_call_result_without_thought_signature():
    """
    Test that tool results without thought_signature still work correctly
    and do not include the thoughtSignature field.
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_gemini_tool_call_result,
    )
    from litellm.types.llms.openai import ChatCompletionToolMessage

    message = ChatCompletionToolMessage(
        role="tool",
        tool_call_id="call_xyz",
        content='{"value": 42}',
    )
    last_message_with_tool_calls = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "compute", "arguments": "{}"},
            }
        ],
    }

    result = convert_to_gemini_tool_call_result(
        message=message,
        last_message_with_tool_calls=last_message_with_tool_calls,
        model="gemini-2.5-flash",
    )
    assert isinstance(result, dict)
    assert "thoughtSignature" not in result


def test_convert_gemini_tool_call_result_gemini3_inline_data_nested():
    """
    Test that for Gemini 3 models, inline_data (images) in tool results
    are nested inside functionResponse.parts instead of being returned
    as separate parts.

    This prevents leaking internal transition tokens into user-visible text.
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_gemini_tool_call_result,
    )
    from litellm.types.llms.openai import ChatCompletionToolMessage

    message = ChatCompletionToolMessage(
        role="tool",
        tool_call_id="call_img",
        content=[
            {"type": "text", "text": "screenshot captured"},
            {"type": "image_url", "image_url": "data:image/jpeg;base64,/9j/4AAQ"},
        ],
    )
    last_message_with_tool_calls = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_img",
                "type": "function",
                "function": {"name": "take_screenshot", "arguments": "{}"},
            }
        ],
    }

    result = convert_to_gemini_tool_call_result(
        message=message,
        last_message_with_tool_calls=last_message_with_tool_calls,
        model="gemini-3-flash-preview",
    )
    # For Gemini 3, should return a single part with nested parts inside functionResponse
    assert isinstance(result, dict)
    assert "function_response" in result
    assert "parts" in result["function_response"]
    assert any("inline_data" in p for p in result["function_response"]["parts"])
