"""Regression tests for Gemini tool-result media shapes (issue #47895).

Pre-Gemini 3 models need sibling inline_data parts; Gemini 3+ keeps media
nested inside functionResponse.parts.
"""

def test_convert_gemini_tool_call_result_sibling_media_for_pre_gemini_3():
    """
    Pre-Gemini 3 models reject media nested in function_response.parts with
    "Multimodal function responses are not supported for this model". They must
    receive the pre-v1.89 sibling shape: a function_response part followed by
    inline_data parts. Gemini 3+ keeps the nested shape.
    Fixes: https://github.com/BerriAI/litellm/issues/47895
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_to_gemini_tool_call_result,
    )
    from litellm.types.llms.openai import ChatCompletionToolMessage

    message = ChatCompletionToolMessage(
        role="tool",
        tool_call_id="call_123",
        content=[
            {"type": "text", "text": "Screenshot taken."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo"}},
        ],
    )
    last_message_with_tool_calls = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "index": 0,
                "function": {"name": "screenshot", "arguments": "{}"},
            }
        ],
    }

    # Pre-Gemini 3 (forward_function_call_id=False): sibling inline_data parts.
    result = convert_to_gemini_tool_call_result(
        message=message,
        last_message_with_tool_calls=last_message_with_tool_calls,
        forward_function_call_id=False,
    )
    assert isinstance(result, list), "expected function_response + sibling media parts"
    assert len(result) == 2
    function_response = result[0]["function_response"]
    assert "parts" not in function_response, "pre-Gemini 3 must not nest media in function_response.parts"
    assert function_response["response"] == {"content": "Screenshot taken."}
    assert result[1] == {"inline_data": {"mime_type": "image/png", "data": "iVBORw0KGgo"}}

    # Gemini 3+ (forward_function_call_id=True): nested shape.
    result_supported = convert_to_gemini_tool_call_result(
        message=message,
        last_message_with_tool_calls=last_message_with_tool_calls,
        forward_function_call_id=True,
    )
    inline_parts = _get_gemini_function_response_inline_data_parts(result_supported)
    assert len(inline_parts) == 1

    # Text-only tool results are unchanged on both shapes.
    text_only = ChatCompletionToolMessage(
        role="tool",
        tool_call_id="call_123",
        content="Screenshot taken.",
    )
    for flag in (False, True):
        plain = convert_to_gemini_tool_call_result(
            message=text_only,
            last_message_with_tool_calls=last_message_with_tool_calls,
            forward_function_call_id=flag,
        )
        assert isinstance(plain, dict)
        assert "parts" not in plain["function_response"]


def test_gemini_convert_messages_gates_multimodal_function_response_by_model():
    """
    End to end through gemini_convert_messages_with_history: a tool result with
    an image goes out as sibling inline_data on Gemini 2.x (pre-v1.89 shape it
    accepts) and nested in function_response.parts on Gemini 3+.
    Fixes: https://github.com/BerriAI/litellm/issues/47895
    """
    from litellm.llms.vertex_ai.gemini.transformation import (
        gemini_convert_messages_with_history,
    )

    messages = [
        {"role": "user", "content": "Take a screenshot, then describe it."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "screenshot", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "text", "text": "Screenshot taken."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo"}},
            ],
        },
    ]

    def _tool_turn_parts(contents):
        for content in contents:
            for part in content.get("parts", []):
                if isinstance(part, dict) and "function_response" in part:
                    return content["parts"]
        return []

    # Gemini 2.x: sibling inline_data next to the function_response, no nesting.
    old_parts = _tool_turn_parts(
        gemini_convert_messages_with_history(messages=list(messages), model="gemini-2.5-flash-lite")
    )
    old_fr = next(p["function_response"] for p in old_parts if "function_response" in p)
    assert "parts" not in old_fr, "Gemini 2.x must not receive nested multimodal function responses"
    assert old_fr["response"] == {"content": "Screenshot taken."}
    siblings = [p["inline_data"] for p in old_parts if "inline_data" in p]
    assert siblings == [{"mime_type": "image/png", "data": "iVBORw0KGgo"}]

    # Gemini 3+: media nested in function_response.parts, no siblings.
    new_parts = _tool_turn_parts(
        gemini_convert_messages_with_history(messages=list(messages), model="gemini-3.8-flash")
    )
    new_fr = next(p["function_response"] for p in new_parts if "function_response" in p)
    nested = new_fr.get("parts") or []
    assert any("inline_data" in part for part in nested), "Gemini 3+ must keep tool-result media nested"
    assert not any("inline_data" in p for p in new_parts), "Gemini 3+ must not emit sibling inline_data"


def test_gemini_convert_messages_sibling_file_block_for_pre_gemini_3():
    """
    File blocks (e.g. PDFs) in tool results take the same inline_data path:
    sibling parts on Gemini 2.x, nested on Gemini 3+.
    Fixes: https://github.com/BerriAI/litellm/issues/47895
    """
    from litellm.llms.vertex_ai.gemini.transformation import (
        gemini_convert_messages_with_history,
    )

    pdf_data_url = "data:application/pdf;base64,JVBERi0xLjQK"
    messages = [
        {"role": "user", "content": "Read this PDF."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "text", "text": "PDF contents follow."},
                {"type": "file", "file": {"file_data": pdf_data_url}},
            ],
        },
    ]

    def _tool_turn_parts(contents):
        for content in contents:
            for part in content.get("parts", []):
                if isinstance(part, dict) and "function_response" in part:
                    return content["parts"]
        return []

    old_parts = _tool_turn_parts(
        gemini_convert_messages_with_history(messages=list(messages), model="gemini-2.5-flash-lite")
    )
    old_fr = next(p["function_response"] for p in old_parts if "function_response" in p)
    assert "parts" not in old_fr
    siblings = [p["inline_data"] for p in old_parts if "inline_data" in p]
    assert siblings == [{"mime_type": "application/pdf", "data": "JVBERi0xLjQK"}]

    new_parts = _tool_turn_parts(
        gemini_convert_messages_with_history(messages=list(messages), model="gemini-3.8-flash")
    )
    new_fr = next(p["function_response"] for p in new_parts if "function_response" in p)
    nested = new_fr.get("parts") or []
    assert any(p.get("inline_data", {}).get("mime_type") == "application/pdf" for p in nested)

