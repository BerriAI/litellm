from litellm.llms.anthropic.experimental_pass_through.messages.mid_conversation_system import (
    CONVERTED_SYSTEM_NOTE,
    convert_mid_conversation_system_turns,
)


def test_convert_mid_conversation_system_turns_converts_system_to_user_in_place():
    result = convert_mid_conversation_system_turns(
        [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": [{"type": "text", "text": "Keep it short."}]},
            {"role": "assistant", "content": "Hi."},
        ]
    )

    assert result == (
        {"role": "user", "content": "hi"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": CONVERTED_SYSTEM_NOTE},
                {"type": "text", "text": "Keep it short."},
            ],
        },
        {"role": "assistant", "content": "Hi."},
    )


def test_convert_mid_conversation_system_turns_wraps_string_content():
    result = convert_mid_conversation_system_turns(
        [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "Keep it short."},
        ]
    )

    assert result[1] == {
        "role": "user",
        "content": [
            {"type": "text", "text": CONVERTED_SYSTEM_NOTE},
            {"type": "text", "text": "Keep it short."},
        ],
    }


def test_convert_mid_conversation_system_turns_moves_system_after_tool_result():
    assistant_tool_use = {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}],
    }
    wedged_system = {"role": "system", "content": "Use the corrected result."}
    tool_result = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Rainy"}],
    }

    result = convert_mid_conversation_system_turns([assistant_tool_use, wedged_system, tool_result])

    assert result[0] is assistant_tool_use
    assert result[1] is tool_result
    assert result[2]["role"] == "user"
    assert result[2]["content"][0]["text"] == CONVERTED_SYSTEM_NOTE
