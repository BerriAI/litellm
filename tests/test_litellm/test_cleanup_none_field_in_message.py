"""`content: null` is the OpenAI-prescribed shape for an assistant tool-call turn."""

from litellm.utils import cleanup_none_field_in_message, validate_and_fix_openai_messages

TOOL_CALLS = [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]


class TestCleanupNoneFieldInMessage:
    def test_keeps_null_content_on_a_tool_call_turn(self):
        cleaned = cleanup_none_field_in_message(
            message={"role": "assistant", "content": None, "tool_calls": TOOL_CALLS}
        )

        assert "content" in cleaned
        assert cleaned["content"] is None

    def test_keeps_null_content_on_a_legacy_function_call_turn(self):
        cleaned = cleanup_none_field_in_message(
            message={
                "role": "assistant",
                "content": None,
                "function_call": {"name": "f", "arguments": "{}"},
            }
        )

        assert "content" in cleaned

    def test_still_strips_other_none_fields_on_that_turn(self):
        # The original purpose of the helper: providers reject e.g. {"function": None}.
        cleaned = cleanup_none_field_in_message(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": TOOL_CALLS,
                "function": None,
                "name": None,
            }
        )

        assert "function" not in cleaned
        assert "name" not in cleaned
        assert "content" in cleaned

    def test_still_strips_none_content_without_tool_calls(self):
        cleaned = cleanup_none_field_in_message(message={"role": "assistant", "content": None})

        assert "content" not in cleaned

    def test_user_message_is_unaffected(self):
        cleaned = cleanup_none_field_in_message(message={"role": "user", "content": None, "name": None})

        assert cleaned == {"role": "user"}


class TestValidateAndFixOpenAIMessages:
    def test_tool_call_turn_survives_the_full_validation_pass(self):
        """End to end through the caller that strips the field."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": TOOL_CALLS},
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]

        fixed = validate_and_fix_openai_messages(messages=messages)

        assert "content" in fixed[1]
        assert fixed[1]["content"] is None
        assert fixed[1]["tool_calls"] == TOOL_CALLS
