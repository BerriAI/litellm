import pytest


@pytest.mark.parametrize(
    "end_user_id",
    [{"litellm_metadata": {"user": "test"}}, {"metadata": {"user_id": "test"}}],
)
def test_get_user_from_metadata(end_user_id):
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
        AnthropicPassthroughLoggingHandler,
        PassthroughStandardLoggingPayload,
    )

    passthrough_logging_payload = PassthroughStandardLoggingPayload(
        url="https://api.anthropic.com/v1/messages",
        request_body={**end_user_id},
        response_body={
            "id": "msg_015uSaCZBvu9gUSkAmZtMfxC",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5-20250929",
            "content": [
                {
                    "type": "text",
                    "text": "Now I'll click on the Firefox icon to launch it.",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01TQsF5p7Pf4LGKyLUDDySVr",
                    "name": "computer",
                    "input": {"action": "mouse_move", "coordinate": [24, 36]},
                },
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 2202, "output_tokens": 89},
        },
    )

    response = AnthropicPassthroughLoggingHandler._get_user_from_metadata(
        passthrough_logging_payload=passthrough_logging_payload
    )

    assert response == "test"
