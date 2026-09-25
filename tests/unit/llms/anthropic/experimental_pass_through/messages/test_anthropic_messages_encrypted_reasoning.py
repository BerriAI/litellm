from litellm.litellm_core_utils.prompt_templates.common_utils import (
    encrypted_reasoning_signature,
)
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)


def _transform(messages):
    return AnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-sonnet-4-5",
        messages=messages,
        anthropic_messages_optional_request_params={"max_tokens": 1024},
        litellm_params={},
        headers={},
    )


def test_reasoning_replayed_from_the_responses_bridge_never_reaches_anthropic():
    """Claude Code resumed on a Claude model echoes the thinking blocks a gpt turn produced."""
    messages = [
        {"role": "user", "content": "Solve it."},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "plan", "signature": encrypted_reasoning_signature("gAAAA_1")},
                {"type": "redacted_thinking", "data": encrypted_reasoning_signature("gAAAA_2")},
                {"type": "text", "text": "The answer."},
            ],
        },
        {"role": "user", "content": "And the next one?"},
    ]
    request = _transform(messages)
    assert request["messages"][1]["content"] == [{"type": "text", "text": "The answer."}]
    assert len(messages[1]["content"]) == 3


def test_anthropic_signed_thinking_blocks_are_forwarded_untouched():
    messages = [
        {"role": "user", "content": "Solve it."},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "plan", "signature": "EqQBCkYIAxgCIkA_anthropic_signed"},
                {"type": "redacted_thinking", "data": "EmwKAhgBEgy_anthropic_minted"},
                {"type": "text", "text": "The answer."},
            ],
        },
    ]
    assert _transform(messages)["messages"] == messages
