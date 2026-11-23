from typing import Final

import pytest

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    encrypted_reasoning_signature,
)
from litellm.llms.anthropic.pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.bedrock_mantle.messages.transformation import BedrockMantleAnthropicMessagesConfig
from litellm.llms.deepseek.messages.transformation import DeepSeekAnthropicMessagesConfig
from litellm.llms.openai_like.messages.transformation import OpenAILikeAnthropicMessagesConfig
from litellm.types.router import GenericLiteLLMParams


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


_CLAUDE_SIGNED_TURN: Final = {
    "role": "assistant",
    "content": [
        {"type": "thinking", "thinking": "claude plan", "signature": "EqQBCkYIAxgCIkA_anthropic_signed"},
        {"type": "text", "text": "Refined."},
    ],
}


def _open_model_turn(thinking_block: dict[str, object]) -> dict[str, object]:
    return {"role": "assistant", "content": [thinking_block, {"type": "text", "text": "Draft."}]}


@pytest.mark.parametrize(
    "thinking_block",
    [
        {"type": "thinking", "thinking": "open model plan", "signature": ""},
        {"type": "thinking", "thinking": "open model plan", "signature": None},
        {"type": "thinking", "thinking": "open model plan"},
    ],
    ids=["empty", "null", "missing"],
)
def test_unsigned_thinking_from_an_open_model_never_reaches_claude(thinking_block: dict[str, object]) -> None:
    messages: Final = [
        {"role": "user", "content": "Solve it."},
        _open_model_turn(thinking_block),
        {"role": "user", "content": "Refine it."},
        _CLAUDE_SIGNED_TURN,
        {"role": "user", "content": "Thanks."},
    ]

    sent: Final = _transform(messages)["messages"]

    assert sent[1]["content"] == [{"type": "text", "text": "Draft."}]
    assert sent[3] == _CLAUDE_SIGNED_TURN
    assert messages[1]["content"][0] == thinking_block


def test_unsigned_thinking_is_still_replayed_to_an_open_model() -> None:
    messages: Final = [
        {"role": "user", "content": "Solve it."},
        _open_model_turn({"type": "thinking", "thinking": "open model plan", "signature": ""}),
        {"role": "user", "content": "Refine it."},
    ]

    sent: Final = DeepSeekAnthropicMessagesConfig().transform_anthropic_messages_request(
        model="deepseek-v4-pro",
        messages=messages,
        anthropic_messages_optional_request_params={"max_tokens": 1024},
        litellm_params={},
        headers={},
    )["messages"]

    assert sent[1]["content"][0]["thinking"] == "open model plan"


def _pending_tool_turn(opening: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {"role": "user", "content": "List the files."},
        {
            "role": "assistant",
            "content": [*opening, {"type": "tool_use", "id": "toolu_1", "name": "ls", "input": {}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.txt"}]},
    ]


def test_unsigned_thinking_that_opens_the_pending_tool_turn_is_dropped_too() -> None:
    messages: Final = [
        {"role": "user", "content": "Solve it."},
        _open_model_turn({"type": "thinking", "thinking": "open model plan", "signature": ""}),
        *_pending_tool_turn([{"type": "thinking", "thinking": "call ls", "signature": ""}]),
    ]

    sent: Final = _transform(messages)["messages"]

    assert sent[1]["content"] == [{"type": "text", "text": "Draft."}]
    assert sent[3]["content"] == [{"type": "tool_use", "id": "toolu_1", "name": "ls", "input": {}}]
    assert sent[4] == messages[4]


def test_unsigned_thinking_is_dropped_when_the_pending_tool_turn_never_opened_with_thinking() -> None:
    messages: Final = [
        {"role": "user", "content": "Solve it."},
        _open_model_turn({"type": "thinking", "thinking": "open model plan", "signature": ""}),
        *_pending_tool_turn([]),
    ]

    sent: Final = _transform(messages)["messages"]

    assert sent[1]["content"] == [{"type": "text", "text": "Draft."}]
    assert sent[2:] == messages[2:]


@pytest.mark.parametrize(
    ("base_model", "sent_opening"),
    [
        ("anthropic/claude-sonnet-4-5", {"type": "text", "text": "Draft."}),
        ("deepseek/deepseek-v4-pro", {"type": "thinking", "thinking": "open model plan", "signature": ""}),
        (None, {"type": "thinking", "thinking": "open model plan", "signature": ""}),
    ],
    ids=["claude-base-model", "open-base-model", "no-base-model"],
)
def test_an_alias_strips_unsigned_thinking_only_when_its_base_model_names_claude(
    base_model: str | None, sent_opening: dict[str, object]
) -> None:
    messages: Final = [
        {"role": "user", "content": "Solve it."},
        _open_model_turn({"type": "thinking", "thinking": "open model plan", "signature": ""}),
        {"role": "user", "content": "Refine it."},
    ]

    sent: Final = AnthropicMessagesConfig().transform_anthropic_messages_request(
        model="prod-sonnet",
        messages=messages,
        anthropic_messages_optional_request_params={"max_tokens": 1024},
        litellm_params=GenericLiteLLMParams(base_model=base_model),
        headers={},
    )["messages"]

    assert sent[1]["content"][0] == sent_opening


def test_a_non_object_message_item_reaches_claude_as_sent() -> None:
    messages: Final = ["hi", 7, {"role": "user", "content": "Solve it."}]

    assert _transform(messages)["messages"] == messages


@pytest.mark.parametrize(
    "config, model",
    [
        (OpenAILikeAnthropicMessagesConfig(), "anthropic/claude-sonnet-4.5"),
        (BedrockMantleAnthropicMessagesConfig(), "anthropic.claude-mythos-preview"),
    ],
    ids=["anthropic_compatible_server", "bedrock_mantle"],
)
def test_unsigned_thinking_never_reaches_claude_on_any_route(config: AnthropicMessagesConfig, model: str) -> None:
    messages: Final = [
        {"role": "user", "content": "Solve it."},
        _open_model_turn({"type": "thinking", "thinking": "open model plan", "signature": ""}),
        {"role": "user", "content": "Refine it."},
    ]

    sent: Final = config.transform_anthropic_messages_request(
        model=model,
        messages=messages,
        anthropic_messages_optional_request_params={"max_tokens": 1024},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )["messages"]

    assert sent[1]["content"] == [{"type": "text", "text": "Draft."}]
