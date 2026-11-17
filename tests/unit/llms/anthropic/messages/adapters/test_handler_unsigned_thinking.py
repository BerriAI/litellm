from typing import Final

import pytest

from litellm.llms.anthropic.messages.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)

_TOOL_LOOP_FROM_AN_OPEN_MODEL: Final = [
    {"role": "user", "content": "Solve it."},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "open model plan", "signature": ""},
            {"type": "text", "text": "Draft."},
        ],
    },
    {"role": "user", "content": "List the files."},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "call ls", "signature": ""},
            {"type": "tool_use", "id": "toolu_1", "name": "ls", "input": {}},
        ],
    },
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.txt"}]},
]


def _sent_assistant_turns(model: str, provider: str) -> list[dict[str, object]]:
    completion_kwargs, _ = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
        max_tokens=1024,
        messages=_TOOL_LOOP_FROM_AN_OPEN_MODEL,
        model=model,
        thinking={"type": "enabled", "budget_tokens": 1024},
        extra_kwargs={"custom_llm_provider": provider},
    )
    return [m for m in completion_kwargs["messages"] if m["role"] == "assistant"]


@pytest.mark.parametrize(
    "model, provider",
    [
        ("bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0", "bedrock"),
        ("github_copilot/claude-sonnet-4.5", "github_copilot"),
    ],
)
def test_unsigned_thinking_never_reaches_claude_through_the_chat_adapter(model: str, provider: str) -> None:
    turns: Final = _sent_assistant_turns(model, provider)

    assert [turn.get("thinking_blocks") for turn in turns] == [None, None]
    assert [turn.get("reasoning_content") for turn in turns] == [None, None]
    assert turns[1]["tool_calls"][0]["id"] == "toolu_1"


def test_unsigned_thinking_is_still_replayed_to_an_open_model_through_the_chat_adapter() -> None:
    turns: Final = _sent_assistant_turns("deepseek/deepseek-v4-pro", "deepseek")

    assert [turn.get("reasoning_content") for turn in turns] == ["open model plan", "call ls"]
