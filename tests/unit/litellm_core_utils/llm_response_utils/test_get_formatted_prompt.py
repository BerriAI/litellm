from typing import Final, Literal

import pytest

from litellm.litellm_core_utils.llm_response_utils.get_formatted_prompt import (
    get_formatted_prompt,
)


@pytest.mark.parametrize("call_type", ["acompletion", "completion"])
def test_null_tool_calls_are_skipped(call_type: Literal["acompletion", "completion"]) -> None:
    data: Final = {
        "messages": [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong", "tool_calls": None},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "f", "arguments": '{"x":1}'}}],
            },
        ]
    }

    assert get_formatted_prompt(data=data, call_type=call_type) == 'pingpong{"x":1}'


@pytest.mark.parametrize("call_type", ["embedding", "moderation"])
def test_non_string_list_input_is_skipped(call_type: Literal["embedding", "moderation"]) -> None:
    """/v1/embeddings accepts token id arrays, so the input list is not always text."""
    data: Final = {"input": ["scan me", 1234, ["nested"], "and me"]}

    assert get_formatted_prompt(data=data, call_type=call_type) == "scan meand me"


def test_text_completion_list_prompt_is_joined() -> None:
    """/v1/completions accepts a prompt array, so the hook must still get a string."""
    data: Final = {"prompt": ["scan me", 1234, ["nested"], "and me"]}

    assert get_formatted_prompt(data=data, call_type="text_completion") == "scan meand me"
