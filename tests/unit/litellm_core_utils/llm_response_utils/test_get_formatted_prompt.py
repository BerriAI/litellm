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
