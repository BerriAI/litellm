from typing import Any, Final, Literal

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


@pytest.mark.parametrize(
    ("data", "call_type", "expected"),
    [
        ({"prompt": "transcribe speech"}, "transcription", "transcribe speech"),
        ({"prompt": "async speech"}, "atranscription", "async speech"),
        ({"prompt": "audio legacy"}, "audio_transcription", "audio legacy"),
        ({"input": "embed async"}, "aembedding", "embed async"),
        ({"input": ["embed", "dings"]}, "embeddings", "embeddings"),
        ({"input": "mod async"}, "amoderation", "mod async"),
        ({"prompt": "gen image async"}, "aimage_generation", "gen image async"),
        ({"prompt": "complete async"}, "atext_completion", "complete async"),
    ],
)
def test_get_formatted_prompt_call_types(data: dict[str, Any], call_type: Any, expected: str) -> None:
    assert get_formatted_prompt(data=data, call_type=call_type) == expected

