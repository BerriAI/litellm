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


@pytest.mark.parametrize("call_type", ["embedding", "aembedding", "moderation"])
@pytest.mark.parametrize(
    ("input_data", "expected"),
    [
        (None, ""),
        ([], ""),
        ([0.1, 0.2], ""),
        ("hello", "hello"),
        (["hello", " world"], "hello world"),
        ([["hello", " world"], "!"], "hello world!"),
        (["hello", [" world", "!"]], "hello world!"),
    ],
)
def test_embedding_and_moderation_preserve_all_input_text(
    call_type: Literal["embedding", "aembedding", "moderation"], input_data: object, expected: str
) -> None:
    assert get_formatted_prompt(data={"input": input_data}, call_type=call_type) == expected


@pytest.mark.parametrize("call_type", ["transcription", "atranscription", "audio_transcription", "image_generation"])
@pytest.mark.parametrize("prompt", [None, "", "Use these spellings"])
def test_prompt_based_calls_extract_only_text(
    call_type: Literal["transcription", "atranscription", "audio_transcription", "image_generation"], prompt: str | None
) -> None:
    assert get_formatted_prompt(data={"prompt": prompt}, call_type=call_type) == (prompt or "")


@pytest.mark.parametrize("call_type", ["embedding", "aembedding", "transcription", "atranscription"])
def test_calls_without_optional_prompt_text_are_empty(
    call_type: Literal["embedding", "aembedding", "transcription", "atranscription"],
) -> None:
    assert get_formatted_prompt(data={}, call_type=call_type) == ""
