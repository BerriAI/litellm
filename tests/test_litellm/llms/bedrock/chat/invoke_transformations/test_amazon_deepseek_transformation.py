import pytest

from litellm.llms.bedrock.chat.invoke_transformations.amazon_deepseek_transformation import (
    AmazonDeepseekR1ResponseIterator,
)

REASONING = "Let me think about this."
ANSWER = "The answer is 4."


def _drain(generations: list[str]) -> tuple[str, str]:
    """Feed one turn through the iterator and return its (reasoning, content) totals."""
    iterator = AmazonDeepseekR1ResponseIterator(streaming_response=None, sync_stream=True)
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for position, generation in enumerate(generations):
        chunk = iterator.chunk_parser(
            {
                "generation": generation,
                "stop_reason": "stop" if position == len(generations) - 1 else None,
                "prompt_token_count": 1,
                "generation_token_count": 1,
            }
        )
        delta = chunk.choices[0].delta
        reasoning_parts.append(getattr(delta, "reasoning_content", None) or "")
        content_parts.append(getattr(delta, "content", None) or "")
    return "".join(reasoning_parts), "".join(content_parts)


@pytest.mark.parametrize(
    "generations",
    [
        pytest.param([REASONING, "</think>", ANSWER], id="marker_alone"),
        pytest.param([REASONING, "</", "think>", ANSWER], id="marker_split_in_two"),
        pytest.param([REASONING, *"</think>", ANSWER], id="marker_split_per_character"),
        pytest.param([REASONING, f"</think>{ANSWER}"], id="marker_glued_to_answer"),
        pytest.param([f"{REASONING}</think>", ANSWER], id="marker_glued_to_reasoning"),
        pytest.param([f"{REASONING}</think>{ANSWER}"], id="whole_turn_in_one_chunk"),
        pytest.param(["Let me think ", "about this.", "</think>", "The answer ", "is 4."], id="both_sides_fragmented"),
    ],
)
def test_end_of_thinking_is_found_however_the_marker_is_chunked(generations):
    """`</think>` is only routed correctly when it lands as a chunk of its own.

    Bedrock does not promise one token per chunk, so the marker can arrive split across chunks or
    glued to the text on either side. Comparing a whole chunk against `"</think>"` misses both, and
    since nothing else flips `has_finished_thinking`, every later chunk stays filed as reasoning and
    the turn reaches the client with no content at all.
    """
    reasoning, content = _drain(generations)

    assert reasoning == REASONING
    assert content == ANSWER


def test_reasoning_and_content_do_not_depend_on_where_the_stream_was_cut():
    """The same generation must assemble identically at every chunk size."""
    whole = f"{REASONING}</think>{ANSWER}"
    results = {
        size: _drain([whole[i : i + size] for i in range(0, len(whole), size)])
        for size in (1, 2, 3, 5, 8, 13, len(whole))
    }

    assert set(results.values()) == {(REASONING, ANSWER)}


def test_text_resembling_the_marker_is_not_swallowed():
    """Holding back a possible marker prefix must not eat text that never completes one."""
    assert _drain(["a < b and c </ d", "</think>", ANSWER]) == ("a < b and c </ d", ANSWER)


def test_unterminated_thinking_is_released_at_the_end_of_the_stream():
    """A turn that never closes its thinking still owes the client every token it produced."""
    assert _drain(["still thinking ", "</thi"]) == ("still thinking </thi", "")


def test_marker_in_the_answer_is_left_alone_once_thinking_ended():
    """Only the first `</think>` ends the thinking phase; a later one is ordinary content."""
    assert _drain([REASONING, "</think>", "write </think> to close the block"]) == (
        REASONING,
        "write </think> to close the block",
    )


def test_usage_and_finish_reason_still_come_from_the_chunk():
    iterator = AmazonDeepseekR1ResponseIterator(streaming_response=None, sync_stream=True)

    chunk = iterator.chunk_parser(
        {
            "generation": "done",
            "stop_reason": "stop",
            "prompt_token_count": 11,
            "generation_token_count": 7,
        }
    )

    assert chunk.choices[0].finish_reason == "stop"
    assert chunk.usage["prompt_tokens"] == 11
    assert chunk.usage["completion_tokens"] == 7
    assert chunk.usage["total_tokens"] == 18
