import json
from typing import Final

import pytest

from litellm.llms.cohere.common_utils import CohereV2ModelResponseIterator, ModelResponseIterator


def test_stream_parses_object_chunks_and_rejects_a_chunk_that_is_not_an_object() -> None:
    stream: Final = ModelResponseIterator(
        streaming_response=iter(['{"text": "Hi", "index": 1}', "[1, 2]"]),
        sync_stream=True,
    )

    first: Final = next(stream)

    assert (first["text"], first["index"], first["is_finished"]) == ("Hi", 1, False)
    with pytest.raises(AttributeError, match=r"^'list' object has no attribute 'get'$"):
        next(stream)


def test_v2_stream_parses_object_chunks_and_rejects_a_chunk_that_is_not_an_object() -> None:
    stream: Final = CohereV2ModelResponseIterator(
        streaming_response=iter(
            [
                '{"type": "content-delta", "index": 0, "delta": {"message": {"content": {"text": "Hi"}}}}',
                "[1, 2]",
            ]
        ),
        sync_stream=True,
    )

    first: Final = next(stream)

    assert (first.choices[0].delta.content, first.choices[0].finish_reason) == ("Hi", None)
    with pytest.raises(RuntimeError):
        next(stream)


@pytest.mark.parametrize(
    ("content", "streamed"),
    [
        ({"text": "Hi"}, ("Hi", None)),
        ({"thinking": "Say hi."}, ("", "Say hi.")),
    ],
)
def test_v2_stream_splits_answer_text_from_thinking(content: object, streamed: tuple[str, str | None]) -> None:
    stream: Final = CohereV2ModelResponseIterator(
        streaming_response=iter(
            [json.dumps({"type": "content-delta", "index": 0, "delta": {"message": {"content": content}}})]
        ),
        sync_stream=True,
    )

    delta: Final = next(stream).choices[0].delta

    assert (delta.content, getattr(delta, "reasoning_content", None)) == streamed
