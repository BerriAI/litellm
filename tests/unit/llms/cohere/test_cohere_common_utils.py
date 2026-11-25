import json
from typing import Final

import pytest

from litellm.llms.cohere.common_utils import CohereV2ModelResponseIterator


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
