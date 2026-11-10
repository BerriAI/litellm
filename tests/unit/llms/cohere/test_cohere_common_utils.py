from typing import Final

import pytest

from litellm.llms.cohere.common_utils import ModelResponseIterator


def test_stream_parses_object_chunks_and_rejects_a_chunk_that_is_not_an_object() -> None:
    stream: Final = ModelResponseIterator(
        streaming_response=iter(['{"text": "Hi", "index": 1}', "[1, 2]"]),
        sync_stream=True,
    )

    first: Final = next(stream)

    assert (first["text"], first["index"], first["is_finished"]) == ("Hi", 1, False)
    with pytest.raises(AttributeError, match=r"^'list' object has no attribute 'get'$"):
        next(stream)
