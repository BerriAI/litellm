import os
import sys

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.streaming_chunk_builder_utils import ChunkProcessor


def test_build_base_response_all_chunks_empty_choices():
    """
    Regression test for the IndexError in ChunkProcessor.build_base_response
    when every chunk has choices=[] (e.g. Anthropic usage-only events).
    ``next((c for c in chunks if c.get("choices")), chunk)`` skips every chunk
    (empty list is falsy) and falls back to ``chunk``, whose ``choices`` is also
    ``[]`` — the old code then did ``choices[0]`` and raised IndexError.
    build_base_response should default to role='assistant'.
    """
    chunks = [
        {
            "id": "chatcmpl-all-empty-choices",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4.1-mini",
            "choices": [],
        },
        {
            "id": "chatcmpl-all-empty-choices",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4.1-mini",
            "choices": [],
        },
    ]

    processor = ChunkProcessor(chunks=chunks)
    response = processor.build_base_response(chunks)

    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == ""


def test_build_base_response_falls_back_to_assistant_when_role_missing():
    """
    When a chunk has non-empty choices but its delta omits the role key,
    build_base_response should still produce a valid response and default
    to 'assistant'.
    """
    chunks = [
        {
            "id": "chatcmpl-no-role",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4.1-mini",
            "choices": [
                {"index": 0, "delta": {"content": "hello"}}
            ],
        }
    ]

    processor = ChunkProcessor(chunks=chunks)
    response = processor.build_base_response(chunks)

    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == ""
