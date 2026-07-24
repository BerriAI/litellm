
from litellm.litellm_core_utils.streaming_chunk_builder_utils import ChunkProcessor


def test_build_base_response_handles_empty_choices():
    """
    Regression test for the IndexError in ChunkProcessor.build_base_response
    when a usage-only chunk has an empty choices list before content chunks
    arrive. build_base_response should default to role='assistant'.
    """
    chunks = [
        {
            "id": "chatcmpl-empty-choices",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4.1-mini",
            "choices": [],
        },
        {
            "id": "chatcmpl-empty-choices",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4.1-mini",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "hello"}}
            ],
        },
    ]

    processor = ChunkProcessor(chunks=chunks)
    response = processor.build_base_response(chunks)

    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == ""


def test_build_base_response_falls_back_to_assistant_when_role_missing():
    """
    When no chunk carries a delta with a role, build_base_response should still
    produce a valid response and default to 'assistant'.
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
