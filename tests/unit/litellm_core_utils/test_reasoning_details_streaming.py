from unittest.mock import MagicMock
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponseStream


def test_is_chunk_non_empty_with_reasoning_details():
    """
    Unit test that chunks carrying only delta.reasoning_details (e.g., OpenRouter
    Anthropic signatures or encrypted reasoning blobs) are treated as non-empty.

    Regression test for https://github.com/BerriAI/litellm/issues/48401
    """
    wrapper = CustomStreamWrapper(
        completion_stream=iter([]),
        model="anthropic/claude-sonnet-4.5",
        custom_llm_provider="openrouter",
        logging_obj=MagicMock(),
    )

    chunk_with_signature = {
        "id": "gen-sig-1",
        "object": "chat.completion.chunk",
        "created": 1741037890,
        "model": "anthropic/claude-sonnet-4.5",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": None,
                    "reasoning_details": [
                        {
                            "type": "reasoning.text",
                            "signature": "sig-abc",
                            "index": 0,
                            "format": "anthropic-claude-v1",
                        }
                    ],
                },
                "finish_reason": None,
            }
        ],
    }
    assert (
        wrapper.is_chunk_non_empty(
            completion_obj={},
            model_response=ModelResponseStream(**chunk_with_signature),
            response_obj={},
        )
        is True
    )

    chunk_with_empty_reasoning_details = {
        "id": "gen-sig-2",
        "object": "chat.completion.chunk",
        "created": 1741037890,
        "model": "anthropic/claude-sonnet-4.5",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": None,
                    "reasoning_details": [],
                },
                "finish_reason": None,
            }
        ],
    }
    assert (
        wrapper.is_chunk_non_empty(
            completion_obj={},
            model_response=ModelResponseStream(**chunk_with_empty_reasoning_details),
            response_obj={},
        )
        is False
    )
