"""
Test that streaming handles empty choices arrays gracefully.

Some providers send streaming chunks with empty choices arrays (choices: [])
as initialization or ping chunks. LiteLLM should not crash when encountering
these chunks.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.litellm_core_utils.streaming_chunk_builder_utils import ChunkProcessor
from litellm.types.utils import (
    Delta,
    ModelResponseStream,
    StreamingChoices,
)


@pytest.fixture
def stream_wrapper() -> CustomStreamWrapper:
    return CustomStreamWrapper(
        completion_stream=None,
        model="test-model",
        logging_obj=MagicMock(),
        custom_llm_provider="openai",
    )


class TestRaiseOnModelRepetitionEmptyChoices:
    """Test raise_on_model_repetition handles chunks with empty choices."""

    def test_empty_choices_on_last_chunk(self, stream_wrapper):
        """Should not crash when the last chunk has empty choices."""
        chunk_with_content = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="hello", role="assistant"),
                    finish_reason=None,
                )
            ]
        )
        chunk_empty_choices = ModelResponseStream(choices=[])

        stream_wrapper.chunks = [chunk_with_content, chunk_empty_choices]
        stream_wrapper.raise_on_model_repetition()
        assert stream_wrapper._repeated_messages_count == 1

    def test_empty_choices_on_second_to_last_chunk(self, stream_wrapper):
        """Should not crash when the second-to-last chunk has empty choices."""
        chunk_empty_choices = ModelResponseStream(choices=[])
        chunk_with_content = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="hello", role="assistant"),
                    finish_reason=None,
                )
            ]
        )

        stream_wrapper.chunks = [chunk_empty_choices, chunk_with_content]
        stream_wrapper.raise_on_model_repetition()
        assert stream_wrapper._repeated_messages_count == 1

    def test_both_chunks_empty_choices(self, stream_wrapper):
        """Should not crash when both last chunks have empty choices."""
        chunk1 = ModelResponseStream(choices=[])
        chunk2 = ModelResponseStream(choices=[])

        stream_wrapper.chunks = [chunk1, chunk2]
        stream_wrapper.raise_on_model_repetition()
        assert stream_wrapper._repeated_messages_count == 1

    def test_normal_chunks_still_detect_repetition(self, stream_wrapper):
        """Repetition detection still works for normal chunks with content."""
        chunk1 = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="repeated content here!", role="assistant"),
                    finish_reason=None,
                )
            ]
        )
        chunk2 = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="repeated content here!", role="assistant"),
                    finish_reason=None,
                )
            ]
        )

        stream_wrapper.chunks = [chunk1, chunk2]
        stream_wrapper.raise_on_model_repetition()
        assert stream_wrapper._repeated_messages_count == 2

    def test_counter_resets_after_empty_chunk(self, stream_wrapper):
        """Counter should reset to 1 after encountering empty choices."""
        # Simulate accumulated count
        stream_wrapper._repeated_messages_count = 5

        chunk_with_content = ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="hello", role="assistant"),
                    finish_reason=None,
                )
            ]
        )
        chunk_empty = ModelResponseStream(choices=[])

        stream_wrapper.chunks = [chunk_with_content, chunk_empty]
        stream_wrapper.raise_on_model_repetition()
        assert stream_wrapper._repeated_messages_count == 1


class TestBuildBaseResponseEmptyChoices:
    """Test build_base_response handles chunks with empty choices."""

    def test_all_chunks_have_empty_choices(self):
        """Should fallback to role='assistant' when no chunk has valid choices."""
        chunks = [
            {
                "id": "1",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test-model",
                "choices": [],
            },
            {
                "id": "2",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test-model",
                "choices": [],
            },
        ]

        processor = ChunkProcessor(chunks=chunks, messages=None)
        response = processor.build_base_response(chunks)
        assert response["choices"][0]["message"]["role"] == "assistant"

    def test_mixed_empty_and_valid_choices(self):
        """Should find the first chunk with non-empty choices."""
        chunks = [
            {
                "id": "1",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test-model",
                "choices": [],
            },
            {
                "id": "2",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            },
        ]

        processor = ChunkProcessor(chunks=chunks, messages=None)
        response = processor.build_base_response(chunks)
        assert response["choices"][0]["message"]["role"] == "assistant"
