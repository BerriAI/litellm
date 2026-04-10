"""
Tests for the generic ModelResponseIterator in litellm/llms/databricks/streaming_utils.py.

Verifies that reasoning_content is correctly extracted from streaming chunks,
fixing the issue where OpenAI-like providers (Watsonx, Cerebras, etc.) lost
reasoning_content during streaming.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import MagicMock

from litellm.llms.databricks.streaming_utils import ModelResponseIterator


class TestModelResponseIteratorReasoningContent:
    """Test reasoning_content extraction in the generic chunk_parser."""

    def test_chunk_parser_extracts_reasoning_content(self):
        """Verify reasoning_content is extracted from a streaming chunk delta."""
        handler = ModelResponseIterator(
            streaming_response=MagicMock(), sync_stream=True
        )

        chunk = {
            "id": "chatcmpl-test-123",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-oss-120b",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "Let me think about this step by step.",
                        "content": None,
                    },
                    "finish_reason": None,
                }
            ],
        }

        result = handler.chunk_parser(chunk)

        assert result["reasoning_content"] == "Let me think about this step by step."
        assert result["text"] == ""

    def test_chunk_parser_reasoning_content_none_when_absent(self):
        """Verify reasoning_content is None when not present in the chunk."""
        handler = ModelResponseIterator(
            streaming_response=MagicMock(), sync_stream=True
        )

        chunk = {
            "id": "chatcmpl-test-456",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-oss-120b",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello world"},
                    "finish_reason": None,
                }
            ],
        }

        result = handler.chunk_parser(chunk)

        assert result["reasoning_content"] is None
        assert result["text"] == "Hello world"

    def test_chunk_parser_both_content_and_reasoning(self):
        """Verify both text and reasoning_content can be extracted simultaneously."""
        handler = ModelResponseIterator(
            streaming_response=MagicMock(), sync_stream=True
        )

        chunk = {
            "id": "chatcmpl-test-789",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-oss-120b",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "The answer is 42.",
                        "reasoning_content": "Computing...",
                    },
                    "finish_reason": None,
                }
            ],
        }

        result = handler.chunk_parser(chunk)

        assert result["text"] == "The answer is 42."
        assert result["reasoning_content"] == "Computing..."
