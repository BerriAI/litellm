"""
Tests for Vertex AI Agent Engine transformation.

Tests the request transformation and streaming chunk parsing without making real API calls.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.vertex_ai.agent_engine.sse_iterator import (
    VertexAgentEngineResponseIterator,
)
from litellm.llms.vertex_ai.agent_engine.transformation import VertexAgentEngineConfig


class TestVertexAgentEngineTransformRequest:
    """Tests for transform_request method."""

    def test_transform_request_basic(self):
        """
        Test that transform_request correctly formats messages into Vertex Agent Engine payload.
        """
        config = VertexAgentEngineConfig()

        messages = [{"role": "user", "content": "Hello, what can you do?"}]
        optional_params = {"user_id": "test-user-123"}
        litellm_params = {}

        result = config.transform_request(
            model="agent_engine/123456789",
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers={},
        )

        assert result["class_method"] == "stream_query"
        assert result["input"]["message"] == "Hello, what can you do?"
        assert result["input"]["user_id"] == "test-user-123"
        assert "session_id" not in result["input"]

    def test_transform_request_with_session_id(self):
        """
        Test that transform_request includes session_id when provided.
        """
        config = VertexAgentEngineConfig()

        messages = [{"role": "user", "content": "Follow up question"}]
        optional_params = {
            "user_id": "test-user-123",
            "session_id": "session-abc-456",
        }
        litellm_params = {}

        result = config.transform_request(
            model="agent_engine/123456789",
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers={},
        )

        assert result["class_method"] == "stream_query"
        assert result["input"]["message"] == "Follow up question"
        assert result["input"]["user_id"] == "test-user-123"
        assert result["input"]["session_id"] == "session-abc-456"


class TestVertexAgentEngineChunkParser:
    """Tests for the streaming chunk parser."""

    def test_chunk_parser_with_text_content(self):
        """
        Test that chunk_parser correctly extracts text from Vertex Agent Engine response format.
        """
        iterator = VertexAgentEngineResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
        )

        chunk = {
            "content": {
                "parts": [{"text": "Hello! I can help you with financial analysis."}],
                "role": "model",
            },
            "finish_reason": "STOP",
            "usage_metadata": {
                "prompt_token_count": 100,
                "candidates_token_count": 50,
                "total_token_count": 150,
            },
        }

        result = iterator.chunk_parser(chunk)

        assert result.choices[0].delta.content == "Hello! I can help you with financial analysis."
        assert result.choices[0].delta.role == "assistant"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage["prompt_tokens"] == 100
        assert result.usage["completion_tokens"] == 50
        assert result.usage["total_tokens"] == 150

    def test_chunk_parser_without_finish_reason(self):
        """
        Test that chunk_parser handles chunks without finish_reason (intermediate chunks).
        """
        iterator = VertexAgentEngineResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
        )

        chunk = {
            "content": {
                "parts": [{"text": "Partial response..."}],
                "role": "model",
            },
        }

        result = iterator.chunk_parser(chunk)

        assert result.choices[0].delta.content == "Partial response..."
        assert result.choices[0].finish_reason is None
        assert result.usage is None

