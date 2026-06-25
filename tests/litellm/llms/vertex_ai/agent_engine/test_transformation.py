"""
Tests for Vertex AI Agent Engine transformation.

Tests the request transformation and streaming chunk parsing without making real API calls.
"""

import json
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

    def _iterator(self) -> VertexAgentEngineResponseIterator:
        return VertexAgentEngineResponseIterator(
            streaming_response=iter([]),
            sync_stream=True,
        )

    def test_chunk_parser_with_text_content(self):
        """
        Test that chunk_parser correctly extracts text from Vertex Agent Engine response format.
        """
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

        result = self._iterator().chunk_parser(chunk)

        assert (
            result.choices[0].delta.content
            == "Hello! I can help you with financial analysis."
        )
        assert result.choices[0].delta.role == "assistant"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage["prompt_tokens"] == 100
        assert result.usage["completion_tokens"] == 50
        assert result.usage["total_tokens"] == 150

    def test_chunk_parser_without_finish_reason(self):
        """
        Test that chunk_parser handles chunks without finish_reason (intermediate chunks).
        """
        chunk = {
            "content": {
                "parts": [{"text": "Partial response..."}],
                "role": "model",
            },
        }

        result = self._iterator().chunk_parser(chunk)

        assert result.choices[0].delta.content == "Partial response..."
        assert result.choices[0].finish_reason is None
        assert result.usage is None

    def test_chunk_parser_intermediate_function_call_does_not_finish_stream(self):
        """
        Multi-agent Agent Engine streams emit one SSE event per inner action
        (e.g. transfer_to_agent, MCP tool call) and each carries
        ``finish_reason: STOP``. STOP here means "this Gemini turn is done",
        not "the Agent Engine stream is done", so we must NOT surface
        ``finish_reason="stop"`` on these chunks — doing so closes the
        downstream stream wrapper before the final text arrives.

        Regression test for https://github.com/BerriAI/litellm/issues/19121.
        """
        chunk = {
            "content": {
                "parts": [
                    {
                        "function_call": {
                            "id": "adk-redacted",
                            "args": {"agent_name": "analyst"},
                            "name": "transfer_to_agent",
                        }
                    }
                ],
                "role": "model",
            },
            "finish_reason": "STOP",
        }

        result = self._iterator().chunk_parser(chunk)

        assert result.choices[0].finish_reason is None
        tool_calls = result.choices[0].delta.tool_calls
        assert tool_calls is not None and len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "transfer_to_agent"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {
            "agent_name": "analyst"
        }
        assert result.choices[0].delta.content is None

    def test_chunk_parser_intermediate_chunk_no_content_drops_finish_reason(self):
        """
        Some Agent Engine chunks have ``finish_reason: STOP`` but neither text
        nor function_call parts (e.g. thought-only chunks). These must not
        terminate the stream — drop the finish_reason.
        """
        chunk = {
            "content": {
                "parts": [{"thought_signature": "..redacted.."}],
                "role": "model",
            },
            "finish_reason": "STOP",
        }

        result = self._iterator().chunk_parser(chunk)

        assert result.choices[0].finish_reason is None
        assert result.choices[0].delta.content is None
        assert result.choices[0].delta.tool_calls is None

    def test_chunk_parser_concatenates_multiple_text_parts(self):
        """
        If Vertex emits a chunk whose ``content.parts`` contains more than one
        text entry we must surface ALL of them, not just the first.
        """
        chunk = {
            "content": {
                "parts": [
                    {"text": "Hello "},
                    {"text": "world"},
                ],
                "role": "model",
            },
        }

        result = self._iterator().chunk_parser(chunk)

        assert result.choices[0].delta.content == "Hello world"

    def test_chunk_parser_surfaces_safety_finish_reason_without_content(self):
        """
        Hard-stop signals (SAFETY, MAX_TOKENS, ...) must propagate even when
        the chunk has no text/tool_call content. Otherwise a safety-blocked
        Agent Engine response looks identical to a benign intermediate chunk.
        """
        chunk = {
            "content": {"parts": [], "role": "model"},
            "finish_reason": "SAFETY",
        }

        result = self._iterator().chunk_parser(chunk)

        # StreamingChoices normalizes Gemini "SAFETY" → OpenAI "content_filter"
        # via map_finish_reason; the raw uppercase value must be passed through
        # so that downstream consumers can distinguish a safety block from a
        # plain stop.
        assert result.choices[0].finish_reason == "content_filter"

    def test_chunk_parser_surfaces_max_tokens_finish_reason_without_content(self):
        chunk = {
            "content": {"parts": [], "role": "model"},
            "finish_reason": "MAX_TOKENS",
        }

        result = self._iterator().chunk_parser(chunk)

        # StreamingChoices normalizes "max_tokens" → OpenAI "length".
        assert result.choices[0].finish_reason == "length"

    def test_chunk_parser_camelcase_function_call(self):
        """
        Vertex's REST API uses ``functionCall`` (camelCase) — make sure we
        handle that as well as the SDK's ``function_call``.
        """
        chunk = {
            "content": {
                "parts": [
                    {
                        "functionCall": {
                            "id": "adk-1",
                            "args": {},
                            "name": "list_cases",
                        }
                    }
                ],
                "role": "model",
            },
            "finish_reason": "STOP",
        }

        result = self._iterator().chunk_parser(chunk)

        assert result.choices[0].finish_reason is None
        tool_calls = result.choices[0].delta.tool_calls
        assert tool_calls is not None and len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "list_cases"
        assert tool_calls[0]["id"] == "adk-1"
