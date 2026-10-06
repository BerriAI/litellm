"""
Tests for Vertex AI Agent Engine transformation.

Tests the request transformation and streaming chunk parsing without making real API calls.
"""


import json
from datetime import datetime
from typing import Final

import httpx
import pytest


from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.llms.vertex_ai.agent_engine.sse_iterator import (
    VertexAgentEngineResponseIterator,
)
from litellm.llms.vertex_ai.agent_engine.transformation import VertexAgentEngineConfig
from litellm.types.utils import LlmProviders
from litellm.utils import CustomStreamWrapper


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


async def test_async_stream_wrapper_without_a_client_posts_through_the_cached_vertex_ai_client():
    api_base: Final = "https://us-central1-aiplatform.googleapis.com/v1/reasoningEngines/123:streamQuery"
    sent_requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent_requests.append(request)
        return httpx.Response(200, text='{"content": {"parts": [{"text": "hi"}], "role": "model"}}')

    cached_client: Final = get_async_httpx_client(llm_provider=LlmProviders.VERTEX_AI, params={})
    cached_client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    messages: Final = [{"role": "user", "content": "hi"}]

    wrapper: Final = await VertexAgentEngineConfig().get_async_custom_stream_wrapper(
        model="agent_engine/123",
        custom_llm_provider="vertex_ai",
        logging_obj=Logging(
            model="agent_engine/123",
            messages=messages,
            stream=True,
            call_type="acompletion",
            start_time=datetime(2026, 1, 1),
            litellm_call_id="call-1",
            function_id="fn-1",
        ),
        api_base=api_base,
        headers={},
        data={"class_method": "stream_query"},
        messages=messages,
        litellm_params={},
    )

    assert type(wrapper) is CustomStreamWrapper
    assert [str(request.url) for request in sent_requests] == [api_base]
    assert json.loads(sent_requests[0].content) == {"class_method": "stream_query"}
