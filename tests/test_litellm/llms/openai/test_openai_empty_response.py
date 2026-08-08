"""
Test for issue #17209: Clearer error when LLM endpoint returns empty response
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.openai.openai import OpenAIChatCompletion
from litellm.llms.openai.common_utils import OpenAIError, try_parse_sse_response_body


class TestEmptyResponseHandling:
    """Test that empty/invalid responses from LLM endpoints produce clear error messages"""

    def test_sync_empty_string_response_raises_clear_error(self):
        """
        Test that when an OpenAI-compatible endpoint returns an empty string,
        we get a clear error instead of "'str' object has no attribute 'model_dump'"
        """
        openai_chat = OpenAIChatCompletion()

        # Mock the raw response to return an empty string from parse()
        mock_raw_response = MagicMock()
        mock_raw_response.headers = {}
        mock_raw_response.parse.return_value = ""  # Empty string response

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.return_value = (
            mock_raw_response
        )

        with pytest.raises(OpenAIError) as exc_info:
            openai_chat.make_sync_openai_chat_completion_request(
                openai_client=mock_client,
                data={"messages": [{"role": "user", "content": "test"}]},
                timeout=30,
                logging_obj=MagicMock(),
            )

        assert "Empty or invalid response from LLM endpoint" in str(exc_info.value)
        assert "Check the reverse proxy or model server configuration" in str(
            exc_info.value
        )

    def test_sync_none_response_raises_clear_error(self):
        """Test that None response also produces a clear error"""
        openai_chat = OpenAIChatCompletion()

        mock_raw_response = MagicMock()
        mock_raw_response.headers = {}
        mock_raw_response.parse.return_value = None

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.return_value = (
            mock_raw_response
        )

        with pytest.raises(OpenAIError) as exc_info:
            openai_chat.make_sync_openai_chat_completion_request(
                openai_client=mock_client,
                data={"messages": [{"role": "user", "content": "test"}]},
                timeout=30,
                logging_obj=MagicMock(),
            )

        assert "Empty or invalid response from LLM endpoint" in str(exc_info.value)

    def test_valid_response_passes_through(self):
        """Test that a valid response with model_dump passes through correctly"""
        openai_chat = OpenAIChatCompletion()

        # Create a mock response that has model_dump (like a real Pydantic model)
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"choices": []}

        mock_raw_response = MagicMock()
        mock_raw_response.headers = {"x-request-id": "123"}
        mock_raw_response.parse.return_value = mock_response

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.return_value = (
            mock_raw_response
        )

        headers, response = openai_chat.make_sync_openai_chat_completion_request(
            openai_client=mock_client,
            data={"messages": [{"role": "user", "content": "test"}]},
            timeout=30,
            logging_obj=MagicMock(),
        )

        assert response == mock_response
        assert headers == {"x-request-id": "123"}

    def test_sync_streaming_response_passes_through_without_model_dump(self):
        """
        Test that streaming responses (which don't have model_dump) pass through
        correctly without raising an error. This validates the fix for VLLM streaming.
        """
        openai_chat = OpenAIChatCompletion()

        # Create a mock response WITHOUT model_dump (like an AsyncStream/Iterator)
        mock_stream = MagicMock(spec=[])  # spec=[] means no attributes

        mock_raw_response = MagicMock()
        mock_raw_response.headers = {"x-request-id": "123"}
        mock_raw_response.parse.return_value = mock_stream

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.return_value = (
            mock_raw_response
        )

        # Key: data has stream=True - this should bypass the model_dump check
        headers, response = openai_chat.make_sync_openai_chat_completion_request(
            openai_client=mock_client,
            data={"messages": [{"role": "user", "content": "test"}], "stream": True},
            timeout=30,
            logging_obj=MagicMock(),
        )

        assert response == mock_stream
        assert headers == {"x-request-id": "123"}

    def test_sync_sse_response_recovers(self):
        """
        Issue #25766: some OpenAI-compatible upstreams ignore stream=false and
        always reply with SSE. The non-streaming code path should recover by
        parsing the SSE chunks and returning an aggregated ModelResponse.
        """
        sse_body = (
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
            '"created":1,"model":"qwen-3.6-plus","choices":[{"index":0,'
            '"delta":{"role":"assistant","content":"Hello"},'
            '"finish_reason":null}]}\n'
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
            '"created":1,"model":"qwen-3.6-plus","choices":[{"index":0,'
            '"delta":{"content":" world"},"finish_reason":"stop"}]}\n'
            "data: [DONE]\n"
        )
        openai_chat = OpenAIChatCompletion()

        mock_raw_response = MagicMock()
        mock_raw_response.headers = {}
        mock_raw_response.parse.return_value = sse_body
        mock_raw_response.text = sse_body

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.return_value = (
            mock_raw_response
        )

        headers, response = openai_chat.make_sync_openai_chat_completion_request(
            openai_client=mock_client,
            data={"messages": [{"role": "user", "content": "test"}]},
            timeout=30,
            logging_obj=MagicMock(),
        )

        assert hasattr(response, "model_dump")
        dumped = response.model_dump()
        assert dumped["choices"][0]["message"]["content"] == "Hello world"
        assert dumped["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_async_sse_response_recovers(self):
        """Async equivalent of test_sync_sse_response_recovers."""
        sse_body = (
            'data: {"id":"chatcmpl-2","object":"chat.completion.chunk",'
            '"created":1,"model":"qwen-3.6-plus","choices":[{"index":0,'
            '"delta":{"role":"assistant","content":"hi"},'
            '"finish_reason":"stop"}]}\n'
            "data: [DONE]\n"
        )
        openai_chat = OpenAIChatCompletion()

        mock_raw_response = MagicMock()
        mock_raw_response.headers = {}
        mock_raw_response.parse.return_value = sse_body
        mock_raw_response.text = sse_body

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create = AsyncMock(
            return_value=mock_raw_response
        )

        headers, response = await openai_chat.make_openai_chat_completion_request(
            openai_aclient=mock_client,
            data={"messages": [{"role": "user", "content": "test"}]},
            timeout=30,
            logging_obj=MagicMock(),
        )

        assert hasattr(response, "model_dump")
        assert response.model_dump()["choices"][0]["message"]["content"] == "hi"

    def test_sync_garbage_text_still_raises(self):
        """
        Non-SSE garbage text must NOT be silently recovered — the original
        "Empty or invalid response" error should still surface.
        """
        openai_chat = OpenAIChatCompletion()

        mock_raw_response = MagicMock()
        mock_raw_response.headers = {}
        mock_raw_response.parse.return_value = "some garbage"
        mock_raw_response.text = "some garbage"

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.return_value = (
            mock_raw_response
        )

        with pytest.raises(OpenAIError) as exc_info:
            openai_chat.make_sync_openai_chat_completion_request(
                openai_client=mock_client,
                data={"messages": [{"role": "user", "content": "test"}]},
                timeout=30,
                logging_obj=MagicMock(),
            )

        assert "Empty or invalid response from LLM endpoint" in str(exc_info.value)


class TestTryParseSSEResponseBody:
    """Unit tests for the try_parse_sse_response_body helper (covers edge-case branches)."""

    def test_blank_lines_between_chunks_are_skipped(self):
        """Blank lines interspersed between SSE chunks must be ignored gracefully."""
        body = (
            'data: {"id":"c1","object":"chat.completion.chunk","created":1,'
            '"model":"m","choices":[{"index":0,"delta":{"role":"assistant",'
            '"content":"hi"},"finish_reason":"stop"}]}\n'
            "\n"  # blank line — exercises the `if not line: continue` branch
            "data: [DONE]\n"
        )
        result = try_parse_sse_response_body(body)
        assert result is not None
        assert result.model_dump()["choices"][0]["message"]["content"] == "hi"

    def test_malformed_json_after_data_prefix_is_skipped(self):
        """A `data:` line with invalid JSON must be skipped; if no valid chunks remain, return None."""
        body = "data: {not valid json}\ndata: [DONE]\n"
        result = try_parse_sse_response_body(body)
        assert result is None

    def test_stream_chunk_builder_exception_returns_none(self):
        """If stream_chunk_builder raises, the helper returns None (caller raises the original error)."""
        body = (
            'data: {"id":"c1","object":"chat.completion.chunk","created":1,'
            '"model":"m","choices":[{"index":0,"delta":{"content":"hi"},'
            '"finish_reason":"stop"}]}\n'
            "data: [DONE]\n"
        )
        with patch("litellm.stream_chunk_builder", side_effect=Exception("boom")):
            result = try_parse_sse_response_body(body)
        assert result is None

    def test_non_model_response_result_returns_none(self):
        """If stream_chunk_builder returns something that isn't a ModelResponse, return None."""
        body = (
            'data: {"id":"c1","object":"chat.completion.chunk","created":1,'
            '"model":"m","choices":[{"index":0,"delta":{"content":"hi"},'
            '"finish_reason":"stop"}]}\n'
            "data: [DONE]\n"
        )
        with patch("litellm.stream_chunk_builder", return_value="unexpected string"):
            result = try_parse_sse_response_body(body)
        assert result is None
