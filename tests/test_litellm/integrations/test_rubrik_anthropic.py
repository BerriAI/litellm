"""
Tests for the Rubrik LiteLLM plugin - Anthropic format.

Covers tool blocking for Anthropic non-streaming and streaming responses,
format detection, and pass-through behavior for text-only responses.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

# Set environment variable BEFORE importing the plugin
os.environ["RUBRIK_WEBHOOK_URL"] = "http://localhost:8080"

# Patch asyncio.create_task to avoid event loop issues during import
with patch("asyncio.create_task", Mock()):
    from litellm.integrations.rubrik import LLMResponseFormat, RubrikLogger  # noqa: E402


@pytest.fixture
def mock_env():
    """Set up environment variables for testing."""
    with patch.dict(
        os.environ,
        {
            "RUBRIK_WEBHOOK_URL": "http://localhost:8080",
            "RUBRIK_API_KEY": "test-api-key",
        },
    ):
        yield


@pytest.fixture
def handler(mock_env):
    """Create a RubrikLogger instance for testing."""
    with patch("asyncio.create_task", Mock()):
        return RubrikLogger()


def create_anthropic_request_data() -> Dict[str, Any]:
    """Create request_data dict for Anthropic format."""
    return {"proxy_server_request": {"url": "https://api.anthropic.com/v1/messages"}}


def create_openai_request_data() -> Dict[str, Any]:
    """Create request_data dict for OpenAI format."""
    return {"proxy_server_request": {"url": "https://api.openai.com/v1/chat/completions"}}


@pytest.mark.asyncio
async def test_anthropic_all_tools_allowed(handler):
    """Test Anthropic format when all tools are allowed."""

    sample_file = Path(__file__).parent / "rubrik_test_sample_data" / "anthropic_tool_call_response.json"
    with open(sample_file) as f:
        response_dict = json.load(f)

    async def mock_post(*args: Any, **kwargs: Any) -> Mock:
        request_json = kwargs.get("json", {})
        mock_response = Mock()
        mock_response.json.return_value = request_json
        mock_response.raise_for_status = Mock()
        return mock_response

    with patch.object(handler.tool_blocking_client, "post", new=mock_post):
        result = await handler.async_post_call_success_hook(
            data=create_anthropic_request_data(),
            user_api_key_dict={},
            response=response_dict,
        )

        assert result is response_dict, "Should return the same dict object"

        tool_uses = [
            block
            for block in result["content"]
            if getattr(block, "type", block.get("type") if isinstance(block, dict) else None) == "tool_use"
        ]
        assert len(tool_uses) == 1, "Should have 1 tool_use block"
        tool_name = getattr(tool_uses[0], "name", tool_uses[0].get("name") if isinstance(tool_uses[0], dict) else None)
        assert tool_name == "get_weather"


@pytest.mark.asyncio
async def test_anthropic_dict_response_with_blocking(handler):
    """Test Anthropic format dict response when tools are blocked."""

    sample_file = Path(__file__).parent / "rubrik_test_sample_data" / "anthropic_tool_call_response.json"
    with open(sample_file) as f:
        response_dict = json.load(f)

    async def mock_post(*args: Any, **kwargs: Any) -> Mock:
        request_json = kwargs.get("json", {})
        mock_response = Mock()
        blocked_response = dict(request_json)
        blocked_response["choices"][0]["message"]["tool_calls"] = []
        blocked_response["choices"][0]["message"]["content"] = "Tool blocked by security policy"
        blocked_response["choices"][0]["finish_reason"] = "stop"
        mock_response.json.return_value = blocked_response
        mock_response.raise_for_status = Mock()
        return mock_response

    with patch.object(handler.tool_blocking_client, "post", new=mock_post):
        result = await handler.async_post_call_success_hook(
            data=create_anthropic_request_data(),
            user_api_key_dict={},
            response=response_dict,
        )

        assert result is response_dict, "Should return the same dict object"

        tool_uses = [
            block
            for block in result["content"]
            if getattr(block, "type", block.get("type") if isinstance(block, dict) else None) == "tool_use"
        ]
        assert len(tool_uses) == 0, "Should have no tool_use blocks"

        text_blocks = [
            block
            for block in result["content"]
            if getattr(block, "type", block.get("type") if isinstance(block, dict) else None) == "text"
        ]
        assert any(
            "Tool blocked by security policy"
            in getattr(block, "text", block.get("text", "") if isinstance(block, dict) else "")
            for block in text_blocks
        ), "Should have explanation text"

        assert result["stop_reason"] == "end_turn", "stop_reason should be updated"


@pytest.mark.asyncio
async def test_anthropic_text_only_response(handler):
    """Test Anthropic format with text-only response (no tool calls)."""

    sample_file = Path(__file__).parent / "rubrik_test_sample_data" / "anthropic_text_response.json"
    with open(sample_file) as f:
        response_dict = json.load(f)

    async def mock_post(*args: Any, **kwargs: Any) -> Mock:
        request_json = kwargs.get("json", {})
        mock_response = Mock()
        mock_response.json.return_value = request_json
        mock_response.raise_for_status = Mock()
        return mock_response

    with patch.object(handler.tool_blocking_client, "post", new=mock_post):
        result = await handler.async_post_call_success_hook(
            data=create_anthropic_request_data(),
            user_api_key_dict={},
            response=response_dict,
        )

        assert result is response_dict, "Should return the same dict object"

        first_block = result["content"][0]
        block_type = getattr(first_block, "type", first_block.get("type") if isinstance(first_block, dict) else None)
        block_text = getattr(first_block, "text", first_block.get("text", "") if isinstance(first_block, dict) else "")
        assert block_type == "text"
        assert "Claude" in block_text
        assert result["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_anthropic_multiple_tool_calls_allowed(handler):
    """Test Anthropic format with multiple tool calls when all tools are allowed."""

    sample_file = Path(__file__).parent / "rubrik_test_sample_data" / "anthropic_multiple_tool_call_response.json"
    with open(sample_file) as f:
        response_dict = json.load(f)

    async def mock_post(*args: Any, **kwargs: Any) -> Mock:
        request_json = kwargs.get("json", {})
        mock_response = Mock()
        mock_response.json.return_value = request_json
        mock_response.raise_for_status = Mock()
        return mock_response

    with patch.object(handler.tool_blocking_client, "post", new=mock_post):
        result = await handler.async_post_call_success_hook(
            data=create_anthropic_request_data(),
            user_api_key_dict={},
            response=response_dict,
        )

        assert result is response_dict, "Should return the same dict object"

        tool_uses = [
            block
            for block in result["content"]
            if getattr(block, "type", block.get("type") if isinstance(block, dict) else None) == "tool_use"
        ]
        assert len(tool_uses) == 2, "Should have 2 tool_use blocks"

        for tool_use in tool_uses:
            tool_name = getattr(tool_use, "name", tool_use.get("name") if isinstance(tool_use, dict) else None)
            assert tool_name == "get_weather"

        locations = []
        for tool_use in tool_uses:
            tool_input = getattr(tool_use, "input", tool_use.get("input") if isinstance(tool_use, dict) else {})
            location = (
                tool_input.get("location") if isinstance(tool_input, dict) else getattr(tool_input, "location", None)
            )
            locations.append(location)

        assert "Portland, OR" in locations, "Should have Portland, OR location"
        assert "San Francisco, CA" in locations, "Should have San Francisco, CA location"


@pytest.mark.asyncio
async def test_anthropic_multiple_tool_calls_blocked(handler):
    """Test Anthropic format with multiple tool calls when all tools are blocked."""

    sample_file = Path(__file__).parent / "rubrik_test_sample_data" / "anthropic_multiple_tool_call_response.json"
    with open(sample_file) as f:
        response_dict = json.load(f)

    async def mock_post(*args: Any, **kwargs: Any) -> Mock:
        request_json = kwargs.get("json", {})
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": request_json.get("id"),
            "object": "chat.completion",
            "created": request_json.get("created"),
            "model": request_json.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The following tool calls were blocked by policy:\n- get_weather (Portland, "
                        "OR): Policy violation\n- get_weather (San Francisco, CA): Policy violation",
                        "tool_calls": [],
                    },
                    "finish_reason": "stop",
                },
            ],
        }
        mock_response.raise_for_status = Mock()
        return mock_response

    with patch.object(handler.tool_blocking_client, "post", new=mock_post):
        result = await handler.async_post_call_success_hook(
            data=create_anthropic_request_data(),
            user_api_key_dict={},
            response=response_dict,
        )

        assert result is response_dict, "Should return the same dict object"

        tool_uses = [
            block
            for block in result["content"]
            if getattr(block, "type", block.get("type") if isinstance(block, dict) else None) == "tool_use"
        ]
        assert len(tool_uses) == 0, "Should have 0 tool_use blocks (all blocked)"

        assert result["stop_reason"] == "end_turn", "Should have stop_reason 'end_turn' when all tools blocked"

        text_blocks = [
            block
            for block in result["content"]
            if getattr(block, "type", block.get("type") if isinstance(block, dict) else None) == "text"
        ]
        assert len(text_blocks) > 0, "Should have at least one text block with explanation"

        text_content = None
        for block in text_blocks:
            text = getattr(block, "text", block.get("text") if isinstance(block, dict) else None)
            if text and "blocked by policy" in text.lower():
                text_content = text
                break

        assert text_content is not None, "Should have explanation text"
        assert "get_weather" in text_content, "Explanation should mention get_weather"
        assert "blocked by policy" in text_content.lower(), "Explanation should mention blocking"


@pytest.mark.asyncio
class TestAnthropicRoundTrip:
    """Test OpenAI↔Anthropic format conversion edge cases."""

    async def test_openai_dict_to_anthropic_without_usage_key(self, handler):
        """Test that _openai_dict_to_anthropic_response doesn't crash when usage is missing.

        translate_openai_response_to_anthropic accesses response.usage
        unconditionally, so the conversion must provide a default.
        """
        openai_dict: Dict[str, Any] = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 123,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                        "tool_calls": [],
                    },
                    "finish_reason": "stop",
                },
            ],
            # No "usage" key — this is what the tool-blocking service may return
        }
        original_response: Dict[str, Any] = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "bad_tool", "input": {}}],
            "stop_reason": "tool_use",
            "model": "test-model",
        }

        # Should not raise AttributeError
        handler._openai_dict_to_anthropic_response(openai_dict, original_response)

        assert original_response["stop_reason"] == "end_turn"
        assert isinstance(original_response["content"], list)

    async def test_openai_dict_to_anthropic_with_usage_key(self, handler):
        """Test that _openai_dict_to_anthropic_response works normally when usage is present."""
        openai_dict: Dict[str, Any] = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 123,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                        "tool_calls": [],
                    },
                    "finish_reason": "stop",
                },
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        original_response: Dict[str, Any] = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "original"}],
            "stop_reason": "end_turn",
            "model": "test-model",
        }

        handler._openai_dict_to_anthropic_response(openai_dict, original_response)

        assert original_response["stop_reason"] == "end_turn"
        assert isinstance(original_response["content"], list)


@pytest.mark.asyncio
class TestAnthropicStreaming:
    """Test Anthropic streaming format support."""

    @staticmethod
    async def _decode_sse(sse_bytes: bytes) -> dict:
        """Decode SSE bytes to dict using the main class parser."""

        async def single_chunk_stream():
            yield sse_bytes

        chunk_dict: dict
        async for chunk_dict in RubrikLogger._parse_anthropic_sse_stream(single_chunk_stream()):
            return chunk_dict

        raise ValueError(f"Invalid SSE format: {sse_bytes.decode('utf-8')}")

    @staticmethod
    async def _create_sse_stream_from_file(file_path: Path):
        """Read an SSE file and yield raw bytes for each event."""
        with open(file_path) as f:
            content = f.read()
        for event in content.split("\n\n"):
            if event.strip():
                yield (event + "\n\n").encode("utf-8")

    def _filter_chunks_by_type(self, chunks: list, chunk_type: str) -> list:
        """Filter chunks by their 'type' field."""
        return [c for c in chunks if c.get("type") == chunk_type]

    def _find_tool_use_blocks(self, chunks: list) -> list:
        """Find content_block_start chunks that contain tool_use blocks."""
        return [
            c
            for c in chunks
            if c.get("type") == "content_block_start" and c.get("content_block", {}).get("type") == "tool_use"
        ]

    def _find_text_deltas(self, chunks: list) -> list:
        """Find content_block_delta chunks that contain text_delta."""
        return [
            c
            for c in chunks
            if c.get("type") == "content_block_delta" and c.get("delta", {}).get("type") == "text_delta"
        ]

    def _extract_full_text(self, chunks: list) -> str:
        """Extract and concatenate all text from text_delta chunks."""
        text_deltas = self._find_text_deltas(chunks)
        return "".join(c["delta"]["text"] for c in text_deltas)

    async def test_openai_format_detection(self, handler):
        """Test that OpenAI streaming format is correctly detected."""
        openai_request_data = create_openai_request_data()
        assert handler._detect_llm_response_format(openai_request_data) == LLMResponseFormat.OPENAI

        anthropic_request_data = create_anthropic_request_data()
        assert handler._detect_llm_response_format(anthropic_request_data) == LLMResponseFormat.ANTHROPIC

        unknown_request_data: Dict[str, Any] = {"id": "some-id", "data": "some-data"}
        assert handler._detect_llm_response_format(unknown_request_data) == LLMResponseFormat.UNKNOWN

    async def test_anthropic_streaming_text_only(self, handler):
        """Test Anthropic streaming with text-only response (no tool calls)."""

        async def mock_stream():
            for m in [
                {"type": "message_start", "message": {"id": "msg_123", "content": [], "stop_reason": None}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}},
                {"type": "message_stop"},
            ]:
                yield RubrikLogger._encode_anthropic_chunk_to_sse(m)

        chunks = []
        async for chunk in handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict={},
            response=mock_stream(),
            request_data=create_anthropic_request_data(),
        ):
            chunks.append(await self._decode_sse(chunk))

        assert len(chunks) == 7
        assert chunks[0]["type"] == "message_start"
        assert chunks[2]["delta"]["text"] == "Hello"
        assert chunks[3]["delta"]["text"] == " world"
        assert chunks[6]["type"] == "message_stop"

    async def test_anthropic_streaming_tool_call_blocked(self, handler):
        """Test Anthropic streaming when tool call is blocked."""

        async def mock_post(*args: Any, **kwargs: Any) -> Mock:
            mock_response = Mock()
            mock_response.json.return_value = {
                "id": "test",
                "object": "chat.completion",
                "created": 123,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Tool blocked by policy",
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    },
                ],
            }
            mock_response.raise_for_status = Mock()
            return mock_response

        with patch.object(handler.tool_blocking_client, "post", new=mock_post):

            async def mock_stream():
                for m in [
                    {"type": "message_start", "message": {"id": "msg_123", "content": [], "stop_reason": None}},
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me help"}},
                    {"type": "content_block_stop", "index": 0},
                    {
                        "type": "content_block_start",
                        "index": 1,
                        "content_block": {"type": "tool_use", "id": "toolu_123", "name": "dangerous_tool", "input": {}},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "input_json_delta", "partial_json": '{"arg": "value"}'},
                    },
                    {"type": "content_block_stop", "index": 1},
                    {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}},
                    {"type": "message_stop"},
                ]:
                    yield RubrikLogger._encode_anthropic_chunk_to_sse(m)

            chunks = []
            async for chunk in handler.async_post_call_streaming_iterator_hook(
                user_api_key_dict={},
                response=mock_stream(),
                request_data=create_anthropic_request_data(),
            ):
                chunks.append(await self._decode_sse(chunk))

            assert len(chunks) > 0
            assert "blocked" in self._extract_full_text(chunks).lower(), "Should have explanation about blocked tool"

            message_deltas = self._filter_chunks_by_type(chunks, "message_delta")
            assert len(message_deltas) == 1
            assert message_deltas[0]["delta"]["stop_reason"] == "end_turn"

    async def test_anthropic_streaming_tool_call_allowed(self, handler):
        """Test Anthropic streaming when tool call is allowed."""

        async def mock_post(*args: Any, **kwargs: Any) -> Mock:
            request_json = kwargs.get("json", {})
            mock_response = Mock()
            mock_response.json.return_value = request_json
            mock_response.raise_for_status = Mock()
            return mock_response

        with patch.object(handler.tool_blocking_client, "post", new=mock_post):

            async def mock_stream():
                for m in [
                    {"type": "message_start", "message": {"id": "msg_123", "content": [], "stop_reason": None}},
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me help"}},
                    {"type": "content_block_stop", "index": 0},
                    {
                        "type": "content_block_start",
                        "index": 1,
                        "content_block": {"type": "tool_use", "id": "toolu_123", "name": "safe_tool", "input": {}},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "input_json_delta", "partial_json": '{"arg": "value"}'},
                    },
                    {"type": "content_block_stop", "index": 1},
                    {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}},
                    {"type": "message_stop"},
                ]:
                    yield RubrikLogger._encode_anthropic_chunk_to_sse(m)

            chunks = []
            async for chunk in handler.async_post_call_streaming_iterator_hook(
                user_api_key_dict={},
                response=mock_stream(),
                request_data=create_anthropic_request_data(),
            ):
                chunks.append(await self._decode_sse(chunk))

            tool_use_blocks = self._find_tool_use_blocks(chunks)
            assert len(tool_use_blocks) >= 1, "Should have tool_use content block"
            assert tool_use_blocks[0]["content_block"]["name"] == "safe_tool"

            message_deltas = self._filter_chunks_by_type(chunks, "message_delta")
            assert len(message_deltas) == 1
            assert message_deltas[0]["delta"]["stop_reason"] == "tool_use"

    async def test_anthropic_streaming_blocking_with_real_data(self, handler):
        """Test blocking tool call using real Anthropic streaming data."""
        sample_data_path = Path(__file__).parent / "rubrik_test_sample_data" / "anthropic_streaming_tool_call_response"

        async def mock_post(*args: Any, **kwargs: Any) -> Mock:
            return Mock(
                status_code=200,
                json=lambda: {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "The get_weather tool has been blocked by policy.",
                                "tool_calls": [],
                            },
                            "finish_reason": "stop",
                        },
                    ],
                },
            )

        with patch.object(handler.tool_blocking_client, "post", new=mock_post):
            chunks = []
            async for chunk in handler.async_post_call_streaming_iterator_hook(
                user_api_key_dict={},
                response=self._create_sse_stream_from_file(sample_data_path),
                request_data=create_anthropic_request_data(),
            ):
                chunks.append(await self._decode_sse(chunk))

            assert len(self._filter_chunks_by_type(chunks, "message_start")) == 1
            assert len(self._find_text_deltas(chunks)) > 0
            assert "blocked by policy" in self._extract_full_text(chunks)
            assert len(self._find_tool_use_blocks(chunks)) == 0

            message_deltas = self._filter_chunks_by_type(chunks, "message_delta")
            assert len(message_deltas) == 1
            assert message_deltas[0]["delta"]["stop_reason"] == "end_turn"

    async def test_anthropic_streaming_allowing_with_real_data(self, handler):
        """Test allowing tool calls using real Anthropic streaming data (no blocking)."""
        sample_data_path = Path(__file__).parent / "rubrik_test_sample_data" / "anthropic_streaming_tool_call_response"

        async def mock_post(*args: Any, **kwargs: Any) -> Mock:
            request_json = kwargs.get("json", {})
            return Mock(status_code=200, json=lambda: request_json)

        with patch.object(handler.tool_blocking_client, "post", new=mock_post):
            chunks = []
            async for chunk in handler.async_post_call_streaming_iterator_hook(
                user_api_key_dict={},
                response=self._create_sse_stream_from_file(sample_data_path),
                request_data=create_anthropic_request_data(),
            ):
                chunks.append(await self._decode_sse(chunk))

            tool_use_blocks = self._find_tool_use_blocks(chunks)
            assert len(tool_use_blocks) == 1
            assert tool_use_blocks[0]["content_block"]["name"] == "get_weather"

            message_deltas = self._filter_chunks_by_type(chunks, "message_delta")
            assert len(message_deltas) == 1
            assert message_deltas[0]["delta"]["stop_reason"] == "tool_use"

    async def test_anthropic_streaming_text_only_with_real_data(self, handler):
        """Test that text-only Anthropic responses (no tool calls) pass through unmodified."""
        sample_data_path = Path(__file__).parent / "rubrik_test_sample_data" / "anthropic_streaming_text_response"
        blocking_service_called: Dict[str, Any] = {}

        async def mock_post(*args: Any, **kwargs: Any) -> Mock:
            blocking_service_called.update(kwargs.get("json", {}))
            return Mock(status_code=200, json=lambda: {})

        with patch.object(handler.tool_blocking_client, "post", new=mock_post):
            chunks = []
            async for chunk in handler.async_post_call_streaming_iterator_hook(
                user_api_key_dict={},
                response=self._create_sse_stream_from_file(sample_data_path),
                request_data=create_anthropic_request_data(),
            ):
                chunks.append(await self._decode_sse(chunk))

            assert blocking_service_called == {}, "Blocking service should not be called for text-only responses"
            assert len(self._find_text_deltas(chunks)) > 0
            assert len(self._find_tool_use_blocks(chunks)) == 0

            message_deltas = self._filter_chunks_by_type(chunks, "message_delta")
            assert len(message_deltas) == 1
            assert message_deltas[0]["delta"]["stop_reason"] == "end_turn"
