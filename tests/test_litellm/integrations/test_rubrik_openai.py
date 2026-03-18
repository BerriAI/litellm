"""
Tests for the Rubrik LiteLLM plugin - OpenAI format.

Covers initialization, tool blocking (non-streaming and streaming),
partial/complete blocking, fail-open behavior, and error recovery.
"""

import json
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

from litellm.integrations.rubrik import LLMResponseFormat, RubrikLogger


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


def create_openai_request_data() -> Dict[str, Any]:
    """Create request_data dict for OpenAI format."""
    return {"proxy_server_request": {"url": "https://api.openai.com/v1/chat/completions"}}


def create_delta_tool_call(tool_call_dict: Dict[str, Any]) -> ChatCompletionDeltaToolCall:
    """Convert a tool call dict to ChatCompletionDeltaToolCall."""
    func_data = tool_call_dict.get("function", {})
    return ChatCompletionDeltaToolCall(
        index=tool_call_dict.get("index"),
        id=tool_call_dict.get("id"),
        type=tool_call_dict.get("type"),
        function=Function(name=func_data.get("name"), arguments=func_data.get("arguments")),
    )


def create_streaming_choice(choice_dict: Dict[str, Any]) -> StreamingChoices:
    """Convert a choice dict to StreamingChoices with properly typed Delta."""
    delta_data = choice_dict.get("delta", {})

    tool_calls = None
    if "tool_calls" in delta_data:
        tool_calls = [create_delta_tool_call(tc) for tc in delta_data["tool_calls"]]

    return StreamingChoices(
        index=choice_dict.get("index", 0),
        delta=Delta(role=delta_data.get("role"), content=delta_data.get("content"), tool_calls=tool_calls),
        finish_reason=choice_dict.get("finish_reason"),
    )


def create_model_response_from_dict(chunk_data: Dict[str, Any]) -> ModelResponseStream:
    """Convert a dict to ModelResponseStream with properly typed nested objects."""
    return ModelResponseStream(
        id=chunk_data.get("id"),
        created=chunk_data.get("created"),
        choices=[create_streaming_choice(choice_data) for choice_data in chunk_data.get("choices", [])],
    )


async def create_sse_stream_from_file(file_path: Path) -> AsyncGenerator[ModelResponseStream, None]:
    """Parse an SSE file and yield ModelResponseStream chunks."""
    with open(file_path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line == "data: [DONE]":
            continue

        if line.startswith("data:"):
            json_str = line[5:].strip()
            chunk_data = json.loads(json_str)
            yield create_model_response_from_dict(chunk_data)


async def create_tool_call_stream(
    deltas: List[ChatCompletionDeltaToolCall],
    finish_reason: str = "tool_calls",
) -> AsyncGenerator[ModelResponseStream, None]:
    """Yield ModelResponseStream chunks for a sequence of tool call deltas followed by a finish chunk."""
    for delta in deltas:
        yield ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content=None, tool_calls=[delta]),
                    finish_reason=None,
                ),
            ],
        )

    yield ModelResponseStream(
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(content=None, tool_calls=None),
                finish_reason=finish_reason,
            ),
        ],
    )


async def create_gpt5_style_stream(
    full_args: str,
) -> AsyncGenerator[ModelResponseStream, None]:
    """GPT-5 style stream: tool header, arguments fragment, then finish chunk repeating the full tool call."""
    # Chunk 1: tool call header with empty arguments
    yield ModelResponseStream(
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            index=0,
                            id="call_refund",
                            type="function",
                            function=Function(name="refund_order", arguments=""),
                        )
                    ]
                ),
                finish_reason=None,
            )
        ],
    )
    # Chunk 2: arguments fragment
    yield ModelResponseStream(
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            index=0,
                            id=None,
                            type=None,
                            function=Function(name=None, arguments=full_args),
                        )
                    ]
                ),
                finish_reason=None,
            )
        ],
    )
    # Chunk 3: finish chunk that also repeats the complete tool call (GPT-5 style)
    yield ModelResponseStream(
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            index=0,
                            id=None,
                            type=None,
                            function=Function(name=None, arguments=full_args),
                        )
                    ]
                ),
                finish_reason="tool_calls",
            )
        ],
    )


def create_blocking_mock_post(
    block_all: bool = True, explanation: str = "Tools blocked by policy"
) -> tuple:
    """Create a mock POST function for the tool blocking service. Returns (mock_post, request_data)."""
    request_data: Dict[str, Any] = {}

    async def mock_post(*args: Any, **kwargs: Any) -> Mock:
        request_data.update(kwargs.get("json", {}))

        mock_response = Mock()
        if block_all:
            mock_response.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": explanation,
                            "tool_calls": [],
                        },
                    },
                ],
            }
        else:
            mock_response.json.return_value = kwargs.get("json", {})

        mock_response.raise_for_status = Mock()
        return mock_response

    return mock_post, request_data


class TestInitialization:
    """Test handler initialization and configuration."""

    def test_init_success(self, mock_env):
        """Test successful initialization with required env vars."""
        with patch("asyncio.create_task", Mock()):
            handler = RubrikLogger()
            assert handler.tool_blocking_endpoint == "http://localhost:8080/v1/after_completion/openai/v1"
            assert handler.logging_endpoint == "http://localhost:8080/v1/litellm/batch"
            assert handler.key == "test-api-key"
            assert handler.async_httpx_client is not None
            assert handler.tool_blocking_client is not None
            assert isinstance(handler.tool_blocking_client, httpx.AsyncClient)

    def test_init_without_url(self):
        """Test initialization fails without RUBRIK_WEBHOOK_URL."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Rubrik webhook URL not configured"):
                RubrikLogger()

    def test_init_without_api_key(self):
        """Test initialization succeeds without optional API key."""
        with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://localhost:8080"}, clear=True):
            with patch("asyncio.create_task", Mock()):
                handler = RubrikLogger()
                assert handler.tool_blocking_endpoint == "http://localhost:8080/v1/after_completion/openai/v1"
                assert handler.key is None

    def test_trailing_slash_removed(self):
        """Test that trailing slash is removed from webhook URL."""
        with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://localhost:8080/"}):
            with patch("asyncio.create_task", Mock()):
                handler = RubrikLogger()
                assert handler.tool_blocking_endpoint == "http://localhost:8080/v1/after_completion/openai/v1"

    def test_v1_suffix_stripped_as_substring_not_charset(self):
        """Test that /v1 is stripped as a substring, not as a character set.

        rstrip("/v1") would incorrectly strip characters {/, v, 1} from the
        end, corrupting URLs like 'http://host/v11'. removesuffix only strips
        the exact substring.
        """
        with patch("asyncio.create_task", Mock()):
            # URL ending in /v1 should be stripped
            with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://host/v1"}):
                h = RubrikLogger()
                assert h.tool_blocking_endpoint == "http://host/v1/after_completion/openai/v1"

            # URL ending in /v11 must NOT be corrupted (rstrip would strip to "http://host/")
            with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://host/v11"}):
                h = RubrikLogger()
                assert h.tool_blocking_endpoint == "http://host/v11/v1/after_completion/openai/v1"

            # URL without /v1 suffix should be unchanged
            with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://host/service"}):
                h = RubrikLogger()
                assert h.tool_blocking_endpoint == "http://host/service/v1/after_completion/openai/v1"

    def test_sampling_rate_fractional(self):
        """Test that fractional sampling rates like 0.5 are correctly parsed.

        isdigit() would reject '0.5', silently leaving the rate at 1.0.
        """
        with patch("asyncio.create_task", Mock()):
            with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://host", "RUBRIK_SAMPLING_RATE": "0.5"}):
                h = RubrikLogger()
                assert h.sampling_rate == 0.5

    def test_sampling_rate_integer(self):
        """Test that integer sampling rates still work."""
        with patch("asyncio.create_task", Mock()):
            with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://host", "RUBRIK_SAMPLING_RATE": "1"}):
                h = RubrikLogger()
                assert h.sampling_rate == 1.0

    def test_sampling_rate_invalid_ignored(self):
        """Test that invalid sampling rates are handled gracefully (default to 1.0)."""
        with patch("asyncio.create_task", Mock()):
            with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://host", "RUBRIK_SAMPLING_RATE": "abc"}):
                h = RubrikLogger()
                assert h.sampling_rate == 1.0

    def test_sampling_rate_unset_defaults_to_one(self):
        """Test that unset sampling rate defaults to 1.0."""
        with patch("asyncio.create_task", Mock()):
            with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://host"}, clear=True):
                h = RubrikLogger()
                assert h.sampling_rate == 1.0

    def test_sampling_rate_clamped_above_one(self):
        """Test that sampling rate > 1.0 is clamped to 1.0."""
        with patch("asyncio.create_task", Mock()):
            with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://host", "RUBRIK_SAMPLING_RATE": "2.0"}):
                h = RubrikLogger()
                assert h.sampling_rate == 1.0

    def test_sampling_rate_clamped_below_zero(self):
        """Test that sampling rate < 0.0 is clamped to 0.0."""
        with patch("asyncio.create_task", Mock()):
            with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://host", "RUBRIK_SAMPLING_RATE": "-0.5"}):
                h = RubrikLogger()
                assert h.sampling_rate == 0.0

    def test_batch_size_invalid_ignored(self):
        """Test that invalid RUBRIK_BATCH_SIZE falls back to default."""
        with patch("asyncio.create_task", Mock()):
            with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://host", "RUBRIK_BATCH_SIZE": "abc"}):
                h = RubrikLogger()
                # Should keep the default from CustomBatchLogger, not crash
                assert isinstance(h.batch_size, int)

    def test_batch_size_valid(self):
        """Test that valid RUBRIK_BATCH_SIZE is applied."""
        with patch("asyncio.create_task", Mock()):
            with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://host", "RUBRIK_BATCH_SIZE": "256"}):
                h = RubrikLogger()
                assert h.batch_size == 256

    def test_batch_size_zero_ignored(self):
        """Test that zero RUBRIK_BATCH_SIZE falls back to default."""
        with patch("asyncio.create_task", Mock()):
            with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://host", "RUBRIK_BATCH_SIZE": "0"}):
                h = RubrikLogger()
                assert h.batch_size > 0

    def test_batch_size_negative_ignored(self):
        """Test that negative RUBRIK_BATCH_SIZE falls back to default."""
        with patch("asyncio.create_task", Mock()):
            with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://host", "RUBRIK_BATCH_SIZE": "-5"}):
                h = RubrikLogger()
                assert h.batch_size > 0

    def test_detect_format_with_query_params(self):
        """Test that URL format detection works even when the URL has query parameters."""
        openai_data = {"proxy_server_request": {"url": "http://host/chat/completions?stream=true"}}
        assert RubrikLogger._detect_llm_response_format(openai_data) == LLMResponseFormat.OPENAI

        anthropic_data = {"proxy_server_request": {"url": "http://host/v1/messages?foo=bar"}}
        assert RubrikLogger._detect_llm_response_format(anthropic_data) == LLMResponseFormat.ANTHROPIC

        unknown_data = {"proxy_server_request": {"url": "http://host/other?q=1"}}
        assert RubrikLogger._detect_llm_response_format(unknown_data) == LLMResponseFormat.UNKNOWN


@pytest.mark.asyncio
class TestBatchLogging:
    """Test the batch logging pipeline."""

    async def test_log_success_event_appends_to_queue(self, handler):
        """Test that async_log_success_event appends to the log queue."""
        kwargs = {
            "standard_logging_object": {"messages": [{"role": "user", "content": "hi"}], "response": "hello"},
        }
        await handler.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)
        assert len(handler.log_queue) == 1

    async def test_log_success_event_sampling_skips(self, handler):
        """Test that events are skipped when random exceeds sampling rate."""
        handler.sampling_rate = 0.0  # Skip all events
        kwargs = {
            "standard_logging_object": {"messages": [{"role": "user", "content": "hi"}], "response": "hello"},
        }
        await handler.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)
        assert len(handler.log_queue) == 0

    async def test_log_success_event_inserts_system_prompt(self, handler):
        """Test that Anthropic system prompt is prepended to messages."""
        kwargs = {
            "standard_logging_object": {"messages": [{"role": "user", "content": "hi"}], "response": "hello"},
            "system": "You are helpful.",
        }
        await handler.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)
        assert len(handler.log_queue) == 1
        messages = handler.log_queue[0]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"

    async def test_flush_queue_sends_batch(self, handler):
        """Test that flush_queue sends queued entries and clears the queue."""
        handler.log_queue = [{"msg": "a"}, {"msg": "b"}]

        mock_response = Mock()
        mock_response.status_code = 200
        handler.async_httpx_client = AsyncMock()
        handler.async_httpx_client.post = AsyncMock(return_value=mock_response)

        await handler.flush_queue()

        handler.async_httpx_client.post.assert_called_once()
        call_kwargs = handler.async_httpx_client.post.call_args
        assert len(call_kwargs.kwargs["json"]) == 2
        assert len(handler.log_queue) == 0

    async def test_flush_queue_empty_noop(self, handler):
        """Test that flush_queue does nothing when queue is empty."""
        handler.async_httpx_client = AsyncMock()
        handler.async_httpx_client.post = AsyncMock()

        await handler.flush_queue()

        handler.async_httpx_client.post.assert_not_called()

    async def test_log_batch_error_does_not_crash(self, handler):
        """Test that HTTP errors in _log_batch_to_rubrik don't propagate."""
        handler.log_queue = [{"msg": "a"}]

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError("err", request=Mock(), response=mock_response)
        )
        handler.async_httpx_client = AsyncMock()
        handler.async_httpx_client.post = AsyncMock(return_value=mock_response)

        await handler.flush_queue()
        # Should not raise; queue should be cleared (fire-and-forget)
        assert len(handler.log_queue) == 0

    async def test_log_success_event_error_does_not_propagate(self, handler):
        """Test that errors in the logging hook don't propagate."""
        kwargs = {}  # Missing standard_logging_object — will raise KeyError
        # Should not raise
        await handler.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)


@pytest.mark.asyncio
class TestCheckAndModifyResponse:
    """Test calling the tool blocking service."""

    async def test_post_to_tool_blocking_service_success(self, handler):
        """Test successful call to tool blocking service returns modified response."""
        response_dict: Dict[str, Any] = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "1", "function": {"name": "dangerous_tool"}},
                        ],
                    },
                },
            ],
        }

        modified_response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "The following tool calls were blocked by policy:\n" "- dangerous_tool: Security policy"
                        ),
                    },
                    "finish_reason": "stop",
                },
            ],
        }
        mock_http_response = Mock()
        mock_http_response.json.return_value = modified_response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_http_response)
        handler.tool_blocking_client = mock_client

        result = await handler._post_to_tool_blocking_service(response_dict)

        assert result == modified_response

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://localhost:8080/v1/after_completion/openai/v1"
        assert call_args[1]["json"] == response_dict
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-api-key"

    async def test_post_to_tool_blocking_service_no_blocks(self, handler):
        """Test when no tools are blocked - returns original response."""
        response_dict: Dict[str, Any] = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "1", "function": {"name": "safe_tool"}},
                        ],
                    },
                },
            ],
        }

        mock_http_response = Mock()
        mock_http_response.json.return_value = response_dict

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_http_response)
        handler.tool_blocking_client = mock_client

        result = await handler._post_to_tool_blocking_service(response_dict)

        assert result == response_dict

    async def test_post_to_tool_blocking_service_timeout(self, handler):
        """Test that timeout raises exception (fail-closed at this layer)."""
        response_dict: Dict[str, Any] = {"choices": [{"message": {"tool_calls": []}}]}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        handler.tool_blocking_client = mock_client

        with pytest.raises(httpx.TimeoutException):
            await handler._post_to_tool_blocking_service(response_dict)

    async def test_post_to_tool_blocking_service_http_error(self, handler):
        """Test that HTTP error raises exception (fail-closed at this layer)."""
        response_dict: Dict[str, Any] = {"choices": [{"message": {"tool_calls": []}}]}

        mock_response = Mock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Server error",
                request=Mock(),
                response=mock_response,
            ),
        )
        handler.tool_blocking_client = mock_client

        with pytest.raises(httpx.HTTPStatusError):
            await handler._post_to_tool_blocking_service(response_dict)


@pytest.mark.asyncio
class TestAsyncPostCallSuccessHook:
    """Test the main hook method."""

    async def test_no_tool_calls_returns_original(self, handler, mock_response_no_tools):
        """Test that responses without tool calls skip the blocking service entirely."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock()
        handler.tool_blocking_client = mock_client

        result = await handler.async_post_call_success_hook(
            data=create_openai_request_data(),
            user_api_key_dict={},
            response=mock_response_no_tools,
        )

        mock_client.post.assert_not_called()
        assert result == mock_response_no_tools

    async def test_successful_blocking(self, handler, mock_response_with_tools):
        """Test successful tool blocking flow."""
        modified_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "The following tool calls were blocked by policy:\n"
                            "- send_email: Email sending not allowed"
                        ),
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            ],
        }
        mock_http_response = Mock()
        mock_http_response.json.return_value = modified_response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_http_response)
        handler.tool_blocking_client = mock_client

        result = await handler.async_post_call_success_hook(
            data=create_openai_request_data(),
            user_api_key_dict={},
            response=mock_response_with_tools,
        )

        choices = result.choices
        tool_calls = choices[0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"

        content = choices[0]["message"]["content"]
        assert "send_email" in content
        assert "Email sending not allowed" in content

    async def test_all_tools_blocked(self, handler, mock_response_with_tools):
        """Test when all tools are blocked by policy."""
        modified_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "The following tool calls were blocked by policy:\n"
                            "- get_weather: Reason 1\n- send_email: Reason 2"
                        ),
                    },
                    "finish_reason": "stop",
                },
            ],
        }
        mock_http_response = Mock()
        mock_http_response.json.return_value = modified_response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_http_response)
        handler.tool_blocking_client = mock_client

        result = await handler.async_post_call_success_hook(
            data=create_openai_request_data(),
            user_api_key_dict={},
            response=mock_response_with_tools,
        )

        choices = result.choices
        assert "tool_calls" not in choices[0]["message"]
        assert choices[0]["finish_reason"] == "stop"

        content = choices[0]["message"]["content"]
        assert "get_weather" in content
        assert "send_email" in content

    async def test_returns_original_response_on_service_timeout(self, handler, mock_response_with_tools):
        """Test that original response is returned when service times out (fail-open)."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        handler.tool_blocking_client = mock_client

        result = await handler.async_post_call_success_hook(
            data=create_openai_request_data(),
            user_api_key_dict={},
            response=mock_response_with_tools,
        )

        assert result == mock_response_with_tools

    async def test_returns_original_response_on_http_error(self, handler, mock_response_with_tools):
        """Test that original response is returned on HTTP error (fail-open)."""
        mock_response = Mock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Server error",
                request=Mock(),
                response=mock_response,
            ),
        )
        handler.tool_blocking_client = mock_client

        result = await handler.async_post_call_success_hook(
            data=create_openai_request_data(),
            user_api_key_dict={},
            response=mock_response_with_tools,
        )

        assert result == mock_response_with_tools

    async def test_no_blocking_when_service_returns_empty(self, handler, mock_response_with_tools):
        """Test that response is unchanged when service allows all tools."""
        mock_http_response = Mock()
        mock_http_response.json.return_value = mock_response_with_tools.model_dump()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_http_response)
        handler.tool_blocking_client = mock_client

        result = await handler.async_post_call_success_hook(
            data=create_openai_request_data(),
            user_api_key_dict={},
            response=mock_response_with_tools,
        )

        assert result == mock_response_with_tools
        choices = result.choices
        tool_calls = choices[0]["message"]["tool_calls"]
        assert len(tool_calls) == 2
        mock_client.post.assert_called_once()


@pytest.mark.asyncio
class TestAsyncPostCallStreamingIteratorHook:
    """Test the streaming iterator hook method."""

    async def test_streaming_iterator_no_tool_calls(self, handler):
        """Test that streaming responses without tool calls pass through unchanged."""

        async def mock_stream():
            yield ModelResponseStream(
                choices=[StreamingChoices(index=0, delta=Delta(content="Hello"), finish_reason=None)],
            )
            yield ModelResponseStream(
                choices=[StreamingChoices(index=0, delta=Delta(content=" world"), finish_reason=None)],
            )
            yield ModelResponseStream(
                choices=[StreamingChoices(index=0, delta=Delta(content=None), finish_reason="stop")],
            )

        chunks = []
        async for chunk in handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict={},
            response=mock_stream(),
            request_data=create_openai_request_data(),
        ):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == " world"
        assert chunks[2].choices[0].finish_reason == "stop"

    async def test_streaming_iterator_blocks_tool(self, handler):
        """Test that blocked tools are removed from the stream."""

        mock_post, _ = create_blocking_mock_post(
            explanation="The following tool calls were blocked by policy:\n- dangerous_tool: Security policy",
        )

        mock_client = AsyncMock()
        mock_client.post = mock_post
        handler.tool_blocking_client = mock_client

        tool_call_deltas = [
            ChatCompletionDeltaToolCall(
                index=0,
                id="call_",
                type="function",
                function=Function(name="dangerous", arguments=None),
            ),
            ChatCompletionDeltaToolCall(
                index=0,
                id=None,
                type=None,
                function=Function(name="_tool", arguments=None),
            ),
            ChatCompletionDeltaToolCall(
                index=0,
                id=None,
                type=None,
                function=Function(name=None, arguments='{"arg": "value"}'),
            ),
        ]

        chunks = []
        async for chunk in handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict={},
            response=create_tool_call_stream(tool_call_deltas),
            request_data=create_openai_request_data(),
        ):
            chunks.append(chunk)

        # Single synthetic chunk with explanation and no tools
        assert len(chunks) == 1

        explanation_chunk = chunks[0]
        assert explanation_chunk.choices[0].delta.content is not None
        assert "dangerous_tool" in explanation_chunk.choices[0].delta.content
        assert "blocked by policy" in explanation_chunk.choices[0].delta.content.lower()
        assert explanation_chunk.choices[0].finish_reason == "stop"

    async def test_streaming_iterator_allows_safe_tool(self, handler):
        """Test that non-blocked tools pass through the stream with original chunks replayed."""

        mock_post, _ = create_blocking_mock_post(block_all=False)

        mock_client = AsyncMock()
        mock_client.post = mock_post
        handler.tool_blocking_client = mock_client

        tool_call_deltas = [
            ChatCompletionDeltaToolCall(
                index=0,
                id="call_123",
                type="function",
                function=Function(name="safe_tool", arguments=None),
            ),
            ChatCompletionDeltaToolCall(
                index=0,
                id=None,
                type=None,
                function=Function(name=None, arguments='{"arg": "value"}'),
            ),
        ]

        chunks = []
        async for chunk in handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict={},
            response=create_tool_call_stream(tool_call_deltas),
            request_data=create_openai_request_data(),
        ):
            chunks.append(chunk)

        # All buffered chunks replayed when all tools are allowed (2 tool deltas + 1 finish)
        assert len(chunks) == 3

        # First chunk has tool call header with original empty arguments — not mutated
        assert chunks[0].choices[0].delta.tool_calls is not None
        assert chunks[0].choices[0].delta.tool_calls[0].id == "call_123"
        assert chunks[0].choices[0].delta.tool_calls[0].function.name == "safe_tool"
        assert chunks[0].choices[0].delta.tool_calls[0].function.arguments == ""

        # Second chunk has the arguments
        assert chunks[1].choices[0].delta.tool_calls[0].function.arguments == '{"arg": "value"}'

        # Last chunk has finish_reason
        assert chunks[-1].choices[0].finish_reason == "tool_calls"

    async def test_streaming_blocking_with_real_data(self, handler):
        """Test blocking multiple delete_table tool calls using real streaming data."""

        mock_post, request_data = create_blocking_mock_post(
            block_all=True,
            explanation="The delete_table tool has been blocked by policy.",
        )

        mock_client = AsyncMock()
        mock_client.post = mock_post
        handler.tool_blocking_client = mock_client

        sample_data_path = Path(__file__).parent / "rubrik_test_sample_data" / "openai_streaming_multiple_tool_call_response"

        chunks = []
        async for chunk in handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict={},
            response=create_sse_stream_from_file(sample_data_path),
            request_data=create_openai_request_data(),
        ):
            chunks.append(chunk)

        # Single synthetic chunk with explanation
        assert len(chunks) == 1

        explanation_chunk = chunks[0]
        assert explanation_chunk.choices[0].delta.content is not None
        assert "delete_table" in explanation_chunk.choices[0].delta.content
        assert "blocked by policy" in explanation_chunk.choices[0].delta.content.lower()
        assert explanation_chunk.choices[0].finish_reason == "stop"

        # Verify the blocking service received both delete_table tool calls
        assert "choices" in request_data
        message = request_data["choices"][0]["message"]
        assert len(message["tool_calls"]) == 2

        tool_call_0 = message["tool_calls"][0]
        assert tool_call_0["function"]["name"] == "delete_table"
        assert tool_call_0["id"] == "call_jKF1G4LDmfcDlWAwvbXjJEk3"
        assert json.loads(tool_call_0["function"]["arguments"]) == {"table": "foo"}

        tool_call_1 = message["tool_calls"][1]
        assert tool_call_1["function"]["name"] == "delete_table"
        assert tool_call_1["id"] == "call_Fng9sRbSfUluWSYt9KrDzc5x"
        assert json.loads(tool_call_1["function"]["arguments"]) == {"table": "bar"}

    async def test_streaming_partial_blocking(self, handler):
        """Test that partial blocking replays allowed tool chunks, adds explanation, and finishes."""
        explanation = "Tool blocked by policy"

        async def mock_post(*_args: Any, **kwargs: Any) -> Mock:
            request_json = kwargs.get("json", {})
            all_tool_calls = request_json["choices"][0]["message"]["tool_calls"]
            allowed = [tc for tc in all_tool_calls if tc["id"] == "call_B"]

            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"role": "assistant", "content": explanation, "tool_calls": allowed}}],
            }
            mock_response.raise_for_status = Mock()
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = mock_post
        handler.tool_blocking_client = mock_client

        tool_call_deltas = [
            ChatCompletionDeltaToolCall(
                index=0,
                id="call_A",
                type="function",
                function=Function(name="blocked_tool", arguments=None),
            ),
            ChatCompletionDeltaToolCall(
                index=0,
                id=None,
                type=None,
                function=Function(name=None, arguments='{"x": 1}'),
            ),
            ChatCompletionDeltaToolCall(
                index=1,
                id="call_B",
                type="function",
                function=Function(name="allowed_tool", arguments=None),
            ),
            ChatCompletionDeltaToolCall(
                index=1,
                id=None,
                type=None,
                function=Function(name=None, arguments='{"y": 2}'),
            ),
        ]

        chunks = []
        async for chunk in handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict={},
            response=create_tool_call_stream(tool_call_deltas),
            request_data=create_openai_request_data(),
        ):
            chunks.append(chunk)

        # Replay: allowed tool chunks (re-indexed) + explanation chunk + finish chunk
        assert len(chunks) == 4

        # Allowed tool call chunks should be re-indexed to 0
        tool_chunks = [c for c in chunks if c.choices[0].delta.tool_calls]
        assert len(tool_chunks) == 2
        assert tool_chunks[0].choices[0].delta.tool_calls[0].id == "call_B"
        assert tool_chunks[0].choices[0].delta.tool_calls[0].index == 0  # Re-indexed from original 1

        # Explanation chunk
        explanation_chunks = [c for c in chunks if c.choices[0].delta.content]
        assert len(explanation_chunks) == 1
        assert explanation in explanation_chunks[0].choices[0].delta.content

        # Last chunk has finish_reason="tool_calls" since some tools are allowed
        assert chunks[-1].choices[0].finish_reason == "tool_calls"

    async def test_streaming_partial_blocking_empty_explanation(self, handler):
        """Test that empty explanation from blocking service does not produce spurious whitespace."""

        async def mock_post(*_args: Any, **kwargs: Any) -> Mock:
            request_json = kwargs.get("json", {})
            all_tool_calls = request_json["choices"][0]["message"]["tool_calls"]
            allowed = [tc for tc in all_tool_calls if tc["id"] == "call_B"]

            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": allowed}}],
            }
            mock_response.raise_for_status = Mock()
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = mock_post
        handler.tool_blocking_client = mock_client

        tool_call_deltas = [
            ChatCompletionDeltaToolCall(
                index=0, id="call_A", type="function",
                function=Function(name="blocked_tool", arguments='{"x": 1}'),
            ),
            ChatCompletionDeltaToolCall(
                index=1, id="call_B", type="function",
                function=Function(name="allowed_tool", arguments='{"y": 2}'),
            ),
        ]

        chunks = []
        async for chunk in handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict={},
            response=create_tool_call_stream(tool_call_deltas),
            request_data=create_openai_request_data(),
        ):
            chunks.append(chunk)

        # Allowed tool chunk + finish chunk (no explanation chunk when explanation is empty)
        assert len(chunks) == 2

        # Should have the allowed tool call
        assert chunks[0].choices[0].delta.tool_calls is not None
        assert len(chunks[0].choices[0].delta.tool_calls) == 1
        assert chunks[0].choices[0].delta.tool_calls[0].id == "call_B"

        # No explanation chunk should be present — empty explanation is suppressed
        for chunk in chunks:
            content = chunk.choices[0].delta.content
            assert content is None or content.strip() == ""
            assert content != "\n\n"

    async def test_streaming_allowing_with_real_data(self, handler):
        """Test allowing tool calls using real streaming data (no blocking) — chunks replayed as-is."""

        mock_post, request_data = create_blocking_mock_post(block_all=False)

        mock_client = AsyncMock()
        mock_client.post = mock_post
        handler.tool_blocking_client = mock_client

        sample_data_path = Path(__file__).parent / "rubrik_test_sample_data" / "openai_streaming_multiple_tool_call_response"

        chunks = []
        async for chunk in handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict={},
            response=create_sse_stream_from_file(sample_data_path),
            request_data=create_openai_request_data(),
        ):
            chunks.append(chunk)

        # All buffered tool call chunks replayed when all tools are allowed
        assert len(chunks) > 1

        # Last chunk has finish_reason
        assert chunks[-1].choices[0].finish_reason == "tool_calls"

        # Verify both tool call IDs appear across the replayed chunks
        all_tool_ids = set()
        for chunk in chunks:
            if chunk.choices[0].delta.tool_calls:
                for tc in chunk.choices[0].delta.tool_calls:
                    if tc.id:
                        all_tool_ids.add(tc.id)
        assert "call_jKF1G4LDmfcDlWAwvbXjJEk3" in all_tool_ids
        assert "call_Fng9sRbSfUluWSYt9KrDzc5x" in all_tool_ids

        # Verify the blocking service received both tool calls
        assert "choices" in request_data
        message = request_data["choices"][0]["message"]
        assert len(message["tool_calls"]) == 2

    async def test_streaming_text_only_with_real_data(self, handler):
        """Test that text-only responses (no tool calls) pass through unmodified."""

        sample_data_path = Path(__file__).parent / "rubrik_test_sample_data" / "openai_streaming_text_response"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock()
        handler.tool_blocking_client = mock_client

        chunks = []
        async for chunk in handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict={},
            response=create_sse_stream_from_file(sample_data_path),
            request_data=create_openai_request_data(),
        ):
            chunks.append(chunk)

        assert len(chunks) == 25
        mock_client.post.assert_not_called()  # Blocking service should not be called for text-only streams

        text_chunks = [c for c in chunks if c.choices[0].delta.content is not None]
        assert len(text_chunks) > 0

        for chunk in chunks:
            if hasattr(chunk.choices[0].delta, "tool_calls"):
                assert chunk.choices[0].delta.tool_calls is None

        assert chunks[-1].choices[0].finish_reason == "stop"

        full_text = "".join(
            chunk.choices[0].delta.content for chunk in chunks if chunk.choices[0].delta.content is not None
        )
        assert (
            full_text
            == "I'm ChatGPT, your AI assistant here to help answer questions and assist with tasks! How can I help "
            "you today?"
        )

    async def test_streaming_error_recovery_yields_buffered_chunks(self, handler):
        """Test that buffered chunks are yielded when an error occurs during streaming."""

        service_call_count = 0

        async def mock_post_that_fails(*_args: Any, **_kwargs: Any) -> None:
            nonlocal service_call_count
            service_call_count += 1
            raise httpx.TimeoutException("Service timeout")

        mock_client = AsyncMock()
        mock_client.post = mock_post_that_fails
        handler.tool_blocking_client = mock_client

        tool_call_deltas = [
            ChatCompletionDeltaToolCall(
                index=0,
                id="call_123",
                type="function",
                function=Function(name="test_tool", arguments=None),
            ),
            ChatCompletionDeltaToolCall(
                index=0,
                id=None,
                type=None,
                function=Function(name=None, arguments='{"arg": "value"}'),
            ),
        ]

        chunks = []
        async for chunk in handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict={},
            response=create_tool_call_stream(tool_call_deltas),
            request_data=create_openai_request_data(),
        ):
            chunks.append(chunk)

        # All 3 buffered chunks should be yielded despite the service error (fail-open)
        assert len(chunks) == 3
        assert service_call_count == 1

        assert chunks[0].choices[0].delta.tool_calls is not None
        assert chunks[0].choices[0].delta.tool_calls[0].id == "call_123"
        assert chunks[2].choices[0].finish_reason == "tool_calls"

    async def test_streaming_finish_chunk_with_tool_calls_does_not_double_args(self, handler):
        """
        Regression test: some models (e.g. GPT-5) repeat the complete tool call in the
        finish chunk instead of sending an empty delta.  This must not cause the arguments
        to be concatenated twice before being sent to the blocking service.
        """
        full_args = '{"order_id":"ORD-12345","reason":"Item arrived damaged"}'
        mock_post, request_data = create_blocking_mock_post(block_all=False)

        mock_client = AsyncMock()
        mock_client.post = mock_post
        handler.tool_blocking_client = mock_client

        chunks = []
        async for chunk in handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict={},
            response=create_gpt5_style_stream(full_args),
            request_data=create_openai_request_data(),
        ):
            chunks.append(chunk)

        # Verify the blocking service received non-doubled arguments
        assert "choices" in request_data, "Blocking service was not called"
        sent_tool_calls = request_data["choices"][0]["message"]["tool_calls"]
        assert len(sent_tool_calls) == 1
        sent_args = sent_tool_calls[0]["function"]["arguments"]
        assert sent_args == full_args, f"Arguments were doubled: {sent_args!r}"
        json.loads(sent_args)  # Must be valid JSON, not two concatenated objects

        # Plugin replays all buffered chunks when all tools are allowed
        assert len(chunks) == 3  # 3 original chunks replayed
        assert chunks[-1].choices[0].finish_reason == "tool_calls"

    async def test_streaming_finish_chunk_with_tool_calls_all_blocked(self, handler):
        """
        Regression test: when a GPT-5 style finish chunk carries tool_calls and ALL tools
        are blocked, the plugin must still emit an explanation chunk with finish_reason="stop".
        """
        full_args = '{"order_id":"ORD-12345","reason":"Item arrived damaged"}'
        explanation = "Tool blocked by policy"
        mock_post, request_data = create_blocking_mock_post(block_all=True, explanation=explanation)

        mock_client = AsyncMock()
        mock_client.post = mock_post
        handler.tool_blocking_client = mock_client

        chunks = []
        async for chunk in handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict={},
            response=create_gpt5_style_stream(full_args),
            request_data=create_openai_request_data(),
        ):
            chunks.append(chunk)

        # Exactly 1 chunk: the explanation with finish_reason="stop"
        assert len(chunks) == 1
        assert chunks[0].choices[0].finish_reason == "stop"
        assert explanation in chunks[0].choices[0].delta.content

    async def test_streaming_finish_chunk_partial_blocking(self, handler):
        """
        Regression test: GPT-5 style finish chunk with two tools where one is blocked.
        The explanation must still be emitted even though the finish chunk has allowed tools.
        """
        args_allowed = '{"query":"weather"}'
        args_blocked = '{"path":"/etc/passwd"}'
        explanation = "Tool blocked by policy"

        async def mock_post(*_args: Any, **kwargs: Any) -> Mock:
            """Return only the allowed tool (index 0), blocking index 1."""
            payload = kwargs.get("json", {})
            tool_calls = payload.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
            allowed = [tc for tc in tool_calls if tc.get("function", {}).get("name") == "get_weather"]
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"role": "assistant", "content": explanation, "tool_calls": allowed}}]
            }
            mock_response.raise_for_status = Mock()
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = mock_post
        handler.tool_blocking_client = mock_client

        async def gpt5_two_tool_stream():
            # Tool 0 header
            yield ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta(tool_calls=[
                ChatCompletionDeltaToolCall(index=0, id="call_weather", type="function", function=Function(name="get_weather", arguments="")),
            ]), finish_reason=None)])
            # Tool 0 args
            yield ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta(tool_calls=[
                ChatCompletionDeltaToolCall(index=0, id=None, type=None, function=Function(name=None, arguments=args_allowed)),
            ]), finish_reason=None)])
            # Tool 1 header
            yield ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta(tool_calls=[
                ChatCompletionDeltaToolCall(index=1, id="call_read_file", type="function", function=Function(name="read_file", arguments="")),
            ]), finish_reason=None)])
            # Tool 1 args
            yield ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta(tool_calls=[
                ChatCompletionDeltaToolCall(index=1, id=None, type=None, function=Function(name=None, arguments=args_blocked)),
            ]), finish_reason=None)])
            # GPT-5 style finish chunk repeating both tools
            yield ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta(tool_calls=[
                ChatCompletionDeltaToolCall(index=0, id=None, type=None, function=Function(name=None, arguments=args_allowed)),
                ChatCompletionDeltaToolCall(index=1, id=None, type=None, function=Function(name=None, arguments=args_blocked)),
            ]), finish_reason="tool_calls")])

        chunks = []
        async for chunk in handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict={},
            response=gpt5_two_tool_stream(),
            request_data=create_openai_request_data(),
        ):
            chunks.append(chunk)

        # Should have: tool 0 chunks (header + args), explanation chunk, finish chunk with allowed tool
        # The blocked tool 1 chunks should be dropped
        has_explanation = any(
            c.choices[0].delta and c.choices[0].delta.content and explanation in c.choices[0].delta.content
            for c in chunks
        )
        assert has_explanation, "Explanation chunk was not emitted for partial blocking"

        # Finish chunk should have finish_reason="tool_calls" (not "stop")
        finish_chunks = [c for c in chunks if c.choices[0].finish_reason]
        assert len(finish_chunks) == 1
        assert finish_chunks[0].choices[0].finish_reason == "tool_calls"


@pytest.fixture
def mock_response_no_tools():
    """Create a real ModelResponse without tool calls to exercise the model_validate path."""
    return ModelResponse(**{
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            },
        ],
    })


@pytest.fixture
def mock_response_with_tools():
    """Create a real ModelResponse with tool calls to exercise the model_validate path."""
    return ModelResponse(**{
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "send_email",
                                "arguments": '{"to": "user@example.com", "subject": "Test"}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco"}',
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            },
        ],
    })
