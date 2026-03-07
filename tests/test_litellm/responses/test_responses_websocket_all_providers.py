"""
Unit tests to verify that all providers support Responses API WebSocket mode.

Tests that:
1. All providers with ResponsesAPIConfig support websocket mode
2. Providers with native websocket support use direct connection
3. Providers without native websocket support use ManagedResponsesWebSocketHandler
"""

import pytest

from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
from litellm.llms.databricks.responses.transformation import (
    DatabricksResponsesAPIConfig,
)
from litellm.llms.github_copilot.responses.transformation import (
    GithubCopilotResponsesAPIConfig,
)
from litellm.llms.hosted_vllm.responses.transformation import (
    HostedVLLMResponsesAPIConfig,
)
from litellm.llms.litellm_proxy.responses.transformation import (
    LiteLLMProxyResponsesAPIConfig,
)
from litellm.llms.manus.responses.transformation import ManusResponsesAPIConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.llms.openrouter.responses.transformation import (
    OpenRouterResponsesAPIConfig,
)
from litellm.llms.perplexity.responses.transformation import PerplexityResponsesConfig
from litellm.llms.volcengine.responses.transformation import (
    VolcEngineResponsesAPIConfig,
)
from litellm.llms.xai.responses.transformation import XAIResponsesAPIConfig


class TestResponsesAPIWebSocketSupport:
    """Test that all providers have websocket support configured correctly"""

    def test_openai_supports_native_websocket(self):
        """OpenAI should support native websocket"""
        config = OpenAIResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is True
        ), "OpenAI should support native websocket"

    def test_azure_supports_native_websocket(self):
        """Azure should support native websocket (inherits from OpenAI)"""
        config = AzureOpenAIResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is True
        ), "Azure should support native websocket"

    def test_xai_uses_managed_websocket(self):
        """XAI should use managed websocket handler"""
        config = XAIResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "XAI should use managed websocket handler"

    def test_github_copilot_uses_managed_websocket(self):
        """GitHub Copilot should use managed websocket handler"""
        config = GithubCopilotResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "GitHub Copilot should use managed websocket handler"

    def test_chatgpt_uses_managed_websocket(self):
        """ChatGPT should use managed websocket handler"""
        config = ChatGPTResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "ChatGPT should use managed websocket handler"

    def test_litellm_proxy_uses_managed_websocket(self):
        """LiteLLM Proxy should use managed websocket handler"""
        config = LiteLLMProxyResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "LiteLLM Proxy should use managed websocket handler"

    def test_volcengine_uses_managed_websocket(self):
        """VolcEngine should use managed websocket handler"""
        config = VolcEngineResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "VolcEngine should use managed websocket handler"

    def test_manus_uses_managed_websocket(self):
        """Manus should use managed websocket handler"""
        config = ManusResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "Manus should use managed websocket handler"

    def test_perplexity_uses_managed_websocket(self):
        """Perplexity should use managed websocket handler"""
        config = PerplexityResponsesConfig()
        assert (
            config.supports_native_websocket() is False
        ), "Perplexity should use managed websocket handler"

    def test_databricks_uses_managed_websocket(self):
        """Databricks should use managed websocket handler"""
        config = DatabricksResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "Databricks should use managed websocket handler"

    def test_openrouter_uses_managed_websocket(self):
        """OpenRouter should use managed websocket handler"""
        config = OpenRouterResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "OpenRouter should use managed websocket handler"

    def test_hosted_vllm_uses_managed_websocket(self):
        """Hosted vLLM should use managed websocket handler"""
        config = HostedVLLMResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "Hosted vLLM should use managed websocket handler"


class TestManagedWebSocketHandlerIntegration:
    """Test that ManagedResponsesWebSocketHandler is properly integrated"""

    @pytest.mark.asyncio
    async def test_managed_handler_instantiation(self):
        """Test that ManagedResponsesWebSocketHandler can be instantiated"""
        from unittest.mock import MagicMock

        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        mock_websocket = MagicMock()
        mock_logging_obj = Logging(
            model="test-model",
            messages=[],
            stream=True,
            call_type="aresponses",
            start_time=0,
            litellm_call_id="test-id",
            function_id="test-func",
        )

        handler = ManagedResponsesWebSocketHandler(
            websocket=mock_websocket,
            model="test-model",
            logging_obj=mock_logging_obj,
            user_api_key_dict=None,
            litellm_metadata={},
            api_key="test-key",
            api_base="https://api.example.com",
            timeout=30.0,
            custom_llm_provider="test_provider",
        )

        assert handler.model == "test-model"
        assert handler.api_key == "test-key"
        assert handler.api_base == "https://api.example.com"
        assert handler.timeout == 30.0
        assert handler.custom_llm_provider == "test_provider"

    @pytest.mark.asyncio
    async def test_websocket_log_messages_marks_sync_success_handler_as_async_origin(
        self,
    ):
        """WebSocket logging should suppress duplicate standard payload emission."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from litellm.responses.streaming_iterator import ResponsesWebSocketStreaming

        mock_logging_obj = MagicMock()
        mock_logging_obj.async_success_handler = AsyncMock()
        mock_logging_obj.success_handler = MagicMock()
        mock_logging_obj.model_call_details = {}

        streaming = ResponsesWebSocketStreaming(
            websocket=MagicMock(),
            backend_ws=MagicMock(),
            logging_obj=mock_logging_obj,
        )
        streaming.messages = [{"type": "response.completed"}]
        streaming.input_messages = [{"role": "user", "content": "hello"}]

        with patch(
            "litellm.responses.streaming_iterator._ws_executor.submit"
        ) as mock_submit:
            await streaming._log_messages()

        assert mock_logging_obj.model_call_details["messages"] == streaming.input_messages
        assert mock_submit.call_args.args == (
            mock_logging_obj.success_handler,
            streaming.messages,
        )
        assert mock_submit.call_args.kwargs.get("called_from_async") is True


class TestChunkTransformation:
    """Test chunk serialization and transformation for WebSocket streaming"""

    def test_serialize_chunk_with_dict(self):
        """Test serialization of dict chunks"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.created",
            "response": {"id": "resp_456", "status": "in_progress"},
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.created" in serialized
        assert "resp_456" in serialized

    def test_serialize_chunk_handles_invalid_json(self):
        """Test that chunks with circular references are handled"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        # Create object with circular reference
        obj = {"a": 1}
        obj["self"] = obj  # type: ignore

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(obj)
        assert serialized is None

    def test_extract_output_messages_with_text_content(self):
        """Test extraction of output messages with text content"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello world"}],
                    }
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["type"] == "message"
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"][0]["text"] == "Hello world"

    def test_extract_output_messages_with_multiple_content_parts(self):
        """Test extraction with multiple content parts"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Part 1. "},
                            {"type": "output_text", "text": "Part 2."},
                        ],
                    }
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "Part 1. Part 2."

    def test_extract_output_messages_with_function_calls(self):
        """Test that function calls are preserved"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "function_call",
                        "id": "call_123",
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    }
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["type"] == "function_call"
        assert messages[0]["id"] == "call_123"
        assert messages[0]["name"] == "get_weather"

    def test_extract_output_messages_filters_empty_text(self):
        """Test that messages with empty text are filtered out"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": ""}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Valid text"}],
                    },
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "Valid text"

    def test_extract_output_messages_handles_non_dict_items(self):
        """Test that non-dict items are skipped"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    "invalid_string",
                    None,
                    123,
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Valid"}],
                    },
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "Valid"

    def test_input_to_messages_with_string(self):
        """Test conversion of string input to messages"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        messages = ManagedResponsesWebSocketHandler._input_to_messages("Hello world")
        assert len(messages) == 1
        assert messages[0]["type"] == "message"
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["type"] == "input_text"
        assert messages[0]["content"][0]["text"] == "Hello world"

    def test_input_to_messages_with_list(self):
        """Test conversion of list input to messages"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        input_list = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Question"}],
            }
        ]

        messages = ManagedResponsesWebSocketHandler._input_to_messages(input_list)
        assert len(messages) == 1
        assert messages[0]["type"] == "message"
        assert messages[0]["content"][0]["text"] == "Question"

    def test_input_to_messages_filters_non_dict_items(self):
        """Test that non-dict items in list input are filtered"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        input_list = [
            "invalid_string",
            None,
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Valid"}],
            },
        ]

        messages = ManagedResponsesWebSocketHandler._input_to_messages(input_list)
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "Valid"

    def test_input_to_messages_handles_empty_input(self):
        """Test that empty input returns empty list"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        assert ManagedResponsesWebSocketHandler._input_to_messages(None) == []
        assert ManagedResponsesWebSocketHandler._input_to_messages([]) == []
        assert ManagedResponsesWebSocketHandler._input_to_messages({}) == []


class TestWebSocketEventTypes:
    """Test that all WebSocket event types are properly handled with dict-based chunks"""

    def test_serialize_response_created_event_dict(self):
        """Test serialization of response.created event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.created",
            "response_id": "resp_123",
            "response": {
                "id": "resp_123",
                "object": "response",
                "status": "in_progress",
                "created_at": 1234567890,
            },
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.created" in serialized
        assert "resp_123" in serialized

    def test_serialize_response_in_progress_event_dict(self):
        """Test serialization of response.in_progress event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {"type": "response.in_progress", "response_id": "resp_123"}

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.in_progress" in serialized

    def test_serialize_output_item_added_event_dict(self):
        """Test serialization of response.output_item.added event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.output_item.added",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "item": {"type": "message", "role": "assistant"},
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.output_item.added" in serialized
        assert "msg_456" in serialized

    def test_serialize_output_text_delta_event_dict(self):
        """Test serialization of response.output_text.delta event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.output_text.delta",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello",
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.output_text.delta" in serialized
        assert "Hello" in serialized

    def test_serialize_output_text_done_event_dict(self):
        """Test serialization of response.output_text.done event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.output_text.done",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "content_index": 0,
            "text": "Hello world",
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.output_text.done" in serialized
        assert "Hello world" in serialized

    def test_serialize_content_part_done_event_dict(self):
        """Test serialization of response.content_part.done event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.content_part.done",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "Complete text"},
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.content_part.done" in serialized

    def test_serialize_output_item_done_event_dict(self):
        """Test serialization of response.output_item.done event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.output_item.done",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "item": {"type": "message", "role": "assistant", "status": "completed"},
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.output_item.done" in serialized
        assert "msg_456" in serialized

    def test_serialize_response_completed_event_dict(self):
        """Test serialization of response.completed event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.completed",
            "response_id": "resp_123",
            "response": {
                "id": "resp_123",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Done"}],
                    }
                ],
            },
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.completed" in serialized
        assert "resp_123" in serialized

    def test_serialize_response_failed_event_dict(self):
        """Test serialization of response.failed event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.failed",
            "response_id": "resp_123",
            "response": {
                "id": "resp_123",
                "status": "failed",
                "status_details": {"error": {"message": "Rate limit exceeded"}},
            },
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.failed" in serialized
        assert "Rate limit exceeded" in serialized

    def test_serialize_response_incomplete_event_dict(self):
        """Test serialization of response.incomplete event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.incomplete",
            "response_id": "resp_123",
            "response": {
                "id": "resp_123",
                "status": "incomplete",
                "status_details": {"reason": "max_output_tokens"},
            },
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.incomplete" in serialized
        assert "max_output_tokens" in serialized


class TestMultiTurnSessionHistory:
    """Test multi-turn conversation handling via session history"""

    def test_extract_output_messages_preserves_multiple_messages(self):
        """Test that multiple output messages are all preserved"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "First message"}],
                    },
                    {
                        "type": "function_call",
                        "id": "call_123",
                        "name": "get_weather",
                        "arguments": "{}",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Second message"}],
                    },
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 3
        assert messages[0]["content"][0]["text"] == "First message"
        assert messages[1]["type"] == "function_call"
        assert messages[2]["content"][0]["text"] == "Second message"

    def test_input_to_messages_with_mixed_content_types(self):
        """Test input conversion with mixed content types"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        input_list = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Question"},
                    {"type": "input_image", "image_url": "https://example.com/img.png"},
                ],
            }
        ]

        messages = ManagedResponsesWebSocketHandler._input_to_messages(input_list)
        assert len(messages) == 1
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][0]["type"] == "input_text"
        assert messages[0]["content"][1]["type"] == "input_image"

    def test_extract_output_messages_with_mixed_text_types(self):
        """Test that both 'output_text' and 'text' types are extracted"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Part 1"},
                            {"type": "text", "text": "Part 2"},
                        ],
                    }
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "Part 1Part 2"

    def test_extract_response_id_from_completed_event(self):
        """Test extraction of response ID from completed event"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {"id": "resp_abc123", "status": "completed"},
        }

        response_id = ManagedResponsesWebSocketHandler._extract_response_id(
            completed_event
        )
        assert response_id == "resp_abc123"

    def test_extract_response_id_handles_missing_response(self):
        """Test that missing response dict returns None"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {"type": "response.completed"}

        response_id = ManagedResponsesWebSocketHandler._extract_response_id(
            completed_event
        )
        assert response_id is None


class TestWebSocketErrorHandling:
    """Test error handling in WebSocket mode"""

    @pytest.mark.asyncio
    async def test_managed_handler_handles_invalid_json(self):
        """Test that invalid JSON in response.create is handled gracefully"""
        from unittest.mock import AsyncMock, MagicMock

        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        mock_websocket = MagicMock()
        mock_websocket.send_text = AsyncMock()
        mock_websocket.recv = AsyncMock(return_value="invalid json {{{")

        mock_logging_obj = Logging(
            model="test-model",
            messages=[],
            stream=True,
            call_type="aresponses",
            start_time=0,
            litellm_call_id="test-id",
            function_id="test-func",
        )

        handler = ManagedResponsesWebSocketHandler(
            websocket=mock_websocket,
            model="test-model",
            logging_obj=mock_logging_obj,
        )

        # Process invalid JSON
        await handler._process_response_create("invalid json {{{")

        # Should have sent an error event
        mock_websocket.send_text.assert_called_once()
        error_event = mock_websocket.send_text.call_args[0][0]
        assert "error" in error_event
        assert "Invalid JSON" in error_event


class TestWebSocketChunkTypes:
    """Test handling of different chunk types from streaming responses"""

    def test_serialize_function_call_chunk(self):
        """Test serialization of function call chunks"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.function_call.added",
            "response_id": "resp_123",
            "item_id": "call_456",
            "output_index": 0,
            "call_id": "call_456",
            "name": "get_weather",
            "arguments": "",
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.function_call.added" in serialized
        assert "get_weather" in serialized

    def test_serialize_function_call_arguments_delta(self):
        """Test serialization of function call arguments delta"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.function_call_arguments.delta",
            "response_id": "resp_123",
            "item_id": "call_456",
            "output_index": 0,
            "call_id": "call_456",
            "delta": '{"location"',
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.function_call_arguments.delta" in serialized
        assert "location" in serialized

    def test_serialize_function_call_arguments_done(self):
        """Test serialization of function call arguments done"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.function_call_arguments.done",
            "response_id": "resp_123",
            "item_id": "call_456",
            "output_index": 0,
            "call_id": "call_456",
            "arguments": '{"location": "Paris"}',
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.function_call_arguments.done" in serialized
        assert "Paris" in serialized

    def test_serialize_reasoning_content_delta(self):
        """Test serialization of reasoning content delta"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.reasoning_content.delta",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "content_index": 0,
            "delta": "Thinking step 1...",
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.reasoning_content.delta" in serialized
        assert "Thinking step 1" in serialized

    def test_serialize_reasoning_content_done(self):
        """Test serialization of reasoning content done"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.reasoning_content.done",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "content_index": 0,
            "reasoning_content": "Complete reasoning...",
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.reasoning_content.done" in serialized
        assert "Complete reasoning" in serialized

    def test_extract_output_messages_preserves_multiple_messages(self):
        """Test that multiple output messages are all preserved"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "First message"}],
                    },
                    {
                        "type": "function_call",
                        "id": "call_123",
                        "name": "get_weather",
                        "arguments": "{}",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Second message"}],
                    },
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 3
        assert messages[0]["content"][0]["text"] == "First message"
        assert messages[1]["type"] == "function_call"
        assert messages[2]["content"][0]["text"] == "Second message"

    def test_input_to_messages_with_mixed_content_types(self):
        """Test input conversion with mixed content types"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        input_list = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Question"},
                    {"type": "input_image", "image_url": "https://example.com/img.png"},
                ],
            }
        ]

        messages = ManagedResponsesWebSocketHandler._input_to_messages(input_list)
        assert len(messages) == 1
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][0]["type"] == "input_text"
        assert messages[0]["content"][1]["type"] == "input_image"

    def test_extract_output_messages_with_mixed_text_types(self):
        """Test that both 'output_text' and 'text' types are extracted"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Part 1"},
                            {"type": "text", "text": "Part 2"},
                        ],
                    }
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "Part 1Part 2"
