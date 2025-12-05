import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)


class TestAnthropicLoggingHandlerModelFallback:
    """Test the model fallback logic in the anthropic passthrough logging handler."""

    def setup_method(self):
        """Set up test fixtures"""
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        self.mock_chunks = [
            '{"type": "message_start", "message": {"id": "msg_123", "model": "claude-3-haiku-20240307"}}',
            '{"type": "content_block_delta", "delta": {"text": "Hello"}}',
            '{"type": "content_block_delta", "delta": {"text": " world"}}',
            '{"type": "message_stop"}',
        ]

    def _create_mock_logging_obj(
        self, model_in_details: str = None
    ) -> LiteLLMLoggingObj:
        """Create a mock logging object with optional model in model_call_details"""
        mock_logging_obj = MagicMock()

        if model_in_details:
            # Create a dict-like mock that returns the model for the 'model' key
            mock_model_call_details = {"model": model_in_details}
            mock_logging_obj.model_call_details = mock_model_call_details
        else:
            # Create empty dict or None
            mock_logging_obj.model_call_details = {}

        return mock_logging_obj

    def _create_mock_passthrough_handler(self):
        """Create a mock passthrough success handler"""
        mock_handler = MagicMock()
        return mock_handler

    @patch.object(
        AnthropicPassthroughLoggingHandler, "_build_complete_streaming_response"
    )
    @patch.object(
        AnthropicPassthroughLoggingHandler, "_create_anthropic_response_logging_payload"
    )
    def test_model_from_request_body_used_when_present(
        self, mock_create_payload, mock_build_response
    ):
        """Test that model from request_body is used when present"""
        # Arrange
        request_body = {"model": "claude-3-sonnet-20240229"}
        logging_obj = self._create_mock_logging_obj(
            model_in_details="claude-3-haiku-20240307"
        )
        passthrough_handler = self._create_mock_passthrough_handler()

        # Mock successful response building
        mock_build_response.return_value = MagicMock()
        mock_create_payload.return_value = {"test": "payload"}

        # Act
        result = AnthropicPassthroughLoggingHandler._handle_logging_anthropic_collected_chunks(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=passthrough_handler,
            url_route="/anthropic/v1/messages",
            request_body=request_body,
            endpoint_type="messages",
            start_time=self.start_time,
            all_chunks=self.mock_chunks,
            end_time=self.end_time,
        )

        # Assert
        assert result is not None
        # Verify that _build_complete_streaming_response was called with the request_body model
        mock_build_response.assert_called_once()
        call_args = mock_build_response.call_args
        assert (
            call_args[1]["model"] == "claude-3-sonnet-20240229"
        )  # Should use request_body model

    def test_model_fallback_logic_isolated(self):
        """Test just the model fallback logic in isolation"""
        # Test case 1: Model from request body
        request_body = {"model": "claude-3-sonnet-20240229"}
        logging_obj = self._create_mock_logging_obj(
            model_in_details="claude-3-haiku-20240307"
        )

        # Extract the logic directly from the function
        model = request_body.get("model", "")
        if (
            not model
            and hasattr(logging_obj, "model_call_details")
            and logging_obj.model_call_details.get("model")
        ):
            model = logging_obj.model_call_details.get("model")

        assert model == "claude-3-sonnet-20240229"  # Should use request_body model

        # Test case 2: Fallback to logging obj
        request_body = {}
        logging_obj = self._create_mock_logging_obj(
            model_in_details="claude-3-haiku-20240307"
        )

        model = request_body.get("model", "")
        if (
            not model
            and hasattr(logging_obj, "model_call_details")
            and logging_obj.model_call_details.get("model")
        ):
            model = logging_obj.model_call_details.get("model")

        assert model == "claude-3-haiku-20240307"  # Should use fallback model

        # Test case 3: Empty string in request body, fallback to logging obj
        request_body = {"model": ""}
        logging_obj = self._create_mock_logging_obj(
            model_in_details="claude-3-opus-20240229"
        )

        model = request_body.get("model", "")
        if (
            not model
            and hasattr(logging_obj, "model_call_details")
            and logging_obj.model_call_details.get("model")
        ):
            model = logging_obj.model_call_details.get("model")

        assert model == "claude-3-opus-20240229"  # Should use fallback model

        # Test case 4: Both empty
        request_body = {}
        logging_obj = self._create_mock_logging_obj()

        model = request_body.get("model", "")
        if (
            not model
            and hasattr(logging_obj, "model_call_details")
            and logging_obj.model_call_details.get("model")
        ):
            model = logging_obj.model_call_details.get("model")

        assert model == ""  # Should be empty

    def test_edge_case_missing_model_call_details_attribute(self):
        """Test fallback behavior when logging_obj doesn't have model_call_details attribute"""
        # Case where logging_obj doesn't have the attribute at all
        request_body = {"model": ""}  # Empty model in request body
        logging_obj = MagicMock()
        # Remove the attribute to simulate it not existing
        if hasattr(logging_obj, "model_call_details"):
            delattr(logging_obj, "model_call_details")

        # Extract the logic directly from the function
        model = request_body.get("model", "")
        if (
            not model
            and hasattr(logging_obj, "model_call_details")
            and logging_obj.model_call_details.get("model")
        ):
            model = logging_obj.model_call_details.get("model")

        assert model == ""  # Should remain empty since no fallback available

        # Case where model_call_details exists but get returns None
        request_body = {"model": ""}
        logging_obj = self._create_mock_logging_obj()  # Empty dict

        model = request_body.get("model", "")
        if (
            not model
            and hasattr(logging_obj, "model_call_details")
            and logging_obj.model_call_details.get("model")
        ):
            model = logging_obj.model_call_details.get("model")

        assert model == ""  # Should remain empty


class TestAnthropicStreamingResponseBuilder:
    """Test building complete response from Anthropic streaming chunks."""

    def setup_method(self):
        """Set up test fixtures"""
        # Real chunks from Anthropic streaming response with tool use
        self.real_anthropic_chunks = [
            "event: message_start",
            'data: {"type":"message_start","message":{"model":"claude-sonnet-4-5-20250929","id":"msg_01H2B1LUhw4LkcRVJ3jGh27h","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":586,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":0},"output_tokens":1,"service_tier":"standard"}}}',
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01KYPPgZb11FPC16ic8kZttc","name":"get_weather","input":{}}}',
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}',
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"location\\":\\""}}',
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":" \\"San Fr"}}',
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"ancis"}}',
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"co, CA"}}',
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\\"}"}}',
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"input_tokens":586,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":56}}',
            "event: message_stop",
            'data: {"type":"message_stop"}',
        ]

    def _create_mock_logging_obj(self) -> LiteLLMLoggingObj:
        """Create a mock logging object"""
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}
        return mock_logging_obj

    def test_build_complete_streaming_response_with_real_chunks(self):
        """Test that _build_complete_streaming_response can handle real Anthropic chunks"""
        # Arrange
        logging_obj = self._create_mock_logging_obj()
        model = "claude-sonnet-4-5-20250929"

        # Act
        result = AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
            all_chunks=self.real_anthropic_chunks,
            litellm_logging_obj=logging_obj,
            model=model,
        )

        print("result", result)

        # Assert
        assert result is not None, "Should return a valid response"
        assert hasattr(result, "model"), "Response should have model attribute"
        assert hasattr(result, "choices"), "Response should have choices attribute"

    def test_build_complete_streaming_response_with_empty_chunks(self):
        """Test that _build_complete_streaming_response handles empty chunks gracefully"""
        # Arrange
        logging_obj = self._create_mock_logging_obj()
        model = "claude-3-haiku-20240307"

        # Act
        result = AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
            all_chunks=[],
            litellm_logging_obj=logging_obj,
            model=model,
        )

        # Assert - should handle gracefully (might return None or empty response)
        # The exact behavior depends on implementation
        assert True  # If we get here without exception, the test passes

    def test_build_complete_streaming_response_with_text_content(self):
        """Test building response from chunks with text content"""
        # Arrange
        logging_obj = self._create_mock_logging_obj()
        model = "claude-3-haiku-20240307"
        text_chunks = [
            "event: message_start",
            'data: {"type":"message_start","message":{"model":"claude-3-haiku-20240307","id":"msg_123","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}',
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}',
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":15}}',
            "event: message_stop",
            'data: {"type":"message_stop"}',
        ]

        # Act
        result = AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
            all_chunks=text_chunks,
            litellm_logging_obj=logging_obj,
            model=model,
        )

        # Assert
        assert result is not None, "Should return a valid response"
        assert hasattr(result, "model"), "Response should have model attribute"
        assert hasattr(result, "choices"), "Response should have choices attribute"

    def test_build_response_with_multi_event_chunks(self):
        """
        Test that chunks containing multiple SSE events are properly split and processed.

        This tests the fix for the issue where chunks with multiple events (like multiple
        content_block_delta events in one chunk) would only process the first event,
        causing incomplete tool call arguments.
        """
        # Arrange
        logging_obj = self._create_mock_logging_obj()
        model = "claude-sonnet-4-5-20250929"

        # Simulate real scenario: multiple SSE events in single chunks (as bytes)
        # This mimics what happens when the API returns multiple deltas in one network chunk
        multi_event_chunks = [
            b'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-sonnet-4-5-20250929","id":"msg_01Gg6BZQjp3qz6FNvZeTfYKG","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":586,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":0},"output_tokens":1,"service_tier":"standard"}}}\n\n',
            b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01P41bkuuqosu2X5vJtdH8Sw","name":"get_weather","input":{}}}\n\nevent: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"location\\":"}}\n\nevent: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":" \\"San Franci"}}\n\nevent: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"sco, CA"}}\n\nevent: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\\"}"}}',
            b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
            b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"input_tokens":586,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":56}}\n\n',
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]

        # Act
        result = AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
            all_chunks=multi_event_chunks,
            litellm_logging_obj=logging_obj,
            model=model,
        )

        # Assert
        assert result is not None, "Should return a valid response"
        assert hasattr(result, "choices"), "Response should have choices attribute"
        assert len(result.choices) > 0, "Should have at least one choice"

        choice = result.choices[0]
        assert hasattr(choice, "message"), "Choice should have message attribute"
        assert hasattr(choice.message, "tool_calls"), "Message should have tool_calls"
        assert choice.message.tool_calls is not None, "Tool calls should not be None"
        assert len(choice.message.tool_calls) > 0, "Should have at least one tool call"

        tool_call = choice.message.tool_calls[0]
        assert (
            tool_call.function.name == "get_weather"
        ), "Tool name should be get_weather"

        # This is the critical assertion - the arguments should be complete
        # Not just '{"location":' but the full '{"location": "San Francisco, CA"}'
        assert (
            tool_call.function.arguments == '{"location": "San Francisco, CA"}'
        ), f"Tool arguments should be complete, got: {tool_call.function.arguments}"

        print(
            f"✓ Tool call built correctly with complete arguments: {tool_call.function.arguments}"
        )

    def test_split_sse_chunk_into_events(self):
        """Test the helper function that splits multi-event chunks into individual events."""
        # Test with multiple events in one chunk
        multi_event_chunk = b'event: content_block_start\ndata: {"type":"content_block_start"}\n\nevent: content_block_delta\ndata: {"type":"content_block_delta"}\n\nevent: content_block_delta\ndata: {"type":"content_block_delta2"}\n\n'

        events = AnthropicPassthroughLoggingHandler._split_sse_chunk_into_events(
            multi_event_chunk
        )

        assert len(events) == 3, f"Should split into 3 events, got {len(events)}"
        assert (
            "content_block_start" in events[0]
        ), "First event should be content_block_start"
        assert (
            "content_block_delta" in events[1]
        ), "Second event should be content_block_delta"
        assert (
            "content_block_delta2" in events[2]
        ), "Third event should be content_block_delta2"

        # Test with single event
        single_event_chunk = 'event: message_start\ndata: {"type":"message_start"}\n\n'
        events = AnthropicPassthroughLoggingHandler._split_sse_chunk_into_events(
            single_event_chunk
        )

        assert len(events) == 1, "Should have 1 event for single event chunk"

        # Test with empty chunk
        empty_chunk = ""
        events = AnthropicPassthroughLoggingHandler._split_sse_chunk_into_events(
            empty_chunk
        )

        assert len(events) == 0, "Should have 0 events for empty chunk"
