"""
Test that websearch interception properly converts agentic response to fake stream.

Fixes https://github.com/BerriAI/litellm/issues/20187

The issue was that when:
1. Client requested stream=True
2. WebSearch interception converted stream to False
3. Agentic loop ran and returned a non-streaming response with correct usage
4. The response was returned without conversion to streaming format

This caused Claude Code to receive a response with 0 output tokens instead of
the actual output tokens from the follow-up request.
"""

import json
import pytest
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
    FakeAnthropicMessagesStreamIterator,
)
from litellm.integrations.custom_logger import CustomLogger


class MockCustomCallback(CustomLogger):
    """Mock callback that inherits from CustomLogger for proper isinstance checks"""
    
    def __init__(self, agentic_response):
        super().__init__()
        self.agentic_response = agentic_response
    
    async def async_should_run_agentic_loop(self, response, model, messages, tools, stream, custom_llm_provider, kwargs):
        return True, {"tool_calls": [{"id": "tc_123"}]}
    
    async def async_run_agentic_loop(self, tools, model, messages, response, anthropic_messages_provider_config, anthropic_messages_optional_request_params, logging_obj, stream, kwargs):
        return self.agentic_response


class TestWebsearchStreamConversion:
    """Tests for websearch interception stream conversion"""

    @pytest.mark.asyncio
    async def test_agentic_response_converted_to_fake_stream_when_stream_was_converted(self):
        """
        Test that when websearch_interception_converted_stream is True,
        the agentic response is converted to a fake stream.
        """
        # Create a mock agentic response with actual output tokens
        mock_agentic_response = {
            "id": "msg_test123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5-20250929",
            "content": [
                {
                    "type": "text",
                    "text": "This is the response from the follow-up request with search results."
                }
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 38196,
                "output_tokens": 395  # This should be preserved in the fake stream
            }
        }
        
        # Create a mock CustomLogger that simulates websearch interception
        mock_callback = MockCustomCallback(mock_agentic_response)
        
        # Create a mock logging object with websearch_interception_converted_stream=True
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {
            "websearch_interception_converted_stream": True
        }
        mock_logging_obj.dynamic_success_callbacks = []
        
        # Create the handler
        handler = BaseLLMHTTPHandler()
        
        # Patch litellm.callbacks to include our mock callback
        with patch("litellm.callbacks", [mock_callback]):
            result = await handler._call_agentic_completion_hooks(
                response={"content": [{"type": "tool_use", "name": "WebSearch"}]},  # Initial response with tool_use
                model="claude-sonnet-4-5-20250929",
                messages=[],
                anthropic_messages_provider_config=MagicMock(),
                anthropic_messages_optional_request_params={},
                logging_obj=mock_logging_obj,
                stream=False,
                custom_llm_provider="bedrock",
                kwargs={},
            )
        
        # The result should be a FakeAnthropicMessagesStreamIterator
        assert isinstance(result, FakeAnthropicMessagesStreamIterator)
        
        # Verify the fake stream contains the correct output tokens
        chunks = list(result)
        
        # Find the message_delta chunk which contains output_tokens
        message_delta_chunk = None
        for chunk in chunks:
            chunk_str = chunk.decode() if isinstance(chunk, bytes) else chunk
            if "message_delta" in chunk_str:
                # Parse the event data
                for line in chunk_str.split('\n'):
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                        if data.get("type") == "message_delta":
                            message_delta_chunk = data
                            break
        
        assert message_delta_chunk is not None
        assert message_delta_chunk["usage"]["output_tokens"] == 395

    @pytest.mark.asyncio
    async def test_agentic_response_not_converted_when_stream_was_not_converted(self):
        """
        Test that when websearch_interception_converted_stream is False,
        the agentic response is returned as-is (not converted to fake stream).
        """
        mock_agentic_response = {
            "id": "msg_test123",
            "content": [{"type": "text", "text": "Response"}],
            "usage": {"output_tokens": 100}
        }
        
        mock_callback = MockCustomCallback(mock_agentic_response)
        
        # websearch_interception_converted_stream is False
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {
            "websearch_interception_converted_stream": False
        }
        mock_logging_obj.dynamic_success_callbacks = []
        
        handler = BaseLLMHTTPHandler()
        
        with patch("litellm.callbacks", [mock_callback]):
            result = await handler._call_agentic_completion_hooks(
                response={},
                model="test-model",
                messages=[],
                anthropic_messages_provider_config=MagicMock(),
                anthropic_messages_optional_request_params={},
                logging_obj=mock_logging_obj,
                stream=False,
                custom_llm_provider="bedrock",
                kwargs={},
            )
        
        # The result should be the dict response, not a FakeAnthropicMessagesStreamIterator
        assert result == mock_agentic_response
        assert not isinstance(result, FakeAnthropicMessagesStreamIterator)


class TestFakeStreamIteratorUsage:
    """Tests for FakeAnthropicMessagesStreamIterator usage handling"""

    def test_fake_stream_preserves_output_tokens(self):
        """Test that FakeAnthropicMessagesStreamIterator preserves output_tokens in the stream"""
        response = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [{"type": "text", "text": "Hello world"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50
            }
        }
        
        iterator = FakeAnthropicMessagesStreamIterator(response=response)
        chunks = list(iterator)
        
        # Find message_delta chunk
        output_tokens_found = None
        for chunk in chunks:
            chunk_str = chunk.decode() if isinstance(chunk, bytes) else chunk
            if "message_delta" in chunk_str:
                for line in chunk_str.split('\n'):
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                        if data.get("type") == "message_delta":
                            output_tokens_found = data["usage"]["output_tokens"]
                            break
        
        assert output_tokens_found == 50

    def test_fake_stream_preserves_input_tokens_in_message_start(self):
        """Test that FakeAnthropicMessagesStreamIterator preserves input_tokens in message_start"""
        response = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 200,
                "output_tokens": 30
            }
        }
        
        iterator = FakeAnthropicMessagesStreamIterator(response=response)
        chunks = list(iterator)
        
        # Find message_start chunk
        input_tokens_found = None
        for chunk in chunks:
            chunk_str = chunk.decode() if isinstance(chunk, bytes) else chunk
            if "message_start" in chunk_str:
                for line in chunk_str.split('\n'):
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                        if data.get("type") == "message_start":
                            input_tokens_found = data["message"]["usage"]["input_tokens"]
                            break
        
        assert input_tokens_found == 200
