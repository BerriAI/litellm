#!/usr/bin/env python3
"""
Comprehensive tests for tool call index handling in LiteLLM completion transformation.

This test suite verifies that tool call indices are correctly assigned and preserved
when transforming between different response formats, particularly ensuring that:
1. Each tool call within the same assistant message has a unique, incremental index
2. Tool calls are properly cached with their indices
3. The transformation maintains index consistency across streaming and non-streaming responses
4. Backward compatibility is maintained for existing cache formats
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
    TOOL_CALLS_CACHE,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Function,
    Choices,
    Message,
    ModelResponse,
)
from litellm.types.responses.main import OutputFunctionToolCall


class TestToolCallIndexTransformation:
    """Test suite for tool call index handling in response transformation."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Clear the cache before each test
        TOOL_CALLS_CACHE.flush_cache()
        self.config = LiteLLMCompletionResponsesConfig()

    def teardown_method(self):
        """Clean up after each test."""
        # Clear the cache after each test
        TOOL_CALLS_CACHE.flush_cache()

    def create_tool_call(self, call_id: str, function_name: str, arguments: str) -> ChatCompletionMessageToolCall:
        """Helper method to create a tool call object."""
        return ChatCompletionMessageToolCall(
            id=call_id,
            type="function",
            function=Function(
                name=function_name,
                arguments=arguments
            )
        )

    def create_model_response(self, tool_calls: List[ChatCompletionMessageToolCall]) -> ModelResponse:
        """Helper method to create a model response with tool calls."""
        message = Message(
            role="assistant",
            content=None,
            tool_calls=tool_calls
        )
        
        choice = Choices(
            finish_reason="tool_calls",
            index=0,
            message=message
        )
        
        return ModelResponse(
            id="test_response",
            object="chat.completion",
            created=1234567890,
            model="gpt-4",
            choices=[choice]
        )

    def test_single_tool_call_index(self):
        """Test that a single tool call gets index 0."""
        tool_call = self.create_tool_call("call_1", "get_weather", '{"location": "New York"}')
        response = self.create_model_response([tool_call])
        
        # Transform to cache the tool calls
        output_tools = self.config.transform_chat_completion_tools_to_responses_tools(response)
        
        assert len(output_tools) == 1
        
        # Test transformation back to chat completion message
        tool_output = {"call_id": "call_1", "output": "It's sunny"}
        messages = self.config._transform_responses_api_tool_call_output_to_chat_completion_message(tool_output)
        
        # Find the tool call chunk and verify index
        tool_call_chunk = None
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_call_chunk = msg.tool_calls[0]
                break
            elif hasattr(msg, 'get') and msg.get('tool_calls'):
                tool_call_chunk = msg['tool_calls'][0]
                break
        
        assert tool_call_chunk is not None
        actual_index = tool_call_chunk.get('index') if isinstance(tool_call_chunk, dict) else getattr(tool_call_chunk, 'index', None)
        assert actual_index == 0, f"Expected index 0, got {actual_index}"

    def test_multiple_tool_calls_incremental_indices(self):
        """Test that multiple tool calls get incremental indices (0, 1, 2, ...)."""
        tool_calls = [
            self.create_tool_call("call_1", "get_weather", '{"location": "New York"}'),
            self.create_tool_call("call_2", "get_time", '{"timezone": "UTC"}'),
            self.create_tool_call("call_3", "send_email", '{"to": "user@example.com", "subject": "Test"}'),
            self.create_tool_call("call_4", "calculate", '{"expression": "2+2"}'),
            self.create_tool_call("call_5", "search", '{"query": "python tutorials"}'),
        ]
        
        response = self.create_model_response(tool_calls)
        
        # Transform to cache the tool calls
        output_tools = self.config.transform_chat_completion_tools_to_responses_tools(response)
        
        assert len(output_tools) == 5
        
        # Test each tool call gets the correct index
        expected_indices = [0, 1, 2, 3, 4]
        tool_outputs = [
            {"call_id": "call_1", "output": "Sunny"},
            {"call_id": "call_2", "output": "12:00 PM"},
            {"call_id": "call_3", "output": "Email sent"},
            {"call_id": "call_4", "output": "4"},
            {"call_id": "call_5", "output": "Found tutorials"},
        ]
        
        for i, tool_output in enumerate(tool_outputs):
            messages = self.config._transform_responses_api_tool_call_output_to_chat_completion_message(tool_output)
            
            # Find the tool call chunk and verify index
            tool_call_chunk = None
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_call_chunk = msg.tool_calls[0]
                    break
                elif hasattr(msg, 'get') and msg.get('tool_calls'):
                    tool_call_chunk = msg['tool_calls'][0]
                    break
            
            assert tool_call_chunk is not None
            actual_index = tool_call_chunk.get('index') if isinstance(tool_call_chunk, dict) else getattr(tool_call_chunk, 'index', None)
            assert actual_index == expected_indices[i], f"Tool call {tool_output['call_id']} expected index {expected_indices[i]}, got {actual_index}"

    def test_tool_call_cache_structure(self):
        """Test that tool calls are cached with the correct structure including index."""
        tool_calls = [
            self.create_tool_call("call_1", "function_a", '{"param": "value1"}'),
            self.create_tool_call("call_2", "function_b", '{"param": "value2"}'),
        ]
        
        response = self.create_model_response(tool_calls)
        
        # Transform to cache the tool calls
        self.config.transform_chat_completion_tools_to_responses_tools(response)
        
        # Verify cache structure
        cached_data_1 = TOOL_CALLS_CACHE.get_cache("call_1")
        cached_data_2 = TOOL_CALLS_CACHE.get_cache("call_2")
        
        # Check that cached data has the correct structure
        assert isinstance(cached_data_1, dict)
        assert isinstance(cached_data_2, dict)
        assert "tool_call" in cached_data_1
        assert "index" in cached_data_1
        assert "tool_call" in cached_data_2
        assert "index" in cached_data_2
        
        # Check indices are correct
        assert cached_data_1["index"] == 0
        assert cached_data_2["index"] == 1
        
        # Check tool call objects are preserved
        assert cached_data_1["tool_call"].id == "call_1"
        assert cached_data_2["tool_call"].id == "call_2"

    def test_backward_compatibility_old_cache_format(self):
        """Test backward compatibility when cache contains old format (direct tool call object)."""
        # Simulate old cache format by directly setting a tool call object
        old_tool_call = self.create_tool_call("old_call", "old_function", '{"old": "param"}')
        TOOL_CALLS_CACHE.set_cache(key="old_call", value=old_tool_call)
        
        # Try to transform with old cache format
        tool_output = {"call_id": "old_call", "output": "Old format result"}
        messages = self.config._transform_responses_api_tool_call_output_to_chat_completion_message(tool_output)
        
        # Should default to index 0 for backward compatibility
        tool_call_chunk = None
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_call_chunk = msg.tool_calls[0]
                break
            elif hasattr(msg, 'get') and msg.get('tool_calls'):
                tool_call_chunk = msg['tool_calls'][0]
                break
        
        assert tool_call_chunk is not None
        actual_index = tool_call_chunk.get('index') if isinstance(tool_call_chunk, dict) else getattr(tool_call_chunk, 'index', None)
        assert actual_index == 0, f"Expected index 0 for backward compatibility, got {actual_index}"

    def test_missing_tool_call_in_cache(self):
        """Test behavior when tool call is not found in cache."""
        # Try to transform without caching first
        tool_output = {"call_id": "missing_call", "output": "Some result"}
        messages = self.config._transform_responses_api_tool_call_output_to_chat_completion_message(tool_output)
        
        # Should still work and create a basic message structure
        assert len(messages) >= 1
        
        # Check that we get a tool message at minimum
        tool_message = None
        for msg in messages:
            if hasattr(msg, 'role') and msg.role == "tool":
                tool_message = msg
                break
            elif hasattr(msg, 'get') and msg.get('role') == "tool":
                tool_message = msg
                break
        
        assert tool_message is not None

    def test_empty_tool_calls_list(self):
        """Test handling of empty tool calls list."""
        response = self.create_model_response([])
        
        # Transform empty tool calls
        output_tools = self.config.transform_chat_completion_tools_to_responses_tools(response)
        
        assert len(output_tools) == 0

    def test_tool_call_id_uniqueness(self):
        """Test that tool calls with the same ID but different indices are handled correctly."""
        # This shouldn't happen in normal usage, but test robustness
        tool_calls = [
            self.create_tool_call("same_id", "function_a", '{"param": "value1"}'),
            self.create_tool_call("same_id", "function_b", '{"param": "value2"}'),  # Same ID
        ]
        
        response = self.create_model_response(tool_calls)
        
        # Transform to cache the tool calls
        output_tools = self.config.transform_chat_completion_tools_to_responses_tools(response)
        
        # Should still process both, but the second one will overwrite the first in cache
        assert len(output_tools) == 2
        
        # The cached version should be the last one processed (index 1)
        cached_data = TOOL_CALLS_CACHE.get_cache("same_id")
        assert cached_data["index"] == 1

    def test_large_number_of_tool_calls(self):
        """Test performance and correctness with a large number of tool calls."""
        num_calls = 50
        tool_calls = [
            self.create_tool_call(f"call_{i}", f"function_{i}", f'{{"param": "value_{i}"}}')
            for i in range(num_calls)
        ]
        
        response = self.create_model_response(tool_calls)
        
        # Transform to cache the tool calls
        output_tools = self.config.transform_chat_completion_tools_to_responses_tools(response)
        
        assert len(output_tools) == num_calls
        
        # Test a few random indices to ensure correctness
        test_indices = [0, 10, 25, 49]
        for i in test_indices:
            tool_output = {"call_id": f"call_{i}", "output": f"Result {i}"}
            messages = self.config._transform_responses_api_tool_call_output_to_chat_completion_message(tool_output)
            
            # Find the tool call chunk and verify index
            tool_call_chunk = None
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_call_chunk = msg.tool_calls[0]
                    break
                elif hasattr(msg, 'get') and msg.get('tool_calls'):
                    tool_call_chunk = msg['tool_calls'][0]
                    break
            
            assert tool_call_chunk is not None
            actual_index = tool_call_chunk.get('index') if isinstance(tool_call_chunk, dict) else getattr(tool_call_chunk, 'index', None)
            assert actual_index == i, f"Tool call call_{i} expected index {i}, got {actual_index}"

    def test_cache_persistence_across_transformations(self):
        """Test that cache persists correctly across multiple transformations."""
        # First transformation
        tool_calls_1 = [
            self.create_tool_call("call_1", "function_a", '{"param": "value1"}'),
            self.create_tool_call("call_2", "function_b", '{"param": "value2"}'),
        ]
        response_1 = self.create_model_response(tool_calls_1)
        self.config.transform_chat_completion_tools_to_responses_tools(response_1)
        
        # Second transformation (should not interfere with first)
        tool_calls_2 = [
            self.create_tool_call("call_3", "function_c", '{"param": "value3"}'),
            self.create_tool_call("call_4", "function_d", '{"param": "value4"}'),
        ]
        response_2 = self.create_model_response(tool_calls_2)
        self.config.transform_chat_completion_tools_to_responses_tools(response_2)
        
        # Verify all tool calls are cached with correct indices
        cached_1 = TOOL_CALLS_CACHE.get_cache("call_1")
        cached_2 = TOOL_CALLS_CACHE.get_cache("call_2")
        cached_3 = TOOL_CALLS_CACHE.get_cache("call_3")
        cached_4 = TOOL_CALLS_CACHE.get_cache("call_4")
        
        assert cached_1["index"] == 0
        assert cached_2["index"] == 1
        assert cached_3["index"] == 0  # New response, so index resets
        assert cached_4["index"] == 1

    def test_tool_call_output_message_structure(self):
        """Test the structure of messages returned by tool call output transformation."""
        tool_call = self.create_tool_call("call_1", "get_weather", '{"location": "NYC"}')
        response = self.create_model_response([tool_call])
        
        # Cache the tool call
        self.config.transform_chat_completion_tools_to_responses_tools(response)
        
        # Transform tool output
        tool_output = {"call_id": "call_1", "output": "Weather is sunny"}
        messages = self.config._transform_responses_api_tool_call_output_to_chat_completion_message(tool_output)
        
        # Should return exactly 2 messages: assistant message with tool call chunk, and tool message
        assert len(messages) == 2
        
        # First message should be assistant message with tool call chunk
        assistant_msg = messages[0]
        assert hasattr(assistant_msg, 'role') or assistant_msg.get('role') == "assistant"
        
        # Second message should be tool message
        tool_msg = messages[1]
        tool_role = tool_msg.role if hasattr(tool_msg, 'role') else tool_msg.get('role')
        assert tool_role == "tool"
        
        # Tool message should have correct content and tool_call_id
        tool_content = tool_msg.content if hasattr(tool_msg, 'content') else tool_msg.get('content')
        tool_call_id = tool_msg.tool_call_id if hasattr(tool_msg, 'tool_call_id') else tool_msg.get('tool_call_id')
        assert tool_content == "Weather is sunny"
        assert tool_call_id == "call_1"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 