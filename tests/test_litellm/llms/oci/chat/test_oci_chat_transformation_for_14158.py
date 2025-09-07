import pytest
from litellm.llms.oci.chat.transformation import adapt_messages_to_generic_oci_standard

def test_adapt_messages_with_empty_content_and_tool_calls():
    """Test that assistant messages with empty content and tool_calls are processed correctly."""
    # Arrange
    messages_with_empty_content = [
        {"role": "user", "content": "Tell me the weather in Tokyo."},
        {
            "role": "assistant",
            "content": "",  # Empty string
            "tool_calls": [
                {
                    "id": "call_test_empty",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Tokyo"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "content": '{"weather": "Sunny", "temperature": "25°C"}',
            "tool_call_id": "call_test_empty"
        }
    ]
    
    # Act
    result = adapt_messages_to_generic_oci_standard(messages_with_empty_content)
    
    # Assert
    assert len(result) == 3
    
    # Check user message
    assert result[0].role == "USER"
    assert result[0].content[0].type == "TEXT"
    assert result[0].content[0].text == "Tell me the weather in Tokyo."
    
    # Check assistant message with tool_calls (should prioritize tool_calls over empty content)
    assert result[1].role == "ASSISTANT"
    assert result[1].toolCalls is not None
    assert len(result[1].toolCalls) == 1
    assert result[1].toolCalls[0].id == "call_test_empty"
    assert result[1].toolCalls[0].name == "get_weather"
    
    # Check tool response message
    assert result[2].role == "TOOL"  # Tool responses have TOOL role, not USER
    assert result[2].content[0].type == "TEXT"
    assert "weather" in result[2].content[0].text
    assert result[2].toolCallId == "call_test_empty"  # Tool call ID is in separate field

def test_adapt_messages_with_none_content_and_tool_calls():
    """Test that assistant messages with None content and tool_calls are processed correctly."""
    # Arrange
    messages_with_none_content = [
        {"role": "user", "content": "Tell me the weather in Tokyo."},
        {
            "role": "assistant",
            "content": None,  # None value
            "tool_calls": [
                {
                    "id": "call_test_none",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Tokyo"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "content": '{"weather": "Sunny", "temperature": "25°C"}',
            "tool_call_id": "call_test_none"
        }
    ]
    
    # Act
    result = adapt_messages_to_generic_oci_standard(messages_with_none_content)
    
    # Assert
    assert len(result) == 3
    
    # Check assistant message prioritizes tool_calls over None content
    assert result[1].role == "ASSISTANT"
    assert result[1].toolCalls is not None
    assert len(result[1].toolCalls) == 1
    assert result[1].toolCalls[0].id == "call_test_none"

def test_adapt_messages_with_tool_calls_only():
    """Test that assistant messages with only tool_calls (no content field) are processed correctly."""
    # Arrange
    messages_no_content = [
        {"role": "user", "content": "Tell me the weather in Tokyo."},
        {
            "role": "assistant",
            # No content field at all
            "tool_calls": [
                {
                    "id": "call_test_no_content",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Tokyo"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "content": '{"weather": "Sunny", "temperature": "25°C"}',
            "tool_call_id": "call_test_no_content"
        }
    ]
    
    # Act
    result = adapt_messages_to_generic_oci_standard(messages_no_content)
    
    # Assert
    assert len(result) == 3
    
    # Check assistant message processes tool_calls correctly
    assert result[1].role == "ASSISTANT"
    assert result[1].toolCalls is not None
    assert len(result[1].toolCalls) == 1
    assert result[1].toolCalls[0].id == "call_test_no_content"

def test_adapt_messages_with_content_only():
    """Test that assistant messages with only content (no tool_calls) are processed correctly."""
    # Arrange
    messages_content_only = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        }
    ]
    
    # Act
    result = adapt_messages_to_generic_oci_standard(messages_content_only)
    
    # Assert
    assert len(result) == 2
    
    # Check assistant message with content only
    assert result[1].role == "ASSISTANT"
    assert result[1].content[0].type == "TEXT"
    assert result[1].content[0].text == "Hello! How can I help you today?"
    assert result[1].toolCalls is None

def test_adapt_messages_tool_id_tracking():
    """Test that tool call IDs are properly tracked for validation."""
    # Arrange
    messages = [
        {"role": "user", "content": "Test"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "test_func",
                        "arguments": '{"param": "value"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "content": "Result",
            "tool_call_id": "call_123"
        }
    ]
    
    # Act
    result = adapt_messages_to_generic_oci_standard(messages)
    
    # Assert
    # Tool call should be processed and ID should be available for validation
    assert result[1].toolCalls[0].id == "call_123"
    
    # Tool response should reference the same ID
    tool_response_text = result[2].content[0].text
    # Tool response text is just the content, tool_call_id is separate
    assert tool_response_text == "Result"  # The actual content
    assert result[2].toolCallId == "call_123"  # Tool call ID is in separate field

def test_adapt_messages_multiple_tool_calls():
    """Test that multiple tool calls in a single message are processed correctly."""
    # Arrange
    messages = [
        {"role": "user", "content": "Test multiple tools"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "func1",
                        "arguments": '{"param": "value1"}'
                    }
                },
                {
                    "id": "call_2", 
                    "type": "function",
                    "function": {
                        "name": "func2",
                        "arguments": '{"param": "value2"}'
                    }
                }
            ]
        }
    ]
    
    # Act
    result = adapt_messages_to_generic_oci_standard(messages)
    
    # Assert
    assert len(result) == 2
    assert result[1].role == "ASSISTANT"
    assert len(result[1].toolCalls) == 2
    assert result[1].toolCalls[0].id == "call_1"
    assert result[1].toolCalls[1].id == "call_2"

