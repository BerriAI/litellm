"""
Test script to understand what the /cursor/chat/completions endpoint returns for tool calls.
This simulates what Cursor would receive.
"""

import json
from typing import Any, Dict, List

# Mock the Responses API streaming events that would come from a model with tool calls
MOCK_TOOL_CALL_STREAM_EVENTS = [
    # Initial response created
    {"type": "response.created", "response": {"id": "resp_123", "status": "in_progress"}},
    
    # Output item added - function call
    {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_123",
            "call_id": "call_abc123",
            "name": "read_file",
            "arguments": "",
            "status": "in_progress"
        }
    },
    
    # Function arguments delta
    {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"path":'},
    {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": ' "/src/app.py"}'},
    
    # Function call done
    {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_123",
            "call_id": "call_abc123",
            "name": "read_file",
            "arguments": '{"path": "/src/app.py"}',
            "status": "completed"
        }
    },
    
    # Response completed
    {"type": "response.completed", "response": {"id": "resp_123", "status": "completed"}}
]


def test_chunk_parser():
    """Test the chunk parser directly to see what it produces."""
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        OpenAiResponsesToChatCompletionStreamIterator,
    )
    
    # Create the iterator (we won't actually iterate, just use chunk_parser)
    iterator = OpenAiResponsesToChatCompletionStreamIterator(
        streaming_response=iter([]),  # empty, we just want to use chunk_parser
        sync_stream=True,
        json_mode=False
    )
    
    print("=" * 60)
    print("Testing chunk_parser with mock Responses API events")
    print("=" * 60)
    
    for i, event in enumerate(MOCK_TOOL_CALL_STREAM_EVENTS):
        print(f"\n--- Event {i+1}: {event.get('type')} ---")
        print(f"Input: {json.dumps(event, indent=2)}")
        
        try:
            result = iterator.chunk_parser(event)
            print(f"Output type: {type(result).__name__}")
            
            if hasattr(result, 'model_dump'):
                print(f"Output: {json.dumps(result.model_dump(), indent=2, default=str)}")
            elif isinstance(result, dict):
                print(f"Output: {json.dumps(result, indent=2, default=str)}")
            else:
                print(f"Output: {result}")
        except Exception as e:
            print(f"Error: {e}")


def test_expected_openai_streaming_format():
    """Show what OpenAI chat completions streaming format looks like for tool calls."""
    print("\n" + "=" * 60)
    print("Expected OpenAI Chat Completions Streaming Format for Tool Calls")
    print("=" * 60)
    
    # This is what OpenAI actually sends for streaming tool calls
    expected_chunks = [
        # First chunk - establishes the tool call
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": ""
                        }
                    }]
                },
                "finish_reason": None
            }]
        },
        # Arguments delta chunks
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "arguments": '{"path":'
                        }
                    }]
                },
                "finish_reason": None
            }]
        },
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "arguments": ' "/src/app.py"}'
                        }
                    }]
                },
                "finish_reason": None
            }]
        },
        # Final chunk with finish_reason
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls"
            }]
        }
    ]
    
    for i, chunk in enumerate(expected_chunks):
        print(f"\nChunk {i+1}:")
        print(json.dumps(chunk, indent=2))


if __name__ == "__main__":
    test_chunk_parser()
    test_expected_openai_streaming_format()
