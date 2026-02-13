"""
Fake Streaming Iterator for Anthropic Messages

This module provides a fake streaming iterator that converts non-streaming
Anthropic Messages responses into proper streaming format.

Used when WebSearch interception converts stream=True to stream=False but
the LLM doesn't make a tool call, and we need to return a stream to the user.
"""

import json
from typing import Any, Dict, List, cast

from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)


class FakeAnthropicMessagesStreamIterator:
    """
    Fake streaming iterator for Anthropic Messages responses.
    
    Used when we need to convert a non-streaming response to a streaming format,
    such as when WebSearch interception converts stream=True to stream=False but
    the LLM doesn't make a tool call.
    
    This creates a proper Anthropic-style streaming response with multiple events:
    - message_start
    - content_block_start (for each content block)
    - content_block_delta (for text content, chunked)
    - content_block_stop
    - message_delta (for usage)
    - message_stop
    """
    
    def __init__(self, response: AnthropicMessagesResponse):
        self.response = response
        self.chunks = self._create_streaming_chunks()
        self.current_index = 0
    
    def _create_streaming_chunks(self) -> List[bytes]:
        """Convert the non-streaming response to streaming chunks"""
        chunks = []
        
        # Cast response to dict for easier access
        response_dict = cast(Dict[str, Any], self.response)
        
        # 1. message_start event
        usage = response_dict.get("usage", {})
        message_start = {
            "type": "message_start",
            "message": {
                "id": response_dict.get("id"),
                "type": "message",
                "role": response_dict.get("role", "assistant"),
                "model": response_dict.get("model"),
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": usage.get("input_tokens", 0) if usage else 0,
                    "output_tokens": 0
                }
            }
        }
        chunks.append(f"event: message_start\ndata: {json.dumps(message_start)}\n\n".encode())
        
        # 2-4. For each content block, send start/delta/stop events
        content_blocks = response_dict.get("content", [])
        if content_blocks:
            for index, block in enumerate(content_blocks):
                # Cast block to dict for easier access
                block_dict = cast(Dict[str, Any], block)
                block_type = block_dict.get("type")
                
                if block_type == "text":
                    # content_block_start
                    content_block_start = {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "text",
                            "text": ""
                        }
                    }
                    chunks.append(f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode())
                    
                    # content_block_delta (send full text as one delta for simplicity)
                    text = block_dict.get("text", "")
                    content_block_delta = {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "text_delta",
                            "text": text
                        }
                    }
                    chunks.append(f"event: content_block_delta\ndata: {json.dumps(content_block_delta)}\n\n".encode())
                    
                    # content_block_stop
                    content_block_stop = {
                        "type": "content_block_stop",
                        "index": index
                    }
                    chunks.append(f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n".encode())
                
                elif block_type == "thinking":
                    # content_block_start for thinking
                    content_block_start = {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "thinking",
                            "thinking": "",
                            "signature": ""
                        }
                    }
                    chunks.append(f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode())
                    
                    # content_block_delta for thinking text
                    thinking_text = block_dict.get("thinking", "")
                    if thinking_text:
                        content_block_delta = {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "thinking_delta",
                                "thinking": thinking_text
                            }
                        }
                        chunks.append(f"event: content_block_delta\ndata: {json.dumps(content_block_delta)}\n\n".encode())
                    
                    # content_block_delta for signature (if present)
                    signature = block_dict.get("signature", "")
                    if signature:
                        signature_delta = {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "signature_delta",
                                "signature": signature
                            }
                        }
                        chunks.append(f"event: content_block_delta\ndata: {json.dumps(signature_delta)}\n\n".encode())
                    
                    # content_block_stop
                    content_block_stop = {
                        "type": "content_block_stop",
                        "index": index
                    }
                    chunks.append(f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n".encode())
                
                elif block_type == "redacted_thinking":
                    # content_block_start for redacted_thinking
                    content_block_start = {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "redacted_thinking"
                        }
                    }
                    chunks.append(f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode())
                    
                    # content_block_stop (no delta for redacted thinking)
                    content_block_stop = {
                        "type": "content_block_stop",
                        "index": index
                    }
                    chunks.append(f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n".encode())
                
                elif block_type == "tool_use":
                    # content_block_start
                    content_block_start = {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": block_dict.get("id"),
                            "name": block_dict.get("name"),
                            "input": {}
                        }
                    }
                    chunks.append(f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode())
                    
                    # content_block_delta (send input as JSON delta)
                    input_data = block_dict.get("input", {})
                    content_block_delta = {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(input_data)
                        }
                    }
                    chunks.append(f"event: content_block_delta\ndata: {json.dumps(content_block_delta)}\n\n".encode())
                    
                    # content_block_stop
                    content_block_stop = {
                        "type": "content_block_stop",
                        "index": index
                    }
                    chunks.append(f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n".encode())
        
        # 5. message_delta event (with final usage and stop_reason)
        message_delta = {
            "type": "message_delta",
            "delta": {
                "stop_reason": response_dict.get("stop_reason"),
                "stop_sequence": response_dict.get("stop_sequence")
            },
            "usage": {
                "output_tokens": usage.get("output_tokens", 0) if usage else 0
            }
        }
        chunks.append(f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n".encode())
        
        # 6. message_stop event
        message_stop = {
            "type": "message_stop",
            "usage": usage if usage else {}
        }
        chunks.append(f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n".encode())
        
        return chunks
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.current_index >= len(self.chunks):
            raise StopAsyncIteration
        
        chunk = self.chunks[self.current_index]
        self.current_index += 1
        return chunk
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_index >= len(self.chunks):
            raise StopIteration
        
        chunk = self.chunks[self.current_index]
        self.current_index += 1
        return chunk
