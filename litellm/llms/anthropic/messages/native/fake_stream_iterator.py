"""
Fake Streaming Iterator for Anthropic Messages

This module provides a fake streaming iterator that converts non-streaming
Anthropic Messages responses into proper streaming format.

Used when WebSearch interception converts stream=True to stream=False but
the LLM doesn't make a tool call, and we need to return a stream to the user.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Final, cast

from pydantic import TypeAdapter

from litellm.llms.anthropic.messages.native.utils import anthropic_message_delta
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)

_CITATIONS: Final = TypeAdapter(tuple[object, ...])


def _citations_of(block: Mapping[str, object]) -> tuple[object, ...]:
    citations: Final = block.get("citations")
    return _CITATIONS.validate_python(citations) if isinstance(citations, list) else ()


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

    def __init__(
        self,
        response: AnthropicMessagesResponse,
        message_delta_fields: Mapping[str, object] | None = None,
    ):
        self.response = response
        self.message_delta_fields = message_delta_fields
        self.chunks = self._create_streaming_chunks()
        self.current_index = 0

    def _create_content_block_chunks(self, block_dict: Mapping[str, object], index: int) -> list[bytes]:
        """Build SSE chunks for a single content block."""
        chunks: Final = list[bytes]()
        block_type: Final = block_dict.get("type")

        if block_type == "text":
            citations: Final = _citations_of(block_dict)
            content_block_start = {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": "", "citations": ()}
                if citations
                else {"type": "text", "text": ""},
            }
            chunks.append(f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode())
            citation_deltas: Final = (
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "citations_delta", "citation": citation},
                }
                for citation in citations
            )
            chunks.extend(
                f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n".encode() for delta in citation_deltas
            )
            text: Final = block_dict.get("text", "")
            content_block_delta = {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": text},
            }
            chunks.append(f"event: content_block_delta\ndata: {json.dumps(content_block_delta)}\n\n".encode())

        elif block_type == "thinking":
            content_block_start = {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            }
            chunks.append(f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode())
            thinking_text: Final = block_dict.get("thinking", "")
            if thinking_text:
                content_block_delta = {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": thinking_text},
                }
                chunks.append(f"event: content_block_delta\ndata: {json.dumps(content_block_delta)}\n\n".encode())
            signature: Final = block_dict.get("signature", "")
            if signature:
                signature_delta: Final = {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "signature_delta", "signature": signature},
                }
                chunks.append(f"event: content_block_delta\ndata: {json.dumps(signature_delta)}\n\n".encode())

        elif block_type in ("tool_use", "server_tool_use"):
            empty_input: Final[Mapping[str, object]] = {}
            tool_use_start: Final = {
                "type": "content_block_start",
                "index": index,
                "content_block": {**block_dict, "input": empty_input},
            }
            chunks.append(f"event: content_block_start\ndata: {json.dumps(tool_use_start)}\n\n".encode())
            tool_use_delta: Final = {
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block_dict.get("input", {})),
                },
            }
            chunks.append(f"event: content_block_delta\ndata: {json.dumps(tool_use_delta)}\n\n".encode())

        else:
            passthrough_start: Final = {
                "type": "content_block_start",
                "index": index,
                "content_block": block_dict,
            }
            chunks.append(f"event: content_block_start\ndata: {json.dumps(passthrough_start)}\n\n".encode())

        content_block_stop: Final = {"type": "content_block_stop", "index": index}
        chunks.append(f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n".encode())
        return chunks

    def _create_streaming_chunks(self) -> list[bytes]:
        """Convert the non-streaming response to streaming chunks"""
        chunks: Final = []

        response_dict: Final[Mapping[str, object]] = self.response

        # 1. message_start event
        usage: Final = self.response.get("usage")
        message_start: Final = {
            "type": "message_start",
            "message": {
                "id": self.response.get("id"),
                "type": "message",
                "role": self.response.get("role", "assistant"),
                "model": self.response.get("model"),
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": usage.get("input_tokens", 0) if usage else 0,
                    "output_tokens": 0,
                },
            },
        }
        chunks.append(f"event: message_start\ndata: {json.dumps(message_start)}\n\n".encode())

        # 2-4. For each content block, send start/delta/stop events
        content_blocks: Final = cast(Sequence[Mapping[str, object]], response_dict.get("content", []))
        for index, block_dict in enumerate(content_blocks):
            chunks.extend(self._create_content_block_chunks(block_dict, index))

        # 5. message_delta event (with final usage and stop_reason)
        # Include cache usage fields so clients that only read message_delta
        # (like Claude Code's SDK) see the full input token breakdown.
        delta_usage: Final[dict[str, int]] = {
            "output_tokens": usage.get("output_tokens", 0) if usage else 0,
        }
        if usage:
            input_tokens: Final = usage.get("input_tokens")
            if input_tokens is not None:
                delta_usage["input_tokens"] = input_tokens
            cache_creation_input_tokens: Final = usage.get("cache_creation_input_tokens")
            if cache_creation_input_tokens is not None:
                delta_usage["cache_creation_input_tokens"] = cache_creation_input_tokens
            cache_read_input_tokens: Final = usage.get("cache_read_input_tokens")
            if cache_read_input_tokens is not None:
                delta_usage["cache_read_input_tokens"] = cache_read_input_tokens
        message_delta: Final = {
            "type": "message_delta",
            "delta": anthropic_message_delta(response_dict, self.message_delta_fields),
            "usage": delta_usage,
        }
        chunks.append(f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n".encode())

        # 6. message_stop event
        message_stop: Final = {"type": "message_stop", "usage": usage if usage else {}}
        chunks.append(f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n".encode())

        return chunks

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.current_index >= len(self.chunks):
            raise StopAsyncIteration

        chunk: Final = self.chunks[self.current_index]
        self.current_index += 1
        return chunk

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.chunks):
            raise StopIteration

        chunk: Final = self.chunks[self.current_index]
        self.current_index += 1
        return chunk
