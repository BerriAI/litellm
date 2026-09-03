"""Streaming handler for the Amazon Bedrock Mantle OpenAI-compatible surface.

Mantle emits streaming ``tool_calls[].index`` values that start at 1 for the
first tool call, whereas the OpenAI spec (and every client that aggregates the
deltas, including the OpenAI SDK) expects a 0-based index. This handler shifts
the provider's indices down so the first tool call reported in a stream is
index 0, leaving already-0-based streams untouched.
"""

from typing import Optional

from litellm.llms.openai.chat.gpt_transformation import (
    OpenAIChatCompletionStreamingHandler,
)
from litellm.types.utils import ModelResponseStream


class BedrockMantleChatCompletionStreamingHandler(OpenAIChatCompletionStreamingHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Offset of the first tool_call index seen in the stream; every index is
        # rebased against it so the first tool call becomes index 0.
        self._tool_call_index_offset: Optional[int] = None

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        parsed = super().chunk_parser(chunk)
        for choice in parsed.choices:
            tool_calls = choice.delta.tool_calls
            if not tool_calls:
                continue
            offset = self._tool_call_index_offset
            if offset is None:
                offset = min(tool_call.index for tool_call in tool_calls)
                self._tool_call_index_offset = offset
            for tool_call in tool_calls:
                tool_call.index -= offset
        return parsed
