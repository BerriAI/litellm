import json
from abc import abstractmethod
from typing import List, Optional, Tuple

import litellm
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.types.utils import (
    ChatCompletionToolCallChunk,
    ChatCompletionUsageBlock,
    GenericStreamingChunk,
    ModelResponse,
)


class FakeStreamResponseIterator:
    def __init__(self, model_response, json_mode: Optional[bool] = False):
        self.model_response = model_response
        self.json_mode = json_mode
        self.is_done = False

    # Sync iterator
    def __iter__(self):
        return self

    def _handle_json_mode_chunk(
        self, text: str, tool_calls: Optional[List[ChatCompletionToolCallChunk]]
    ) -> Tuple[str, Optional[ChatCompletionToolCallChunk]]:
        """
        If JSON mode is enabled, convert the tool call to a message.

        Bedrock returns the JSON schema as part of the tool call
        OpenAI returns the JSON schema as part of the content, this handles placing it in the content

        Args:
            text: str
            tool_use: Optional[ChatCompletionToolCallChunk]
        Returns:
            Tuple[str, Optional[ChatCompletionToolCallChunk]]

            text: The text to use in the content
            tool_use: The ChatCompletionToolCallChunk to use in the chunk response
        """
        tool_use: Optional[ChatCompletionToolCallChunk] = None
        if self.json_mode is True and tool_calls is not None:
            message = litellm.AnthropicConfig()._convert_tool_response_to_message(
                tool_calls=tool_calls
            )
            if message is not None:
                text = message.content or ""
                tool_use = None
        elif tool_calls is not None and len(tool_calls) > 0:
            tool_use = tool_calls[0]
        return text, tool_use

    def chunk_parser(self, chunk: ModelResponse) -> GenericStreamingChunk:
        try:
            chunk_usage: litellm.Usage = getattr(chunk, "usage")
            text = chunk.choices[0].message.content or ""  # type: ignore
            tool_use = None
            if self.json_mode is True:
                text, tool_use = self._handle_json_mode_chunk(
                    text=text,
                    tool_calls=chunk.choices[0].message.tool_calls,  # type: ignore
                )
            processed_chunk = GenericStreamingChunk(
                text=text,
                tool_use=tool_use,
                is_finished=True,
                finish_reason=map_finish_reason(
                    finish_reason=chunk.choices[0].finish_reason or ""
                ),
                usage=ChatCompletionUsageBlock(
                    prompt_tokens=chunk_usage.prompt_tokens,
                    completion_tokens=chunk_usage.completion_tokens,
                    total_tokens=chunk_usage.total_tokens,
                ),
                index=0,
            )
            return processed_chunk
        except Exception as e:
            raise ValueError(f"Failed to decode chunk: {chunk}. Error: {e}")

    def __next__(self):
        if self.is_done:
            raise StopIteration
        self.is_done = True
        return self.chunk_parser(self.model_response)

    # Async iterator
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.is_done:
            raise StopAsyncIteration
        self.is_done = True
        return self.chunk_parser(self.model_response)
