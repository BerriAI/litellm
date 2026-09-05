"""
Streaming handler for Amazon Bedrock Mantle (OpenAI-compatible API).

Bedrock Mantle emits a fresh ``id`` on every SSE chunk, which violates the
OpenAI streaming contract that all chunks in one response share a single
``id``. Clients that validate this (e.g. the openai-go SDK's
``ChatCompletionAccumulator``) drop every chunk after the first, losing
content and tool-call arguments. We pin the id from the first chunk that
carries one and reuse it for the rest of the stream.
"""

from typing import Optional

from litellm.llms.openai.chat.gpt_transformation import (
    OpenAIChatCompletionStreamingHandler,
)
from litellm.types.utils import ModelResponseStream


class BedrockMantleChatCompletionStreamingHandler(OpenAIChatCompletionStreamingHandler):
    def __init__(self, streaming_response, sync_stream: bool, json_mode: Optional[bool] = False):
        super().__init__(streaming_response=streaming_response, sync_stream=sync_stream, json_mode=json_mode)
        self._response_id: Optional[str] = None

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        parsed = super().chunk_parser(chunk)
        raw_id = chunk.get("id")
        if self._response_id is None and raw_id:
            self._response_id = raw_id
        if self._response_id is not None:
            parsed.id = self._response_id
        return parsed
