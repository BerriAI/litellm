"""
SSE Stream Iterator for Vertex AI Agent Engine.

Handles Server-Sent Events (SSE) streaming responses from Vertex AI Reasoning Engines.
"""

from typing import Any, Union

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.types.llms.openai import ChatCompletionUsageBlock
from litellm.types.utils import (
    Delta,
    GenericStreamingChunk,
    ModelResponseStream,
    StreamingChoices,
)


class VertexAgentEngineResponseIterator(BaseModelResponseIterator):
    """
    Iterator for Vertex Agent Engine SSE streaming responses.

    Uses BaseModelResponseIterator which handles sync/async iteration.
    We just need to implement chunk_parser to parse Vertex Agent Engine response format.
    """

    def __init__(self, streaming_response: Any, sync_stream: bool) -> None:
        super().__init__(streaming_response=streaming_response, sync_stream=sync_stream)

    def chunk_parser(
        self, chunk: dict
    ) -> Union[GenericStreamingChunk, ModelResponseStream]:
        """
        Parse a Vertex Agent Engine response chunk into ModelResponseStream.

        Vertex Agent Engine response format:
        {
            "content": {
                "parts": [{"text": "..."}],
                "role": "model"
            },
            "finish_reason": "STOP",
            "usage_metadata": {
                "prompt_token_count": 100,
                "candidates_token_count": 50,
                "total_token_count": 150
            }
        }
        """
        # Extract text from content.parts
        text = None
        content = chunk.get("content", {})
        parts = content.get("parts", [])
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                text = part["text"]
                break

        # Extract finish_reason
        finish_reason = None
        raw_finish_reason = chunk.get("finish_reason")
        if raw_finish_reason == "STOP":
            finish_reason = "stop"
        elif raw_finish_reason:
            finish_reason = raw_finish_reason.lower()

        # Extract usage from usage_metadata
        usage = None
        usage_metadata = chunk.get("usage_metadata", {})
        if usage_metadata:
            usage = ChatCompletionUsageBlock(
                prompt_tokens=usage_metadata.get("prompt_token_count", 0),
                completion_tokens=usage_metadata.get("candidates_token_count", 0),
                total_tokens=usage_metadata.get("total_token_count", 0),
            )

        # Return ModelResponseStream (OpenAI-compatible chunk)
        return ModelResponseStream(
            choices=[
                StreamingChoices(
                    finish_reason=finish_reason,
                    index=0,
                    delta=Delta(
                        content=text,
                        role="assistant" if text else None,
                    ),
                )
            ],
            usage=usage,
        )
