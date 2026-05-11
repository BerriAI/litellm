"""
SSE Stream Iterator for Vertex AI Agent Engine.

Handles Server-Sent Events (SSE) streaming responses from Vertex AI Reasoning Engines.
"""

import json
from typing import Any, List, Optional, Union

from litellm._uuid import uuid
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.types.llms.openai import (
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionUsageBlock,
)
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

    @staticmethod
    def _extract_parts_from_chunk(
        chunk: dict,
    ) -> tuple[Optional[str], List[ChatCompletionToolCallChunk]]:
        """
        Walk the ``content.parts`` array and split it into:
          - the first text part (if any)
          - any function_call parts converted to OpenAI tool_calls

        Vertex Agent Engine returns parts in either ``functionCall`` (camelCase,
        REST API) or ``function_call`` (snake_case, Python SDK) form.
        """
        text: Optional[str] = None
        tool_calls: List[ChatCompletionToolCallChunk] = []

        content = chunk.get("content") or {}
        parts = content.get("parts") or []

        for part in parts:
            if not isinstance(part, dict):
                continue

            if text is None and "text" in part:
                text = part["text"]
                continue

            function_call = part.get("functionCall") or part.get("function_call")
            if function_call:
                call_id = function_call.get("id") or f"call_{uuid.uuid4()}"
                tool_calls.append(
                    ChatCompletionToolCallChunk(
                        id=call_id,
                        type="function",
                        function=ChatCompletionToolCallFunctionChunk(
                            name=function_call.get("name", ""),
                            arguments=json.dumps(function_call.get("args") or {}),
                        ),
                        index=len(tool_calls),
                    )
                )

        return text, tool_calls

    def chunk_parser(
        self, chunk: dict
    ) -> Union[GenericStreamingChunk, ModelResponseStream]:
        """
        Parse a Vertex Agent Engine response chunk into ModelResponseStream.

        Vertex Agent Engine emits one SSE event per ADK action (e.g. an inner
        ``transfer_to_agent`` call, an MCP tool call, and the final text reply).
        Each event carries ``finish_reason: "STOP"`` because ``STOP`` is the
        Gemini-level terminator for that single action — it does NOT mean the
        Agent Engine stream is finished. The SSE stream ending is the only true
        end-of-response signal.

        We therefore only surface ``finish_reason`` when the chunk has
        user-facing text content. Function-call and thought-only chunks must
        keep ``finish_reason=None`` so the downstream stream wrapper does not
        close the stream after the first inner action and drop the actual
        response (see issue #19121).
        """
        text, tool_calls = self._extract_parts_from_chunk(chunk)

        finish_reason: Optional[str] = None
        raw_finish_reason = chunk.get("finish_reason")
        if raw_finish_reason and text is not None:
            finish_reason = (
                "stop" if raw_finish_reason == "STOP" else raw_finish_reason.lower()
            )

        usage = None
        usage_metadata = chunk.get("usage_metadata") or {}
        if usage_metadata:
            usage = ChatCompletionUsageBlock(
                prompt_tokens=usage_metadata.get("prompt_token_count", 0),
                completion_tokens=usage_metadata.get("candidates_token_count", 0),
                total_tokens=usage_metadata.get("total_token_count", 0),
            )

        delta_kwargs: dict = {}
        if text is not None:
            delta_kwargs["content"] = text
            delta_kwargs["role"] = "assistant"
        elif tool_calls:
            delta_kwargs["tool_calls"] = tool_calls
            delta_kwargs["role"] = "assistant"
        else:
            delta_kwargs["content"] = None

        return ModelResponseStream(
            choices=[
                StreamingChoices(
                    finish_reason=finish_reason,
                    index=0,
                    delta=Delta(**delta_kwargs),
                )
            ],
            usage=usage,
        )
