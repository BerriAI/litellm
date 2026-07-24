"""
Stream iterator for the Command Code API.

The /alpha/generate endpoint returns a newline-delimited stream of typed
JSON events (text-delta, reasoning-delta, tool-call, finish, error, ...).
This iterator translates them into OpenAI-style ModelResponseStream chunks.
Supports both sync and async iteration.
"""

import json
from typing import Optional

import httpx

from litellm.llms.command_code.common_utils import (
    CommandCodeError,
    map_command_code_finish_reason,
    parse_stream_event_line,
    parse_tool_call_input,
    usage_from_finish_event,
)
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)


class CommandCodeSSEStreamIterator:
    """Iterates Command Code stream events as ModelResponseStream chunks."""

    def __init__(self, response: httpx.Response, model: str):
        self.response = response
        self.model = model
        self.finished = False
        self.tool_call_index = 0
        self.line_iterator = None
        self.async_line_iterator = None

    def __iter__(self):
        """Initialize sync iteration. Idempotent: CustomStreamWrapper may
        re-enter iteration for every chunk."""
        if self.line_iterator is None:
            self.line_iterator = self.response.iter_lines()
        return self

    def __aiter__(self):
        """Initialize async iteration. Idempotent: CustomStreamWrapper may
        re-enter iteration for every chunk."""
        if self.async_line_iterator is None:
            self.async_line_iterator = self.response.aiter_lines()
        return self

    def _build_chunk(
        self,
        delta: Delta,
        finish_reason: Optional[str] = None,
        usage: Optional[Usage] = None,
    ) -> ModelResponseStream:
        chunk = ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=delta, finish_reason=finish_reason)],
            model=self.model,
            usage=usage,
        )
        return chunk

    def _process_line(self, line: str) -> Optional[ModelResponseStream]:
        """Translate one stream line into a chunk, or None for no-op events."""
        event = parse_stream_event_line(line)
        if event is None:
            return None

        event_type = event.get("type")

        if event_type == "text-delta":
            return self._build_chunk(Delta(content=event.get("text") or "", role="assistant"))

        if event_type == "reasoning-delta":
            return self._build_chunk(Delta(reasoning_content=event.get("text") or "", role="assistant"))

        if event_type == "tool-call":
            arguments = event.get("input")
            if arguments is None:
                arguments = event.get("args")
            if arguments is None:
                arguments = event.get("arguments")
            tool_call = ChatCompletionDeltaToolCall(
                id=event.get("toolCallId") or "",
                type="function",
                index=self.tool_call_index,
                function=Function(
                    name=event.get("toolName") or "",
                    arguments=json.dumps(parse_tool_call_input(arguments)),
                ),
            )
            self.tool_call_index += 1
            return self._build_chunk(Delta(tool_calls=[tool_call], role="assistant"))

        if event_type == "finish":
            self.finished = True
            usage = None
            total_usage = event.get("totalUsage")
            if isinstance(total_usage, dict):
                usage = usage_from_finish_event(total_usage)
            return self._build_chunk(
                Delta(),
                finish_reason=map_command_code_finish_reason(event.get("finishReason")),
                usage=usage,
            )

        if event_type == "error":
            error = event.get("error")
            if isinstance(error, dict):
                message = error.get("message") or "Command Code stream error"
            elif isinstance(error, str):
                message = error
            else:
                message = "Command Code stream error"
            raise CommandCodeError(status_code=500, message=message)

        # reasoning-start / reasoning-end / tool-result and unknown events
        # carry no OpenAI-visible payload.
        return None

    def _final_chunk(self) -> ModelResponseStream:
        """Terminal chunk for streams that end without a finish event."""
        self.finished = True
        return self._build_chunk(Delta(), finish_reason="stop")

    def __next__(self) -> ModelResponseStream:
        if self.line_iterator is None:
            self.line_iterator = self.response.iter_lines()

        try:
            for line in self.line_iterator:
                chunk = self._process_line(line)
                if chunk is not None:
                    return chunk
        except (httpx.StreamConsumed, httpx.StreamClosed):
            pass

        if not self.finished:
            return self._final_chunk()
        raise StopIteration

    async def __anext__(self) -> ModelResponseStream:
        if self.async_line_iterator is None:
            self.async_line_iterator = self.response.aiter_lines()

        try:
            async for line in self.async_line_iterator:
                chunk = self._process_line(line)
                if chunk is not None:
                    return chunk
        except (httpx.StreamConsumed, httpx.StreamClosed):
            pass

        if not self.finished:
            return self._final_chunk()
        raise StopAsyncIteration
