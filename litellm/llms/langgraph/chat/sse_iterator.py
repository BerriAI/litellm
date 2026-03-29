"""
SSE Stream Iterator for LangGraph.

Handles Server-Sent Events (SSE) streaming responses from LangGraph.
"""

import json
import uuid
from typing import TYPE_CHECKING, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

if TYPE_CHECKING:
    pass


class LangGraphSSEStreamIterator:
    """
    Iterator for LangGraph SSE streaming responses.
    Supports both sync and async iteration.

    LangGraph stream format with stream_mode="messages-tuple":
    Each SSE event is a tuple: (event_type, data)
    Common event types: "messages", "metadata"
    """

    def __init__(self, response: httpx.Response, model: str):
        self.response = response
        self.model = model
        self.finished = False
        self._has_received_messages = False
        self.line_iterator = None
        self.async_line_iterator = None
        self._current_event_type: Optional[str] = None

    def __iter__(self):
        """Initialize sync iteration."""
        self.line_iterator = self.response.iter_lines()
        return self

    def __aiter__(self):
        """Initialize async iteration."""
        self.async_line_iterator = self.response.aiter_lines()
        return self

    def _parse_sse_line(self, line: str) -> Optional[ModelResponseStream]:
        """
        Parse a single SSE line and return a ModelResponse chunk if applicable.

        LangGraph SSE format can vary:
        - data: [...] (tuple format)
        - event: ...\ndata: ...
        """
        line = line.strip()
        if not line:
            self._current_event_type = None
            return None

        # Handle SSE event: lines (standard SSE - event type in header)
        if line.startswith("event:"):
            self._current_event_type = line[len("event:"):].strip()
            return None

        # Handle SSE data lines
        if line.startswith("data:"):
            json_str = line[5:].strip()
            if not json_str:
                return None

            try:
                data = json.loads(json_str)
                event_type = self._current_event_type
                self._current_event_type = None
                return self._process_data(data, event_type)
            except json.JSONDecodeError:
                verbose_logger.debug(f"Skipping non-JSON SSE line: {line[:100]}")
                return None

        return None

    def _process_data(self, data, event_type: Optional[str] = None) -> Optional[ModelResponseStream]:
        """
        Process parsed data from SSE stream.

        LangGraph uses tuple format: [event_type, payload]
        """
        # Standard SSE: event type from header, data is [msg_obj, meta_obj]
        if event_type == "messages" and isinstance(data, list) and len(data) >= 1:
            return self._process_messages_event(data)
        if event_type == "metadata":
            if isinstance(data, dict):
                return self._process_metadata_event(data)
            elif isinstance(data, list) and len(data) >= 1:
                return self._process_metadata_event(data[0] if isinstance(data[0], dict) else data)

        # Handle tuple format: ["messages", ...] (backward compatibility)
        if isinstance(data, list) and len(data) >= 2:
            evt = data[0]
            payload = data[1]

            if evt == "messages":
                return self._process_messages_event(payload)
            elif evt == "metadata":
                # Metadata event, might contain usage info
                return self._process_metadata_event(payload)

        # Handle dict format (alternative response format)
        elif isinstance(data, dict):
            if "content" in data:
                self._has_received_messages = True
                text, reasoning = self._extract_text_from_ai_content(
                    data.get("content", "")
                )
                tools = self._normalize_tool_calls_for_delta(data.get("tool_calls"))
                return self._create_delta_chunk(
                    text,
                    tool_calls=tools,
                    reasoning_content=reasoning,
                )
            elif "messages" in data:
                messages = data.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, dict) and last_msg.get("type") in (
                        "ai",
                        "AIMessageChunk",
                    ):
                        self._has_received_messages = True
                        text, reasoning = self._extract_text_from_ai_content(
                            last_msg.get("content", "")
                        )
                        tools = self._normalize_tool_calls_for_delta(
                            last_msg.get("tool_calls")
                        )
                        return self._create_delta_chunk(
                            text,
                            tool_calls=tools,
                            reasoning_content=reasoning,
                        )

        return None

    def _extract_text_from_ai_content(self, content):
        """Flatten string or block-list AI content; return (text, reasoning_or_none)."""
        if content is None:
            return "", None
        if isinstance(content, str):
            return content, None
        if isinstance(content, list):
            text_parts = []
            reasoning_parts = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text" and "text" in block:
                        text_parts.append(str(block["text"]))
                    elif btype in ("reasoning", "thinking"):
                        r = (
                            block.get("reasoning")
                            or block.get("text")
                            or block.get("thinking")
                        )
                        if r is not None:
                            reasoning_parts.append(str(r))
                    elif "text" in block:
                        text_parts.append(str(block["text"]))
            reasoning = "".join(reasoning_parts) if reasoning_parts else None
            return "".join(text_parts), reasoning
        return str(content), None

    def _normalize_tool_calls_for_delta(self, tool_calls):
        """Map LangGraph / LangChain tool call dicts to OpenAI streaming delta shape."""
        if not tool_calls or not isinstance(tool_calls, list):
            return None
        out = []
        for idx, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            fn_obj = tc.get("function")
            if isinstance(fn_obj, dict):
                name = fn_obj.get("name", "")
                args = fn_obj.get("arguments", "{}")
                if isinstance(args, dict):
                    args = json.dumps(args)
                else:
                    args = str(args)
                out.append(
                    {
                        "id": tc.get("id"),
                        "type": tc.get("type") or "function",
                        "function": {"name": name, "arguments": args},
                        "index": tc.get("index", idx),
                    }
                )
                continue
            name = tc.get("name")
            if name is None and isinstance(fn_obj, str):
                name = fn_obj
            args = tc.get("args", tc.get("arguments", {}))
            if isinstance(args, dict):
                args = json.dumps(args)
            elif args is None:
                args = "{}"
            else:
                args = str(args)
            if not name:
                continue
            out.append(
                {
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                    "index": tc.get("index", idx),
                }
            )
        return out or None

    def _create_delta_chunk(
        self,
        content: str,
        tool_calls=None,
        reasoning_content=None,
    ) -> ModelResponseStream:
        """Build a streaming chunk with optional content, tool_calls, and reasoning."""
        delta_kwargs = {"role": "assistant"}
        if content:
            delta_kwargs["content"] = content
        else:
            delta_kwargs["content"] = None
        if reasoning_content:
            delta_kwargs["reasoning_content"] = reasoning_content
        if tool_calls:
            delta_kwargs["tool_calls"] = tool_calls

        chunk = ModelResponseStream(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=0,
            model=self.model,
            object="chat.completion.chunk",
        )
        chunk.choices = [
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(**delta_kwargs),
            )
        ]
        return chunk

    def _process_messages_event(self, payload) -> Optional[ModelResponseStream]:
        """
        Process a messages event from the stream.

        payload format: [[message_object, metadata], ...]
        """
        if isinstance(payload, list):
            for item in payload:
                msg = None
                if isinstance(item, list) and len(item) >= 1:
                    msg = item[0]
                elif isinstance(item, dict):
                    msg = item

                if not isinstance(msg, dict):
                    continue

                msg_type = msg.get("type", "")
                if msg_type not in ("ai", "AIMessageChunk"):
                    continue

                self._has_received_messages = True
                text, reasoning = self._extract_text_from_ai_content(msg.get("content"))
                tools = self._normalize_tool_calls_for_delta(msg.get("tool_calls"))
                if not text and reasoning is None and not tools:
                    continue

                return self._create_delta_chunk(
                    text,
                    tool_calls=tools,
                    reasoning_content=reasoning,
                )

        return None

    def _process_metadata_event(self, payload) -> Optional[ModelResponseStream]:
        """
        Process a metadata event, which may signal the end of the stream.
        """
        if isinstance(payload, dict):
            if "run_id" in payload:
                if not self._has_received_messages:
                    return None
                self.finished = True
                return self._create_final_chunk()
        return None

    def _create_content_chunk(self, text: str) -> ModelResponseStream:
        """Create a ModelResponseStream chunk with content."""
        return self._create_delta_chunk(text)

    def _create_final_chunk(self) -> ModelResponseStream:
        """Create a final ModelResponseStream chunk with finish_reason."""
        chunk = ModelResponseStream(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=0,
            model=self.model,
            object="chat.completion.chunk",
        )

        chunk.choices = [
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(),
            )
        ]

        return chunk

    def __next__(self) -> ModelResponseStream:
        """Sync iteration - parse SSE events and yield ModelResponse chunks."""
        try:
            if self.line_iterator is None:
                raise StopIteration

            for line in self.line_iterator:
                result = self._parse_sse_line(line)
                if result is not None:
                    return result

            # Stream ended naturally - send final chunk if not already finished
            if not self.finished:
                self.finished = True
                return self._create_final_chunk()

            raise StopIteration

        except StopIteration:
            raise
        except httpx.StreamConsumed:
            raise StopIteration
        except httpx.StreamClosed:
            raise StopIteration
        except Exception as e:
            verbose_logger.error(f"Error in LangGraph SSE stream: {str(e)}")
            raise StopIteration

    async def __anext__(self) -> ModelResponseStream:
        """Async iteration - parse SSE events and yield ModelResponse chunks."""
        try:
            if self.async_line_iterator is None:
                raise StopAsyncIteration

            async for line in self.async_line_iterator:
                result = self._parse_sse_line(line)
                if result is not None:
                    return result

            # Stream ended naturally - send final chunk if not already finished
            if not self.finished:
                self.finished = True
                return self._create_final_chunk()

            raise StopAsyncIteration

        except StopAsyncIteration:
            raise
        except httpx.StreamConsumed:
            raise StopAsyncIteration
        except httpx.StreamClosed:
            raise StopAsyncIteration
        except Exception as e:
            verbose_logger.error(f"Error in LangGraph SSE stream: {str(e)}")
            raise StopAsyncIteration
