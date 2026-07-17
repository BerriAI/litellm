# What is this?
## Translates OpenAI call to Anthropic `/v1/messages` format
import json
import traceback
from collections import deque
from typing import Any, AsyncIterator, Dict

from litellm import verbose_logger
from litellm._uuid import uuid


class AnthropicResponsesStreamWrapper:
    """
    Wraps a Responses API streaming iterator and re-emits events in Anthropic SSE format.

    Responses API event flow (relevant subset):
      response.created                   -> message_start
      response.output_item.added         -> content_block_start (if message/function_call)
      response.output_text.delta         -> content_block_delta (text_delta)
      response.reasoning_summary_text.delta -> content_block_delta (thinking_delta)
      response.function_call_arguments.delta -> content_block_delta (input_json_delta)
      response.output_item.done          -> content_block_stop
      response.completed                 -> message_delta + message_stop
    """

    def __init__(
        self,
        responses_stream: Any,
        model: str,
    ) -> None:
        self.responses_stream = responses_stream
        self.model = model
        self._message_id: str = f"msg_{uuid.uuid4()}"
        self._current_block_index: int = -1
        # Map item_id -> content_block_index so we can stop the right block later
        self._item_id_to_block_index: Dict[str, int] = {}
        # Track open function_call items by item_id so we can emit tool_use start
        self._pending_tool_ids: Dict[str, str] = {}  # item_id -> call_id / name accumulator
        self._sent_message_start = False
        self._sent_message_stop = False
        self._chunk_queue: deque = deque()

    def _make_message_start(self) -> Dict[str, Any]:
        return {
            "type": "message_start",
            "message": {
                "id": self._message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        }

    def _next_block_index(self) -> int:
        self._current_block_index += 1
        return self._current_block_index

    def _make_error_chunk(self, message: str) -> dict[str, Any]:
        return {
            "type": "error",
            "error": {"type": "api_error", "message": message},
        }

    def _process_event(self, event: Any) -> None:
        """Convert one Responses API event into zero or more Anthropic chunks queued for emission."""
        event_type = getattr(event, "type", None)
        if event_type is None and isinstance(event, dict):
            event_type = event.get("type")

        if event_type is None:
            return

        # ---- message_start ----
        if event_type == "response.created":
            self._sent_message_start = True
            self._chunk_queue.append(self._make_message_start())
            return

        # ---- content_block_start for a new output message item ----
        if event_type == "response.output_item.added":
            item = getattr(event, "item", None) or (event.get("item") if isinstance(event, dict) else None)
            if item is None:
                return
            item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
            item_id = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)

            if item_type == "message":
                block_idx = self._next_block_index()
                if item_id:
                    self._item_id_to_block_index[item_id] = block_idx
                self._chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
            elif item_type == "function_call":
                call_id = (
                    getattr(item, "call_id", None) or (item.get("call_id") if isinstance(item, dict) else None) or ""
                )
                name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else None) or ""
                block_idx = self._next_block_index()
                if item_id:
                    self._item_id_to_block_index[item_id] = block_idx
                    self._pending_tool_ids[item_id] = call_id
                self._chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": call_id,
                            "name": name,
                            "input": {},
                        },
                    }
                )
            elif item_type == "reasoning":
                block_idx = self._next_block_index()
                if item_id:
                    self._item_id_to_block_index[item_id] = block_idx
                self._chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {"type": "thinking", "thinking": ""},
                    }
                )
            return

        # ---- text delta ----
        if event_type == "response.output_text.delta":
            item_id = getattr(event, "item_id", None) or (event.get("item_id") if isinstance(event, dict) else None)
            delta = getattr(event, "delta", "") or (event.get("delta", "") if isinstance(event, dict) else "")
            block_idx = self._item_id_to_block_index.get(item_id, -1) if item_id else self._current_block_index
            if block_idx < 0:
                # Some providers (e.g. LMStudio) skip response.output_item.added,
                # so no text block is open yet; synthesize content_block_start
                # instead of emitting a delta with index -1
                block_idx = self._next_block_index()
                if item_id:
                    self._item_id_to_block_index[item_id] = block_idx
                self._chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
            self._chunk_queue.append(
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "text_delta", "text": delta},
                }
            )
            return

        # ---- reasoning summary text delta ----
        if event_type == "response.reasoning_summary_text.delta":
            item_id = getattr(event, "item_id", None) or (event.get("item_id") if isinstance(event, dict) else None)
            delta = getattr(event, "delta", "") or (event.get("delta", "") if isinstance(event, dict) else "")
            block_idx = (
                self._item_id_to_block_index.get(item_id, self._current_block_index)
                if item_id
                else self._current_block_index
            )
            self._chunk_queue.append(
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "thinking_delta", "thinking": delta},
                }
            )
            return

        # ---- function call arguments delta ----
        if event_type == "response.function_call_arguments.delta":
            item_id = getattr(event, "item_id", None) or (event.get("item_id") if isinstance(event, dict) else None)
            delta = getattr(event, "delta", "") or (event.get("delta", "") if isinstance(event, dict) else "")
            block_idx = (
                self._item_id_to_block_index.get(item_id, self._current_block_index)
                if item_id
                else self._current_block_index
            )
            self._chunk_queue.append(
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "input_json_delta", "partial_json": delta},
                }
            )
            return

        # ---- output item done -> content_block_stop ----
        if event_type == "response.output_item.done":
            item = getattr(event, "item", None) or (event.get("item") if isinstance(event, dict) else None)
            item_id = (
                getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None) if item else None
            )
            block_idx = (
                self._item_id_to_block_index.get(item_id, self._current_block_index)
                if item_id
                else self._current_block_index
            )
            self._chunk_queue.append(
                {
                    "type": "content_block_stop",
                    "index": block_idx,
                }
            )
            return

        # ---- response failed -> error ----
        if event_type == "response.failed":
            response_obj = getattr(event, "response", None) or (
                event.get("response") if isinstance(event, dict) else None
            )
            error_obj = None
            if response_obj is not None:
                error_obj = getattr(response_obj, "error", None) or (
                    response_obj.get("error") if isinstance(response_obj, dict) else None
                )
            message = "The upstream provider reported the response as failed."
            if error_obj is not None:
                error_message = getattr(error_obj, "message", None) or (
                    error_obj.get("message") if isinstance(error_obj, dict) else None
                )
                error_code = getattr(error_obj, "code", None) or (
                    error_obj.get("code") if isinstance(error_obj, dict) else None
                )
                if error_message:
                    message = f"{error_code}: {error_message}" if error_code else str(error_message)
            self._chunk_queue.append(self._make_error_chunk(message))
            self._sent_message_stop = True
            return

        # ---- error event -> error ----
        if event_type == "error":
            nested_error = getattr(event, "error", None) or (event.get("error") if isinstance(event, dict) else None)
            message = getattr(event, "message", None) or (event.get("message") if isinstance(event, dict) else None)
            if message is None and nested_error is not None:
                message = getattr(nested_error, "message", None) or (
                    nested_error.get("message") if isinstance(nested_error, dict) else None
                )
            if message is None:
                message = "The upstream provider returned an error while streaming."
            self._chunk_queue.append(self._make_error_chunk(str(message)))
            self._sent_message_stop = True
            return

        # ---- response completed -> message_delta + message_stop ----
        if event_type in (
            "response.completed",
            "response.incomplete",
        ):
            response_obj = getattr(event, "response", None) or (
                event.get("response") if isinstance(event, dict) else None
            )
            stop_reason = "end_turn"
            input_tokens = 0
            output_tokens = 0
            cache_creation_tokens = 0
            cache_read_tokens = 0

            if response_obj is not None:
                status = getattr(response_obj, "status", None)
                if status == "incomplete":
                    stop_reason = "max_tokens"
                usage = getattr(response_obj, "usage", None)
                if usage is not None:
                    input_tokens = getattr(usage, "input_tokens", 0) or 0
                    output_tokens = getattr(usage, "output_tokens", 0) or 0
                    cache_creation_tokens = getattr(usage, "input_tokens_details", None)  # type: ignore[assignment]
                    cache_read_tokens = getattr(usage, "output_tokens_details", None)  # type: ignore[assignment]
                    # Prefer direct cache fields if present
                    cache_creation_tokens = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
                    cache_read_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)

            # Check if tool_use was in the output to override stop_reason
            if response_obj is not None:
                output = getattr(response_obj, "output", []) or []
                for out_item in output:
                    out_type = getattr(out_item, "type", None) or (
                        out_item.get("type") if isinstance(out_item, dict) else None
                    )
                    if out_type == "function_call":
                        stop_reason = "tool_use"
                        break

            usage_delta: Dict[str, Any] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            if cache_creation_tokens:
                usage_delta["cache_creation_input_tokens"] = cache_creation_tokens
            if cache_read_tokens:
                usage_delta["cache_read_input_tokens"] = cache_read_tokens

            self._chunk_queue.append(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": usage_delta,
                }
            )
            self._chunk_queue.append({"type": "message_stop"})
            self._sent_message_stop = True
            return

    def __aiter__(self) -> "AnthropicResponsesStreamWrapper":
        return self

    async def __anext__(self) -> Dict[str, Any]:
        # Return any queued chunks first
        if self._chunk_queue:
            return self._chunk_queue.popleft()

        # Don't consume the upstream again after a terminal chunk
        # (message_stop / error) has been emitted
        if self._sent_message_stop:
            raise StopAsyncIteration

        # Emit message_start if not yet done (fallback if response.created wasn't fired)
        if not self._sent_message_start:
            self._sent_message_start = True
            self._chunk_queue.append(self._make_message_start())
            return self._chunk_queue.popleft()

        # Consume the upstream stream
        try:
            async for event in self.responses_stream:
                self._process_event(event)
                if self._chunk_queue:
                    return self._chunk_queue.popleft()
        except StopAsyncIteration:
            pass
        except Exception as e:
            verbose_logger.error(f"AnthropicResponsesStreamWrapper error: {e}\n{traceback.format_exc()}")
            self._chunk_queue.append(self._make_error_chunk("Upstream provider error while streaming."))
            self._sent_message_stop = True

        # Drain any remaining queued chunks
        if self._chunk_queue:
            return self._chunk_queue.popleft()

        raise StopAsyncIteration

    async def async_anthropic_sse_wrapper(self) -> AsyncIterator[bytes]:
        """Yield SSE-encoded bytes for each Anthropic event chunk."""
        async for chunk in self:
            if isinstance(chunk, dict):
                event_type: str = str(chunk.get("type", "message"))
                payload = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                yield payload.encode()
            else:
                yield chunk
