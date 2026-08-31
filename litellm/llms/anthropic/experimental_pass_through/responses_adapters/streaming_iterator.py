# What is this?
## Translates OpenAI call to Anthropic `/v1/messages` format
import json
from collections import deque
from collections.abc import AsyncIterator, Mapping
from typing import Any, Final

from litellm._uuid import uuid
from litellm.exceptions import APIError, MidStreamFallbackError

from .transformation import LiteLLMAnthropicToResponsesAPIAdapter

INCOMPLETE_STREAM_ERROR_MESSAGE: Final = (
    "Provider stream ended before emitting a message_stop event; "
    "the response is incomplete and any partial content (e.g. tool_use input JSON) may be truncated."
)


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
        self._item_id_to_block_index: dict[str, int] = {}
        self._output_item_types: dict[str, str] = {}
        self._function_call_argument_deltas: dict[str, list[str]] = {}
        self._finalized_function_call_item_ids: set[str] = set()
        self._closed_output_item_ids: set[str] = set()
        self._sent_message_start = False
        self._sent_message_stop = False
        self._sent_error = False
        self._open_block_indexes: set[int] = set()
        self._chunk_queue: deque = deque()

    def _make_message_start(self) -> dict[str, Any]:
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

    def _open_block(self, item_id: str | None, content_block: Mapping[str, Any]) -> int:
        block_idx = self._next_block_index()
        if item_id:
            self._item_id_to_block_index[item_id] = block_idx
        self._open_block_indexes.add(block_idx)
        self._chunk_queue.append(
            {
                "type": "content_block_start",
                "index": block_idx,
                "content_block": content_block,
            }
        )
        return block_idx

    @staticmethod
    def _event_value(event: object, name: str) -> object | None:
        if isinstance(event, Mapping):
            return event.get(name)
        return getattr(event, name, None)

    def _queue_error(self, message: str) -> None:
        if self._sent_message_stop or self._sent_error:
            return
        self._chunk_queue.append({"type": "error", "error": {"type": "api_error", "message": message}})
        self._sent_error = True

    def _queue_completion(self, response_obj: object, stop_reason: str) -> None:
        if self._sent_message_stop or self._sent_error:
            return
        anthropic_usage: Final = LiteLLMAnthropicToResponsesAPIAdapter.translate_responses_api_usage_to_anthropic_usage(
            self._event_value(response_obj, "usage")
        )
        self._chunk_queue.append(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": dict(anthropic_usage),
            }
        )
        self._chunk_queue.append({"type": "message_stop"})
        self._sent_message_stop = True

    def _has_replayable_incomplete_output(self, response_obj: object) -> bool:
        output: Final = self._event_value(response_obj, "output")
        output_items: Final = output if isinstance(output, list) else ()
        if not output_items or self._open_block_indexes:
            return False

        for output_item in output_items:
            item_id: Final = self._event_value(output_item, "id")
            item_type: Final = self._event_value(output_item, "type")
            if (
                not isinstance(item_id, str)
                or item_type not in {"message", "reasoning", "function_call"}
                or self._output_item_types.get(item_id) != item_type
                or item_id not in self._closed_output_item_ids
            ):
                return False
            if item_type == "function_call" and item_id not in self._finalized_function_call_item_ids:
                return False
        return True

    def _has_complete_terminal_state(self) -> bool:
        return not self._open_block_indexes

    def _has_complete_function_calls(self, output_items: object) -> bool:
        if not isinstance(output_items, list):
            return True
        return all(
            self._event_value(output_item, "type") != "function_call"
            or (
                isinstance(self._event_value(output_item, "id"), str)
                and self._event_value(output_item, "id") in self._finalized_function_call_item_ids
            )
            for output_item in output_items
        )

    def _response_incomplete_reason(self, response_obj: object) -> str | None:
        incomplete_details: Final = self._event_value(response_obj, "incomplete_details")
        return_value: Final = self._event_value(incomplete_details, "reason")
        return return_value if isinstance(return_value, str) else None

    def _process_incomplete_response(self, response_obj: object) -> None:
        incomplete_reason: Final = self._response_incomplete_reason(response_obj)
        if incomplete_reason == "max_output_tokens" and self._has_replayable_incomplete_output(response_obj):
            self._queue_completion(response_obj, "max_tokens")
            return
        self._queue_error("Provider returned an incomplete response that cannot be safely continued.")

    def _process_event(self, event: Any) -> None:
        """Convert one Responses API event into zero or more Anthropic chunks queued for emission."""
        event_type = getattr(event, "type", None)
        if event_type is None and isinstance(event, dict):
            event_type = event.get("type")

        if event_type is None or self._sent_message_stop or self._sent_error:
            return

        # ---- message_start ----
        if event_type == "response.created":
            if not self._sent_message_start:
                self._sent_message_start = True
                self._chunk_queue.append(self._make_message_start())
            return

        # ---- content_block_start for a new output message item ----
        if event_type == "response.output_item.added":
            item = getattr(event, "item", None) or (event.get("item") if isinstance(event, dict) else None)
            if item is None:
                return
            item_type: Final = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
            item_id = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)

            if isinstance(item_id, str) and isinstance(item_type, str):
                self._output_item_types[item_id] = item_type

            if item_type == "message":
                self._open_block(item_id, {"type": "text", "text": ""})
            elif item_type == "function_call":
                call_id: Final = (
                    getattr(item, "call_id", None) or (item.get("call_id") if isinstance(item, dict) else None) or ""
                )
                name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else None) or ""
                self._open_block(
                    item_id,
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": {},
                    },
                )
            return

        # ---- text delta ----
        if event_type == "response.output_text.delta":
            item_id = getattr(event, "item_id", None) or (event.get("item_id") if isinstance(event, dict) else None)
            delta = getattr(event, "delta", "") or (event.get("delta", "") if isinstance(event, dict) else "")
            block_idx = self._item_id_to_block_index.get(item_id, -1) if item_id else self._current_block_index
            if block_idx < 0:
                if isinstance(item_id, str):
                    self._output_item_types[item_id] = "message"
                block_idx = self._open_block(item_id, {"type": "text", "text": ""})
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
            block_idx = self._item_id_to_block_index.get(item_id, -1) if item_id else self._current_block_index
            if block_idx < 0:
                if not delta:
                    return
                block_idx = self._open_block(
                    item_id,
                    {"type": "thinking", "thinking": "", "signature": ""},  # mutable-ok: API message payload
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
            if not isinstance(item_id, str) or self._output_item_types.get(item_id) != "function_call":
                return
            if not isinstance(delta, str):
                return
            self._function_call_argument_deltas.setdefault(item_id, []).append(delta)
            block_idx = self._item_id_to_block_index.get(item_id, -1)
            if block_idx < 0:
                return
            self._chunk_queue.append(
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "input_json_delta", "partial_json": delta},
                }
            )
            return

        if event_type == "response.function_call_arguments.done":
            item_id = getattr(event, "item_id", None) or (event.get("item_id") if isinstance(event, dict) else None)
            arguments = getattr(event, "arguments", None) or (
                event.get("arguments") if isinstance(event, dict) else None
            )
            argument_deltas: Final = (
                self._function_call_argument_deltas.pop(item_id, []) if isinstance(item_id, str) else []
            )
            if (
                isinstance(item_id, str)
                and self._output_item_types.get(item_id) == "function_call"
                and isinstance(arguments, str)
                and "".join(argument_deltas) == arguments
            ):
                try:
                    parsed_arguments: Final = json.loads(arguments)
                except json.JSONDecodeError:
                    return
                if not isinstance(parsed_arguments, Mapping):
                    return
                self._finalized_function_call_item_ids.add(item_id)
            return

        # ---- output item done -> content_block_stop ----
        if event_type == "response.output_item.done":
            item = getattr(event, "item", None) or (event.get("item") if isinstance(event, dict) else None)
            item_id = (
                getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None) if item else None
            )
            if isinstance(item_id, str):
                self._closed_output_item_ids.add(item_id)
            block_idx = self._item_id_to_block_index.get(item_id, -1) if item_id else self._current_block_index
            if block_idx < 0 or block_idx not in self._open_block_indexes:
                return
            self._open_block_indexes.remove(block_idx)
            self._chunk_queue.append(
                {
                    "type": "content_block_stop",
                    "index": block_idx,
                }
            )
            return

        response_obj: Final = self._event_value(event, "response")
        if event_type == "response.completed":
            if response_obj is None:
                self._queue_error("Provider completed a response without a response body.")
                return
            if self._event_value(response_obj, "status") == "incomplete":
                self._process_incomplete_response(response_obj)
                return
            if not self._has_complete_terminal_state():
                self._queue_error("Provider completed a response with an unclosed content block.")
                return
            output: Final = self._event_value(response_obj, "output")
            output_items: Final = output if isinstance(output, list) else ()
            if not self._has_complete_function_calls(output_items):
                self._queue_error("Provider completed a response with an incomplete function call.")
                return
            stop_reason: Final = (
                "tool_use"
                if any(self._event_value(output_item, "type") == "function_call" for output_item in output_items)
                else "end_turn"
            )
            self._queue_completion(response_obj, stop_reason)
            return

        if event_type == "response.failed":
            message: Final = self._event_value(self._event_value(response_obj, "error"), "message")
            self._queue_error(message if isinstance(message, str) else "Provider failed to generate a response.")
            return

        if event_type == "response.incomplete":
            if response_obj is not None:
                self._process_incomplete_response(response_obj)
                return
            self._queue_error("Provider returned an incomplete response that cannot be safely continued.")
            return

    def __aiter__(self) -> "AnthropicResponsesStreamWrapper":
        return self

    async def __anext__(self) -> dict[str, Any]:
        # Return any queued chunks first
        if self._chunk_queue:
            return self._chunk_queue.popleft()

        # Emit message_start if not yet done (fallback if response.created wasn't fired)
        if not self._sent_message_start:
            self._sent_message_start = True
            self._chunk_queue.append(self._make_message_start())
            return self._chunk_queue.popleft()

        try:
            async for event in self.responses_stream:
                self._process_event(event)
                if self._chunk_queue:
                    return self._chunk_queue.popleft()
        except (APIError, MidStreamFallbackError):
            raise
        except Exception:
            self._queue_error("Provider stream failed before a terminal response was emitted.")

        if self._chunk_queue:
            return self._chunk_queue.popleft()

        if not self._sent_message_stop and not self._sent_error:
            self._queue_error(INCOMPLETE_STREAM_ERROR_MESSAGE)
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
