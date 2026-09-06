# What is this?
## Translates OpenAI call to Anthropic `/v1/messages` format
import asyncio
import json
import traceback
from collections import deque
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Final

from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.llms.anthropic.experimental_pass_through.messages.utils import (
    refusal_stop_details,
    responses_output_refusal_text,
)
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicUsage

from .transformation import LiteLLMAnthropicToResponsesAPIAdapter

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObject


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
        litellm_logging_obj: "LiteLLMLoggingObject | None" = None,
    ) -> None:
        self.responses_stream = responses_stream
        self.model = model
        self._message_id: str = f"msg_{uuid.uuid4()}"
        if litellm_logging_obj is not None:
            litellm_logging_obj.record_streamed_anthropic_message_id(self._message_id)
        self._current_block_index: int = -1
        # Map item_id -> content_block_index so we can stop the right block later
        self._item_id_to_block_index: dict[str, int] = {}
        # Track open function_call items by item_id so we can emit tool_use start
        self._pending_tool_ids: dict[str, str] = {}  # item_id -> call_id / name accumulator
        self._sent_message_start = False
        self._sent_message_stop = False
        self._chunk_queue: deque[dict[str, object]] = deque()
        self._refusal_text: str = ""
        self._sync_responses_iterator: Iterator[object] | None = None

    def _make_message_start(self) -> dict[str, object]:
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

    def _open_block(self, item_id: str | None, content_block: Mapping[str, object]) -> int:
        block_idx = self._next_block_index()
        if item_id:
            self._item_id_to_block_index[item_id] = block_idx
        self._chunk_queue.append(
            {
                "type": "content_block_start",
                "index": block_idx,
                "content_block": content_block,
            }
        )
        return block_idx

    def _process_event(self, event: object) -> None:
        """Convert one Responses API event into zero or more Anthropic chunks queued for emission."""
        event_type = getattr(event, "type", None)
        if event_type is None and isinstance(event, dict):
            event_type = event.get("type")

        if event_type is None:
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

            if item_type == "message":
                self._open_block(item_id, {"type": "text", "text": ""})
            elif item_type == "function_call":
                call_id: Final = (
                    getattr(item, "call_id", None) or (item.get("call_id") if isinstance(item, dict) else None) or ""
                )
                name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else None) or ""
                if item_id:
                    self._pending_tool_ids[item_id] = call_id
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

        if event_type == "response.refusal.delta":
            delta = getattr(event, "delta", "") or (event.get("delta", "") if isinstance(event, dict) else "")
            if not isinstance(delta, str) or not delta:
                return
            self._refusal_text = self._refusal_text + delta
            item_id = getattr(event, "item_id", None) or (event.get("item_id") if isinstance(event, dict) else None)
            block_idx = self._item_id_to_block_index.get(item_id, -1) if item_id else self._current_block_index
            if block_idx < 0:
                block_idx = self._open_block(item_id, {"type": "text", "text": ""})
            self._chunk_queue.append(
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "text_delta", "text": delta},
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
            block_idx = self._item_id_to_block_index.get(item_id, -1) if item_id else self._current_block_index
            if block_idx < 0:
                return
            self._chunk_queue.append(
                {
                    "type": "content_block_stop",
                    "index": block_idx,
                }
            )
            return

        # ---- response completed -> message_delta + message_stop ----
        if event_type in (
            "response.completed",
            "response.failed",
            "response.incomplete",
        ):
            response_obj: Final = getattr(event, "response", None) or (
                event.get("response") if isinstance(event, dict) else None
            )
            output: Final = (getattr(response_obj, "output", None) or ()) if response_obj is not None else ()
            refusal_text: Final = responses_output_refusal_text(output) or (self._refusal_text or None)
            status: Final = getattr(response_obj, "status", None) if response_obj is not None else None
            has_tool_call: Final = any(
                getattr(item, "type", None) == "function_call"
                or (isinstance(item, dict) and item.get("type") == "function_call")
                for item in output
            )
            stop_reason: Final = (
                "max_tokens"
                if status == "incomplete"
                else "refusal"
                if refusal_text is not None
                else "tool_use"
                if has_tool_call
                else "end_turn"
            )
            anthropic_usage: Final[AnthropicUsage] = (
                LiteLLMAnthropicToResponsesAPIAdapter.translate_responses_api_usage_to_anthropic_usage(
                    getattr(response_obj, "usage", None)
                )
                if response_obj is not None
                else AnthropicUsage(input_tokens=0, output_tokens=0)
            )

            message_delta_payload: Final = {  # mutable-ok: fresh message_delta payload built per chunk
                "stop_reason": stop_reason,
                "stop_sequence": None,
                **(
                    {  # mutable-ok: fresh message_delta stop_details entry built per chunk
                        "stop_details": refusal_stop_details(refusal_text)
                    }
                    if stop_reason == "refusal"
                    else {}  # mutable-ok: empty spread placeholder for non-refusal stop
                ),
            }

            self._chunk_queue.append(
                {
                    "type": "message_delta",
                    "delta": message_delta_payload,
                    "usage": dict(anthropic_usage),
                }
            )
            self._chunk_queue.append({"type": "message_stop"})
            self._sent_message_stop = True
            return

    def __aiter__(self) -> "AnthropicResponsesStreamWrapper":
        return self

    async def __anext__(self) -> dict[str, object]:
        # Return any queued chunks first
        if self._chunk_queue:
            return self._chunk_queue.popleft()

        # Emit message_start if not yet done (fallback if response.created wasn't fired)
        if not self._sent_message_start:
            self._sent_message_start = True
            self._chunk_queue.append(self._make_message_start())
            return self._chunk_queue.popleft()

        # Consume the upstream stream
        try:
            if hasattr(self.responses_stream, "__aiter__"):
                async for event in self.responses_stream:
                    self._process_event(event)
                    if self._chunk_queue:
                        return self._chunk_queue.popleft()
            else:
                if self._sync_responses_iterator is None:
                    self._sync_responses_iterator = iter(self.responses_stream)
                sync_iterator: Final = self._sync_responses_iterator
                missing: Final = object()
                while (event := await asyncio.to_thread(next, sync_iterator, missing)) is not missing:
                    self._process_event(event)
                    if self._chunk_queue:
                        return self._chunk_queue.popleft()
        except StopAsyncIteration:
            pass
        except Exception as e:
            verbose_logger.error("AnthropicResponsesStreamWrapper error: %s\n%s", e, traceback.format_exc())

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
