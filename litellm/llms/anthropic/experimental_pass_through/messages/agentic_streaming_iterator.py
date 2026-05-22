"""
Agentic Streaming Iterator for Anthropic Messages

Wraps the raw SSE byte stream from the Anthropic pass-through endpoint,
yields every chunk to the caller (preserving real streaming), collects
all bytes, and on stream exhaustion rebuilds the full Anthropic response
to run through agentic completion hooks. If an agentic hook fires, the
follow-up response is chained as Phase 2 of the same iterator.
"""

import json
from typing import Any, AsyncIterator, Dict, List, Optional, cast

from litellm._logging import verbose_logger


# ---------------------------------------------------------------------------
# SSE parsing helpers (module-level to keep the class lean)
# ---------------------------------------------------------------------------


def _parse_sse_events(raw: bytes) -> List[tuple]:
    """Return a list of (event_type, parsed_data_dict) from raw SSE bytes."""
    text = raw.decode("utf-8", errors="replace")
    lines = text.split("\n")
    events: List[tuple] = []
    current_event_type: Optional[str] = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("event:"):
            current_event_type = stripped[len("event:") :].strip()
            continue
        if not stripped.startswith("data:"):
            continue
        data_str = stripped[len("data:") :].strip()
        try:
            data = json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            continue
        event_type = current_event_type or data.get("type", "")
        current_event_type = None
        events.append((event_type, data))
    return events


def _handle_message_start(data: Dict, response: Dict) -> None:
    msg = data.get("message", {})
    response["id"] = msg.get("id", response["id"])
    response["model"] = msg.get("model", response["model"])
    response["role"] = msg.get("role", response["role"])
    usage = msg.get("usage", {})
    if usage:
        response["usage"]["input_tokens"] = usage.get("input_tokens", 0)
        for key in ("cache_creation_input_tokens", "cache_read_input_tokens"):
            if key in usage:
                response["usage"][key] = usage[key]


def _handle_content_block_start(data: Dict, content_blocks: Dict[int, Dict]) -> None:
    idx = data.get("index", len(content_blocks))
    block = data.get("content_block", {})
    block_type = block.get("type", "text")

    _BLOCK_TEMPLATES: Dict[str, Dict] = {
        "text": {"type": "text", "text": ""},
        "thinking": {"type": "thinking", "thinking": "", "signature": ""},
        "redacted_thinking": {
            "type": "redacted_thinking",
            "data": block.get("data", ""),
        },
    }
    if block_type == "tool_use":
        content_blocks[idx] = {
            "type": "tool_use",
            "id": block.get("id", ""),
            "name": block.get("name", ""),
            "input": {},
            "_partial_json": "",
        }
    elif block_type in _BLOCK_TEMPLATES:
        content_blocks[idx] = dict(_BLOCK_TEMPLATES[block_type])
    else:
        content_blocks[idx] = dict(block)


def _handle_content_block_delta(data: Dict, content_blocks: Dict[int, Dict]) -> None:
    idx = data.get("index", 0)
    delta = data.get("delta", {})
    delta_type = delta.get("type", "")
    block = content_blocks.get(idx)
    if block is None:
        return

    if delta_type == "text_delta":
        block["text"] = block.get("text", "") + delta.get("text", "")
    elif delta_type == "input_json_delta":
        block["_partial_json"] = block.get("_partial_json", "") + delta.get(
            "partial_json", ""
        )
    elif delta_type == "thinking_delta":
        block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
    elif delta_type == "signature_delta":
        block["signature"] = delta.get("signature", block.get("signature", ""))


def _handle_content_block_stop(data: Dict, content_blocks: Dict[int, Dict]) -> None:
    idx = data.get("index", 0)
    block = content_blocks.get(idx)
    if block and block.get("type") == "tool_use":
        partial = block.pop("_partial_json", "")
        if partial:
            try:
                block["input"] = json.loads(partial)
            except (json.JSONDecodeError, ValueError):
                block["input"] = {"_raw": partial}


def _handle_message_delta(data: Dict, response: Dict) -> None:
    delta = data.get("delta", {})
    if "stop_reason" in delta:
        response["stop_reason"] = delta["stop_reason"]
    if "stop_sequence" in delta:
        response["stop_sequence"] = delta["stop_sequence"]
    usage = data.get("usage", {})
    if usage.get("output_tokens") is not None:
        response["usage"]["output_tokens"] = usage["output_tokens"]
    for key in (
        "input_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        if key in usage:
            response["usage"][key] = usage[key]


class AgenticAnthropicStreamingIterator:
    """
    Two-phase async iterator that enables agentic hooks on streaming
    Anthropic Messages pass-through responses.

    Phase 1: Yield raw SSE bytes from the upstream response while
             accumulating them. When the inner iterator is exhausted,
             rebuild the full Anthropic response dict and call agentic hooks.

    Phase 2: If an agentic hook fires and returns a follow-up response
             (streaming or non-streaming), yield those bytes to the caller.
    """

    def __init__(
        self,
        completion_stream: AsyncIterator,
        http_handler: Any,
        model: str,
        messages: List[Dict],
        anthropic_messages_provider_config: Any,
        anthropic_messages_optional_request_params: Dict,
        logging_obj: Any,
        custom_llm_provider: str,
        kwargs: Dict,
    ):
        self._inner = completion_stream.__aiter__()
        self._http_handler = http_handler
        self._model = model
        self._messages = messages
        self._anthropic_messages_provider_config = anthropic_messages_provider_config
        self._anthropic_messages_optional_request_params = (
            anthropic_messages_optional_request_params
        )
        self._logging_obj = logging_obj
        self._custom_llm_provider = custom_llm_provider
        self._kwargs = kwargs

        self._collected_bytes: List[bytes] = []
        self._stream_exhausted = False
        self._hook_processing_done = False
        self._follow_up_iterator: Optional[AsyncIterator] = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        # Phase 1: yield from upstream, collect bytes
        if not self._stream_exhausted:
            try:
                chunk = await self._inner.__anext__()
                self._collected_bytes.append(chunk)
                return chunk
            except StopAsyncIteration:
                self._stream_exhausted = True
                await self._process_agentic_hooks()
                # Fall through to Phase 2

        # Phase 2: yield from follow-up stream if one was created
        if self._follow_up_iterator is not None:
            chunk = await self._follow_up_iterator.__anext__()
            return chunk

        raise StopAsyncIteration

    async def _process_agentic_hooks(self) -> None:
        """Rebuild the Anthropic response from collected SSE bytes and call hooks."""
        if self._hook_processing_done:
            return
        self._hook_processing_done = True

        if not self._collected_bytes:
            return

        try:
            rebuilt = self._rebuild_anthropic_response_from_sse(self._collected_bytes)
            if rebuilt is None:
                verbose_logger.debug(
                    "AgenticStreamingIterator: Could not rebuild response from SSE bytes"
                )
                return

            [
                (
                    f"{b.get('type')}({b.get('name', '')})"
                    if b.get("type") == "tool_use"
                    else b.get("type")
                )
                for b in rebuilt.get("content", [])
            ]

            result = await self._http_handler._call_agentic_completion_hooks(
                response=rebuilt,
                model=self._model,
                messages=self._messages,
                anthropic_messages_provider_config=self._anthropic_messages_provider_config,
                anthropic_messages_optional_request_params=self._anthropic_messages_optional_request_params,
                logging_obj=self._logging_obj,
                stream=True,
                custom_llm_provider=self._custom_llm_provider,
                kwargs=self._kwargs,
            )

            if result is None:
                return

            if hasattr(result, "__aiter__"):
                self._follow_up_iterator = result.__aiter__()
            elif isinstance(result, dict):
                from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
                    FakeAnthropicMessagesStreamIterator,
                )
                from litellm.types.llms.anthropic_messages.anthropic_response import (
                    AnthropicMessagesResponse,
                )

                fake = FakeAnthropicMessagesStreamIterator(
                    response=cast(AnthropicMessagesResponse, result)
                )
                self._follow_up_iterator = fake.__aiter__()
            else:
                verbose_logger.warning(
                    "AgenticStreamingIterator: Unexpected result type from hooks: %s",
                    type(result).__name__,
                )
        except Exception as e:
            _call_id = getattr(self._logging_obj, "litellm_call_id", "unknown")
            verbose_logger.exception(
                "AgenticStreamingIterator: Error in agentic hook processing "
                "[call_id=%s model=%s]: %s",
                _call_id,
                self._model,
                str(e),
            )

    @staticmethod
    def _rebuild_anthropic_response_from_sse(
        raw_bytes: List[bytes],
    ) -> Optional[Dict[str, Any]]:
        """
        Parse collected SSE bytes into an Anthropic Messages response dict.

        Processes SSE events in order:
        - message_start   -> envelope (id, model, role, usage)
        - content_block_start -> new content block
        - content_block_delta -> accumulate text/json/thinking deltas
        - content_block_stop  -> finalize block
        - message_delta   -> stop_reason, output usage
        - message_stop    -> end
        """
        events = _parse_sse_events(b"".join(raw_bytes))

        response: Dict[str, Any] = {
            "id": "",
            "type": "message",
            "role": "assistant",
            "model": "",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        content_blocks: Dict[int, Dict[str, Any]] = {}
        saw_message_start = False

        for event_type, data in events:
            if event_type == "message_start":
                saw_message_start = True
                _handle_message_start(data, response)
            elif event_type == "content_block_start":
                _handle_content_block_start(data, content_blocks)
            elif event_type == "content_block_delta":
                _handle_content_block_delta(data, content_blocks)
            elif event_type == "content_block_stop":
                _handle_content_block_stop(data, content_blocks)
            elif event_type == "message_delta":
                _handle_message_delta(data, response)

        if not saw_message_start:
            return None

        for idx in sorted(content_blocks.keys()):
            block = content_blocks[idx]
            block.pop("_partial_json", None)
            response["content"].append(block)

        return response
