"""
Agentic Streaming Iterator for Anthropic Messages

Wraps the raw SSE byte stream from the Anthropic pass-through endpoint,
yields every chunk to the caller (preserving real streaming), collects
all bytes, and on stream exhaustion rebuilds the full Anthropic response
to run through agentic completion hooks. If an agentic hook fires, the
follow-up response is chained as Phase 2 of the same iterator.

In hold-back mode (``hold_back=True``) chunks are buffered instead of yielded
live, keepalive pings run whenever no other byte is ready, and then either the
follow-up replaces the message or the buffer replays, except that a tool_use for
a server-fulfilled tool fails the turn rather than reaching a client that cannot
execute it.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any, Final, cast

from litellm._logging import verbose_logger
from litellm.constants import STREAM_SSE_KEEPALIVE_PING_BYTES

HOLD_BACK_PING_INTERVAL_SECONDS: Final = 15.0
SERVER_FULFILLED_TOOL_LEAK_ERROR_SSE_BYTES: Final = (
    b"event: error\n"
    b'data: {"type": "error", "error": {"type": "api_error", "message": '
    b'"Server-side tool retrieval failed, so this turn could not be completed. Please retry."}}\n\n'
)


def is_server_fulfilled_tool_leak_error(chunk: object) -> bool:
    return chunk == SERVER_FULFILLED_TOOL_LEAK_ERROR_SSE_BYTES


async def _anext_or_none(iterator: AsyncIterator) -> bytes | None:
    try:
        return await iterator.__anext__()
    except StopAsyncIteration:
        return None


# ---------------------------------------------------------------------------
# SSE parsing helpers (module-level to keep the class lean)
# ---------------------------------------------------------------------------


def _parse_sse_events(raw: bytes) -> list[tuple]:
    """Return a list of (event_type, parsed_data_dict) from raw SSE bytes."""
    text: Final = raw.decode("utf-8", errors="replace")
    lines: Final = text.split("\n")
    events: Final[list[tuple]] = []
    current_event_type: str | None = None

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


def _handle_message_start(data: dict, response: dict) -> None:
    msg: Final = data.get("message", {})
    response["id"] = msg.get("id", response["id"])
    response["model"] = msg.get("model", response["model"])
    response["role"] = msg.get("role", response["role"])
    usage: Final = msg.get("usage", {})
    if usage:
        response["usage"]["input_tokens"] = usage.get("input_tokens", 0)
        for key in ("cache_creation_input_tokens", "cache_read_input_tokens"):
            if key in usage:
                response["usage"][key] = usage[key]


def _handle_content_block_start(data: dict, content_blocks: dict[int, dict]) -> None:
    idx: Final = data.get("index", len(content_blocks))
    block: Final = data.get("content_block", {})
    block_type: Final = block.get("type", "text")

    _BLOCK_TEMPLATES: Final[dict[str, dict]] = {
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


def _handle_content_block_delta(data: dict, content_blocks: dict[int, dict]) -> None:
    idx: Final = data.get("index", 0)
    delta: Final = data.get("delta", {})
    delta_type: Final = delta.get("type", "")
    block: Final = content_blocks.get(idx)
    if block is None:
        return

    if delta_type == "text_delta":
        block["text"] = block.get("text", "") + delta.get("text", "")
    elif delta_type == "input_json_delta":
        block["_partial_json"] = block.get("_partial_json", "") + delta.get("partial_json", "")
    elif delta_type == "thinking_delta":
        block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
    elif delta_type == "signature_delta":
        block["signature"] = delta.get("signature", block.get("signature", ""))


def _handle_content_block_stop(data: dict, content_blocks: dict[int, dict]) -> None:
    idx: Final = data.get("index", 0)
    block: Final = content_blocks.get(idx)
    if block and block.get("type") == "tool_use":
        partial: Final = block.pop("_partial_json", "")
        if partial:
            try:
                block["input"] = json.loads(partial)
            except (json.JSONDecodeError, ValueError):
                block["input"] = {"_raw": partial}


def _handle_message_delta(data: dict, response: dict) -> None:
    delta: Final = data.get("delta", {})
    if "stop_reason" in delta:
        response["stop_reason"] = delta["stop_reason"]
    if "stop_sequence" in delta:
        response["stop_sequence"] = delta["stop_sequence"]
    usage: Final = data.get("usage", {})
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
        messages: list[dict],
        anthropic_messages_provider_config: Any,
        anthropic_messages_optional_request_params: dict,
        logging_obj: Any,
        custom_llm_provider: str,
        kwargs: dict,
        hold_back: bool = False,
        server_fulfilled_tool_names: frozenset[str] = frozenset(),
        ping_interval_seconds: float = HOLD_BACK_PING_INTERVAL_SECONDS,
    ):
        self._inner = completion_stream.__aiter__()
        self._http_handler = http_handler
        self._model = model
        self._messages = messages
        self._anthropic_messages_provider_config = anthropic_messages_provider_config
        self._anthropic_messages_optional_request_params = anthropic_messages_optional_request_params
        self._logging_obj = logging_obj
        self._custom_llm_provider = custom_llm_provider
        self._kwargs = kwargs
        self._hold_back = hold_back
        self._server_fulfilled_tool_names = server_fulfilled_tool_names
        self._ping_interval_seconds = ping_interval_seconds

        self._collected_bytes: list[bytes] = []
        self._stream_exhausted = False
        self._hook_processing_done = False
        self._follow_up_iterator: AsyncIterator | None = None
        self._drain_task: asyncio.Task | None = None
        self._hook_task: asyncio.Task | None = None
        self._follow_up_chunk_task: asyncio.Task | None = None
        self._replay_index = 0
        self._error_emitted = False

    @property
    def has_buffered_provider_output(self) -> bool:
        """Whether provider output was received but withheld from the client behind keepalive pings."""
        return self._hold_back and bool(self._collected_bytes)

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if self._hold_back:
            return await self._anext_held_back()

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

    async def _drain_upstream(self) -> None:
        try:
            while True:
                self._collected_bytes.append(await self._inner.__anext__())
        except StopAsyncIteration:
            return

    async def _completed_within_ping_interval(self, task: asyncio.Task) -> bool:
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=self._ping_interval_seconds)
        except asyncio.TimeoutError:
            return False
        return True

    async def _anext_held_back(self) -> bytes:
        if self._drain_task is None:
            self._drain_task = asyncio.create_task(self._drain_upstream())
            return STREAM_SSE_KEEPALIVE_PING_BYTES

        if not self._stream_exhausted:
            if not await self._completed_within_ping_interval(self._drain_task):
                return STREAM_SSE_KEEPALIVE_PING_BYTES
            self._stream_exhausted = True

        if self._hook_task is None:
            self._hook_task = asyncio.create_task(self._process_agentic_hooks())
        if not await self._completed_within_ping_interval(self._hook_task):
            return STREAM_SSE_KEEPALIVE_PING_BYTES

        if self._follow_up_iterator is not None:
            return await self._next_follow_up_chunk(self._follow_up_iterator)

        if self._buffer_holds_server_fulfilled_tool_use():
            if self._error_emitted:
                raise StopAsyncIteration
            self._error_emitted = True
            verbose_logger.error(
                "AgenticStreamingIterator: hooks did not replace a message containing a server-fulfilled "
                "tool_use [model=%s]; emitting an SSE error instead of leaking the tool call to the client",
                self._model,
            )
            return SERVER_FULFILLED_TOOL_LEAK_ERROR_SSE_BYTES

        if self._replay_index < len(self._collected_bytes):
            chunk: Final = self._collected_bytes[self._replay_index]
            self._replay_index += 1
            return chunk

        raise StopAsyncIteration

    async def _next_follow_up_chunk(self, follow_up_iterator: AsyncIterator) -> bytes:
        if self._follow_up_chunk_task is None:
            self._follow_up_chunk_task = asyncio.create_task(_anext_or_none(follow_up_iterator))
        if not await self._completed_within_ping_interval(self._follow_up_chunk_task):
            return STREAM_SSE_KEEPALIVE_PING_BYTES
        chunk: Final = self._follow_up_chunk_task.result()
        self._follow_up_chunk_task = None
        if chunk is None:
            raise StopAsyncIteration
        return chunk

    def _buffer_holds_server_fulfilled_tool_use(self) -> bool:
        if not self._server_fulfilled_tool_names:
            return False
        started_blocks: Final = (
            data.get("content_block")
            for event_type, data in _parse_sse_events(b"".join(self._collected_bytes))
            if event_type == "content_block_start"
        )
        return any(
            isinstance(block, dict)
            and block.get("type") == "tool_use"
            and block.get("name") in self._server_fulfilled_tool_names
            for block in started_blocks
        )

    @staticmethod
    async def _settle_task(task: asyncio.Task | None) -> None:
        if task is None:
            return
        if task.done():
            if not task.cancelled():
                task.exception()
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def aclose(self) -> None:
        from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
            aclose_if_supported,
        )

        await self._settle_task(self._drain_task)
        await self._settle_task(self._hook_task)
        await self._settle_task(self._follow_up_chunk_task)
        await aclose_if_supported(self._inner)
        await aclose_if_supported(self._follow_up_iterator)

    async def _process_agentic_hooks(self) -> None:
        """Rebuild the Anthropic response from collected SSE bytes and call hooks."""
        if self._hook_processing_done:
            return
        self._hook_processing_done = True

        if not self._collected_bytes:
            return

        try:
            rebuilt: Final = self._rebuild_anthropic_response_from_sse(self._collected_bytes)
            if rebuilt is None:
                verbose_logger.debug("AgenticStreamingIterator: Could not rebuild response from SSE bytes")
                return

            result: Final = await self._http_handler._call_agentic_completion_hooks(
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

                fake: Final = FakeAnthropicMessagesStreamIterator(response=cast(AnthropicMessagesResponse, result))
                self._follow_up_iterator = fake.__aiter__()
            else:
                verbose_logger.warning(
                    "AgenticStreamingIterator: Unexpected result type from hooks: %s",
                    type(result).__name__,
                )
        except Exception as e:
            _call_id: Final = getattr(self._logging_obj, "litellm_call_id", "unknown")
            verbose_logger.exception(
                "AgenticStreamingIterator: Error in agentic hook processing [call_id=%s model=%s]: %s",
                _call_id,
                self._model,
                str(e),
            )

    @staticmethod
    def _rebuild_anthropic_response_from_sse(
        raw_bytes: list[bytes],
    ) -> dict[str, Any] | None:
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
        events: Final = _parse_sse_events(b"".join(raw_bytes))

        response: Final[dict[str, Any]] = {
            "id": "",
            "type": "message",
            "role": "assistant",
            "model": "",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        content_blocks: Final[dict[int, dict[str, Any]]] = {}
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
