"""
Agentic Streaming Iterator for Anthropic Messages

Wraps the raw SSE byte stream from the Anthropic pass-through endpoint,
yields every chunk to the caller (preserving real streaming), collects
all bytes, and on stream exhaustion rebuilds the full Anthropic response
to run through agentic completion hooks. If an agentic hook fires, the
follow-up response is chained as Phase 2 of the same iterator.

In hold-back mode (``hold_back=True``) frames stream live until the first tool_use
block or the frames that end the message, which are held behind keepalive pings while
the hooks run. A follow-up then continues the same message, a server-fulfilled tool_use
the hooks did not replace fails the turn with an error frame, and otherwise the held
frames replay.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Mapping, Sequence
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from pydantic import TypeAdapter

from litellm._logging import verbose_logger
from litellm.constants import STREAM_SSE_KEEPALIVE_PING_BYTES
from litellm.litellm_core_utils.core_helpers import has_attributes, loads_json_object
from litellm.llms.anthropic.messages.native.streaming_iterator import (
    SSE_FRAME_TERMINATORS,
    sse_frame_event,
    sse_frames,
)
from litellm.llms.anthropic.messages.utils import anthropic_sse_frame, object_mapping
from litellm.llms.custom_httpx.stream_close import aclose_if_supported
from litellm.types.integrations.custom_logger import AgenticLoopLimitError, AgenticLoopSafetyError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.base_llm.anthropic_messages.transformation import (
        BaseAnthropicMessagesConfig,
    )

HOLD_BACK_PING_INTERVAL_SECONDS: Final = 15.0
SERVER_FULFILLED_TOOL_LEAK_ERROR_SSE_BYTES: Final = (
    b"event: error\n"
    b'data: {"type": "error", "error": {"type": "api_error", "message": '
    b'"Server-side tool retrieval failed, so this turn could not be completed. Please retry."}}\n\n'
)
AGENTIC_LOOP_LIMIT_ERROR_SSE_BYTES: Final = (
    b"event: error\n"
    b'data: {"type": "error", "error": {"type": "api_error", "message": '
    b'"The agentic loop reached its limit before this turn finished. '
    b'Raise max_agentic_loops to allow more loops."}}\n\n'
)
_AGENTIC_HOOK_RESULT: Final = TypeAdapter(object)
_CONTENT_BLOCKS: Final = TypeAdapter(list[object])
_HOLD_START_EVENTS: Final = frozenset(("message_delta", "message_stop"))
_CONTENT_BLOCK_EVENTS: Final = frozenset(("content_block_start", "content_block_delta", "content_block_stop"))


class _KeepAlive(Enum):
    PING_DUE = auto()


def is_server_fulfilled_tool_leak_error(chunk: object) -> bool:
    return chunk in (SERVER_FULFILLED_TOOL_LEAK_ERROR_SSE_BYTES, AGENTIC_LOOP_LIMIT_ERROR_SSE_BYTES)


async def _anext_or_none(iterator: AsyncIterator[bytes]) -> bytes | None:
    try:
        return await iterator.__anext__()
    except StopAsyncIteration:
        return None


async def _settles_by(task: asyncio.Task[object], deadline: float) -> bool:
    done, _ = await asyncio.wait((task,), timeout=max(deadline - asyncio.get_running_loop().time(), 0))
    return bool(done)


def _frame_event(frame: object) -> Mapping[str, object] | None:
    return sse_frame_event(frame) if isinstance(frame, bytes) else None


def _is_content_block_start(frame: bytes) -> bool:
    event: Final = _frame_event(frame)
    return event is not None and event.get("type") == "content_block_start"


def _is_whole_frame(frame: object) -> bool:
    return isinstance(frame, bytes) and frame.endswith(SSE_FRAME_TERMINATORS)


def _starts_hold(frame: bytes) -> bool:
    if not _is_whole_frame(frame):
        return True
    event: Final = _frame_event(frame)
    if event is None:
        return False
    if event.get("type") in _HOLD_START_EVENTS:
        return True
    block: Final = object_mapping(event.get("content_block"))
    return event.get("type") == "content_block_start" and block is not None and block.get("type") == "tool_use"


def _block_shifted(frame: bytes, event: Mapping[str, object], block_offset: int) -> bytes:
    index: Final = event.get("index")
    if block_offset == 0 or event.get("type") not in _CONTENT_BLOCK_EVENTS or not isinstance(index, int):
        return frame
    return anthropic_sse_frame({**event, "index": index + block_offset})


async def _continuing_message(follow_up: AsyncIterator[bytes], block_offset: int) -> AsyncGenerator[bytes, None]:
    async with contextlib.aclosing(sse_frames(follow_up)) as frames:
        async for frame in frames:
            event = _frame_event(frame)
            if event is None:
                yield frame
            elif event.get("type") != "message_start":
                yield _block_shifted(frame, event, block_offset)


def _without_blocks_the_client_has(
    response: Mapping[str, object], live_message_id: object, live_block_count: int
) -> Mapping[str, object]:
    content: Final = response.get("content")
    if response.get("id") != live_message_id or not isinstance(content, list):
        return response
    return {**response, "content": _CONTENT_BLOCKS.validate_python(content)[live_block_count:]}


async def _replayed(frames: Sequence[bytes]) -> AsyncGenerator[bytes, None]:
    for frame in frames:
        yield frame


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
        named_event_type = current_event_type
        current_event_type = None
        data = loads_json_object(stripped[len("data:") :])
        if data is None:
            continue
        events.append((named_event_type or data.get("type", ""), data))
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

    In hold-back mode only the frames from the first tool_use block or the message's
    end onward wait for the hooks, and a follow-up continues the message the client is
    already reading, with its content blocks numbered after the ones it has seen.
    """

    def __init__(
        self,
        completion_stream: AsyncIterator,
        http_handler: Any,
        model: str,
        messages: list[dict],
        anthropic_messages_provider_config: "BaseAnthropicMessagesConfig",
        anthropic_messages_optional_request_params: dict,
        logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str,
        kwargs: dict,
        hold_back: bool = False,
        server_fulfilled_tool_names: frozenset[str] = frozenset(),
        ping_interval_seconds: float = HOLD_BACK_PING_INTERVAL_SECONDS,
    ):
        self._inner: Final[AsyncIterator[bytes]] = completion_stream.__aiter__()
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
        self._upstream_frames: Final = sse_frames(self._inner)
        self._holding = False
        self._live_frame_count = 0
        self._frame_task: asyncio.Task[bytes | None] | None = None
        self._hook_task: asyncio.Task[None] | None = None
        self._frames_after_hooks: AsyncIterator[bytes] | None = None
        self._loop_limit_reached = False

    @property
    def has_buffered_provider_output(self) -> bool:
        """Whether a held-back stream received provider output, sent live or still held behind keepalive pings."""
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

    async def _next_frame(
        self, frames: AsyncIterator[bytes], deadline: float
    ) -> bytes | None | Literal[_KeepAlive.PING_DUE]:
        if self._frame_task is None:
            self._frame_task = asyncio.create_task(_anext_or_none(frames))
        task: Final = self._frame_task
        if not await _settles_by(task, deadline):
            return _KeepAlive.PING_DUE
        frame: Final = task.result()
        self._frame_task = None
        return frame

    async def _anext_held_back(self) -> bytes:
        deadline: Final = asyncio.get_running_loop().time() + self._ping_interval_seconds
        while not self._stream_exhausted:
            frame = await self._next_frame(self._upstream_frames, deadline)
            if frame is _KeepAlive.PING_DUE:
                return STREAM_SSE_KEEPALIVE_PING_BYTES
            if frame is None:
                self._stream_exhausted = True
                continue
            self._collected_bytes.append(frame)
            self._holding = self._holding or _starts_hold(frame)
            if not self._holding:
                self._live_frame_count += 1
                return frame

        if self._hook_task is None:
            self._hook_task = asyncio.create_task(self._process_agentic_hooks())
        if not await _settles_by(self._hook_task, deadline):
            return STREAM_SSE_KEEPALIVE_PING_BYTES
        self._hook_task.result()

        if self._frames_after_hooks is None:
            self._frames_after_hooks = self._ending_frames()
        frame_after_hooks: Final = await self._next_frame(self._frames_after_hooks, deadline)
        if frame_after_hooks is _KeepAlive.PING_DUE:
            return STREAM_SSE_KEEPALIVE_PING_BYTES
        if frame_after_hooks is None:
            raise StopAsyncIteration
        return frame_after_hooks

    def _ending_frames(self) -> AsyncIterator[bytes]:
        if self._follow_up_iterator is not None:
            return _continuing_message(self._follow_up_iterator, self._live_block_count())
        if self._buffer_holds_server_fulfilled_tool_use():
            verbose_logger.error(
                "AgenticStreamingIterator: hooks did not replace a message containing a server-fulfilled "
                "tool_use [model=%s]; emitting an SSE error instead of leaking the tool call to the client",
                self._model,
            )
            if self._loop_limit_reached:
                return _replayed((AGENTIC_LOOP_LIMIT_ERROR_SSE_BYTES,))
            return _replayed((SERVER_FULFILLED_TOOL_LEAK_ERROR_SSE_BYTES,))
        return _replayed(tuple(self._collected_bytes[self._live_frame_count :]))

    def _live_block_count(self) -> int:
        return sum(1 for frame in self._collected_bytes[: self._live_frame_count] if _is_content_block_start(frame))

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
    async def _settle_task(task: asyncio.Task[object] | None) -> None:
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
        await self._settle_task(self._hook_task)
        await self._settle_task(self._frame_task)
        await self._upstream_frames.aclose()
        await aclose_if_supported(self._frames_after_hooks)
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

            result: Final[object] = _AGENTIC_HOOK_RESULT.validate_python(
                await self._http_handler._call_agentic_completion_hooks(
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
            )

            if result is None:
                return

            if has_attributes(result, AsyncIterable[object], "__aiter__"):
                self._follow_up_iterator = result.__aiter__()
            elif (whole_message := object_mapping(result)) is not None:
                from litellm.llms.anthropic.messages.native.fake_stream_iterator import (
                    FakeAnthropicMessagesStreamIterator,
                )
                from litellm.types.llms.anthropic_messages.anthropic_response import (
                    AnthropicMessagesResponse,
                )

                unseen: Final = _without_blocks_the_client_has(
                    whole_message, rebuilt.get("id"), self._live_block_count()
                )
                fake: Final = FakeAnthropicMessagesStreamIterator(response=cast(AnthropicMessagesResponse, unseen))
                self._follow_up_iterator = fake.__aiter__()
            else:
                verbose_logger.warning(
                    "AgenticStreamingIterator: Unexpected result type from hooks: %s",
                    type(result).__name__,
                )
        except AgenticLoopSafetyError as e:
            self._loop_limit_reached = isinstance(e, AgenticLoopLimitError)
            verbose_logger.warning("AgenticStreamingIterator: agentic loop refused [rail=%s]", type(e).__name__)

    @staticmethod
    def _rebuild_anthropic_response_from_sse(
        raw_bytes: list[bytes],
    ) -> dict[str, object] | None:
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

        content: Final[list[dict[str, object]]] = []
        response: Final[dict[str, object]] = {
            "id": "",
            "type": "message",
            "role": "assistant",
            "model": "",
            "content": content,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        content_blocks: Final[dict[int, dict[str, object]]] = {}
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
            content.append(block)

        return response
