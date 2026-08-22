# What is this?
## Translates OpenAI call to Anthropic `/v1/messages` format
import copy
import json
import traceback
from collections import deque
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Protocol,
    get_args,
)

from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.types.llms.anthropic import (
    AppliedEdit,
    CompactionBlock,
    ContentBlockDelta,
    ContextManagementResponse,
    MessageBlockDelta,
    StreamingContentBlockDeltaType,
    UsageDelta,
    UsageIteration,
)
from litellm.types.utils import AdapterCompletionStreamWrapper, Delta

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponseStream


_STREAMING_DELTA_TYPES: Final = frozenset(get_args(StreamingContentBlockDeltaType))


class _UsageDeltaWithIterations(UsageDelta, total=False):
    iterations: list[UsageIteration]


class _ChunkStream(Protocol):
    def __iter__(self) -> "Iterator[ModelResponseStream]": ...

    def __aiter__(self) -> "AsyncIterator[ModelResponseStream]": ...


def _optional_attr(obj: object, name: str) -> object:
    return getattr(obj, name, None)


def _optional_attr_sequence(obj: object, name: str) -> Sequence[object]:
    value: Final = getattr(obj, name, None)
    return value if value else ()


def _delta_payload_field(delta_type: StreamingContentBlockDeltaType) -> str:
    match delta_type:
        case "text_delta":
            return "text"
        case "input_json_delta":
            return "partial_json"
        case "thinking_delta":
            return "thinking"
        case "signature_delta":
            return "signature"
        case _:
            assert_never(delta_type)


class _CombinedChunkSplitter:
    """
    Splits a streaming chunk that carries BOTH response content and a
    ``finish_reason`` into two chunks: a content-only chunk followed by a
    finish-only chunk.

    ``AnthropicStreamWrapper`` (via ``translate_streaming_openai_response_to_anthropic``)
    assumes content and ``finish_reason`` never arrive in the same chunk — true for
    real provider streams, but false for fake-streamed providers (e.g. Vertex AI
    Gemma ``:predict``) where ``MockResponseIterator`` collapses the entire response
    into a single chunk. Without this split the assumption causes all content to be
    silently dropped (only the ``message_delta`` stop event is emitted).

    Supports both sync and async iteration, since ``AnthropicStreamWrapper`` exposes
    both ``__next__`` and ``__anext__``. An instance is single-mode: callers must
    iterate it either synchronously or asynchronously, never both — the two modes
    hold independent iterator references on the upstream stream and mixing them
    would advance them out of sync.
    """

    def __init__(self, completion_stream: _ChunkStream):
        self._stream: _ChunkStream = completion_stream
        self._sync_iter: Iterator[ModelResponseStream] | None = None
        self._async_iter: AsyncIterator[ModelResponseStream] | None = None
        self._buffer: deque[ModelResponseStream] = deque()

    @staticmethod
    def _is_combined(chunk: "ModelResponseStream") -> bool:
        """True if ``chunk`` carries response content AND a finish_reason."""
        choices: Final = _optional_attr_sequence(chunk, "choices")
        if not choices:
            return False
        choice: Final = choices[0]
        if _optional_attr(choice, "finish_reason") is None:
            return False
        delta: Final = _optional_attr(choice, "delta")
        if delta is None:
            return False
        return bool(
            _optional_attr(delta, "content")
            or _optional_attr(delta, "tool_calls")
            or _optional_attr(delta, "reasoning_content")
            or _optional_attr(delta, "thinking_blocks")
        )

    _PAYLOAD_FIELD_GROUPS: "tuple[tuple[str, ...], ...]" = (
        ("reasoning_content", "thinking_blocks"),
        ("content",),
        ("tool_calls",),
    )

    @staticmethod
    def _clear_usage(chunk: "ModelResponseStream") -> None:
        if hasattr(chunk, "usage"):
            chunk.usage = None
        hidden_params: Final = getattr(chunk, "_hidden_params", None)
        if isinstance(hidden_params, dict) and "usage" in hidden_params:
            chunk._hidden_params = {key: value for key, value in hidden_params.items() if key != "usage"}

    @staticmethod
    def _split_by_payload_kind(chunk: "ModelResponseStream") -> "tuple[ModelResponseStream, ...]":
        """Return ``(chunk,)``, or one piece per payload kind it carries.

        Each piece's delta is rebuilt as a fresh ``Delta`` carrying exactly one
        payload kind (reasoning, text, tool calls), in native Anthropic block
        order: thinking, then text, then tool_use. Runs downstream of
        ``_split``, which has already peeled ``finish_reason`` and usage onto
        their own finish chunk.

        Chunks that must not be split pass through unchanged: multi-choice
        chunks (the translators read every choice, so slicing one would drop
        or repeat payload) and tool-argument continuations (splitting one
        would close the in-flight ``tool_use`` block mid-arguments). A
        reasoning piece whose ``thinking_blocks`` carry no signature is
        normalized to ``reasoning_content`` so the synthesized block start
        stays empty and the thinking text is emitted exactly once.
        """
        choices: Final = _optional_attr_sequence(chunk, "choices")
        if len(choices) != 1:
            return (chunk,)
        delta: Final = _optional_attr(choices[0], "delta")
        if delta is None:
            return (chunk,)
        tool_calls: Final = _optional_attr_sequence(delta, "tool_calls")
        if tool_calls and not any(
            _optional_attr(_optional_attr(tool_call, "function"), "name") for tool_call in tool_calls
        ):
            return (chunk,)
        present_groups: Final = tuple(
            group
            for group in _CombinedChunkSplitter._PAYLOAD_FIELD_GROUPS
            if any(_optional_attr(delta, field) for field in group)
        )
        if len(present_groups) <= 1:
            return (chunk,)

        pieces: Final = tuple(copy.deepcopy(chunk) for _ in present_groups)
        for index, (piece, group) in enumerate(zip(pieces, present_groups)):
            copied_delta = piece.choices[0].delta
            fields = {field: value for field in group if (value := getattr(copied_delta, field, None))}
            fields = _CombinedChunkSplitter._normalize_reasoning_fields(fields)
            role = getattr(copied_delta, "role", None) if index == 0 else None
            piece.choices[0].delta = Delta(role=role, **fields)
        return pieces

    @staticmethod
    def _normalize_reasoning_fields(fields: "dict[str, Any]") -> "dict[str, Any]":
        """Collapse signature-less ``thinking_blocks`` into ``reasoning_content``.

        The block opener seeds a ``thinking_blocks`` start body with the full
        thinking text while the delta re-emits it, so SSE accumulators would
        collect it twice; the ``reasoning_content`` branch opens an empty body.
        Signature-carrying blocks are kept intact so ``signature_delta``
        suppression of the full-text snapshot still applies.
        """
        thinking_blocks: Final = fields.get("thinking_blocks")
        if not thinking_blocks:
            return fields
        if any(block.get("signature") for block in thinking_blocks if isinstance(block, dict)):
            return fields
        thinking_text: Final = "".join(
            block.get("thinking") or ""
            for block in thinking_blocks
            if isinstance(block, dict) and block.get("type") == "thinking"
        )
        if not thinking_text:
            return fields
        return {"reasoning_content": thinking_text}

    @staticmethod
    def _split(chunk: "ModelResponseStream") -> "list[ModelResponseStream]":
        """Return ``[chunk]``, or ``[content_chunk, finish_chunk]`` if combined."""
        if not _CombinedChunkSplitter._is_combined(chunk):
            return [chunk]

        # Content chunk: keep the delta payload, clear the finish_reason.
        content_chunk: Final = copy.deepcopy(chunk)
        content_chunk.choices[0].finish_reason = None
        _CombinedChunkSplitter._clear_usage(content_chunk)

        # Finish chunk: keep finish_reason (and usage), clear the delta payload.
        finish_chunk: Final = copy.deepcopy(chunk)
        finish_delta: Final = finish_chunk.choices[0].delta
        finish_delta.content = None
        if hasattr(finish_delta, "tool_calls"):
            finish_delta.tool_calls = None
        if hasattr(finish_delta, "reasoning_content"):
            finish_delta.reasoning_content = None
        if hasattr(finish_delta, "thinking_blocks"):
            finish_delta.thinking_blocks = None
        return [content_chunk, finish_chunk]

    def __iter__(self) -> "Iterator[ModelResponseStream]":
        return self

    def __next__(self) -> "ModelResponseStream":
        if self._buffer:
            return self._buffer.popleft()
        if self._sync_iter is None:
            self._sync_iter = iter(self._stream)
        chunk: Final = next(self._sync_iter)  # propagates StopIteration when exhausted
        self._buffer.extend(
            split_chunk
            for combined_chunk in self._split(chunk)
            for split_chunk in self._split_by_payload_kind(combined_chunk)
        )
        return self._buffer.popleft()

    def __aiter__(self) -> "AsyncIterator[ModelResponseStream]":
        return self

    async def __anext__(self) -> "ModelResponseStream":
        if self._buffer:
            return self._buffer.popleft()
        if self._async_iter is None:
            self._async_iter = self._stream.__aiter__()
        chunk: Final = await self._async_iter.__anext__()  # propagates StopAsyncIteration
        self._buffer.extend(
            split_chunk
            for combined_chunk in self._split(chunk)
            for split_chunk in self._split_by_payload_kind(combined_chunk)
        )
        return self._buffer.popleft()


class AnthropicStreamWrapper(AdapterCompletionStreamWrapper):
    """
    - first chunk return 'message_start'
    - content block must be started and stopped
    - finish_reason must map exactly to anthropic reason, else anthropic client won't be able to parse it.
    """

    from litellm.types.llms.anthropic import (
        ContentBlockContentBlockDict,
        ContentBlockStart,
        ContentBlockStartText,
        TextBlock,
    )

    sent_first_chunk: bool = False
    sent_content_block_start: bool = False
    sent_content_block_finish: bool = False
    current_content_block_type: Literal["text", "tool_use", "thinking"] = "text"
    sent_last_message: bool = False
    holding_chunk: ContentBlockDelta | None = None
    holding_stop_reason_chunk: MessageBlockDelta | None = None
    queued_usage_chunk: bool = False
    current_content_block_index: int = 0

    def __init__(
        self,
        completion_stream: _ChunkStream,
        model: str,
        tool_name_mapping: dict[str, str] | None = None,
        applied_edits: list[AppliedEdit] | None = None,
        compaction_block: CompactionBlock | None = None,
        iterations_usage: list[UsageIteration] | None = None,
    ):
        # Wrap the upstream stream so chunks that carry both content and a
        # finish_reason (fake-streamed providers) are split into two — see
        # _CombinedChunkSplitter.
        super().__init__(_CombinedChunkSplitter(completion_stream))
        self.model = model
        # Mapping of truncated tool names to original names (for OpenAI's 64-char limit)
        self.tool_name_mapping = tool_name_mapping or {}
        # Polyfill applied_edits on final message_delta.
        self.applied_edits: list[AppliedEdit] = list(applied_edits or [])
        # Synthesized compaction block from compact_20260112 polyfill (streaming).
        self.compaction_block = compaction_block
        self.iterations_usage = iterations_usage
        self.sent_compaction_block: bool = False
        # Per-phase flags so the compaction block's start/delta/stop events
        # are emitted (and the public state machine is advanced) in
        # lock-step with the caller actually consuming each event. Pre-
        # queuing all three would set ``sent_content_block_finish=True``
        # before the client received ``content_block_stop``, leaving the
        # observable state inconsistent during the drain window.
        self.sent_compaction_block_start: bool = False
        self.sent_compaction_block_delta: bool = False
        # Per-instance queue for buffering multiple chunks. Must be initialized
        # here (not at class level) so concurrent streams don't share the same
        # deque and corrupt each other's SSE event order.
        self.chunk_queue: deque = deque()
        # Per-instance default content block. Must be initialized here (not at
        # class level) so concurrent streams don't share the same mutable dict
        # — `_should_start_new_content_block` mutates `tool_block["name"]` in
        # place, which would otherwise leak across streams.
        self.current_content_block_start: AnthropicStreamWrapper.ContentBlockContentBlockDict = self.TextBlock(
            type="text",
            text="",
        )

    def _merge_usage_into_held_stop_reason_chunk(self, chunk: Any) -> MessageBlockDelta:
        """Merge usage data from ``chunk`` into the held ``message_delta`` chunk.

        Shared by both the sync ``__next__`` and async ``__anext__`` paths so
        the subtle hold-and-merge logic (cache tokens, ``context_management``
        attachment, ``UsageDelta`` shape) lives in exactly one place.

        Caller is responsible for managing ``self.holding_stop_reason_chunk``
        and ``self.queued_usage_chunk`` state and for queuing the returned
        merged chunk.
        """
        assert self.holding_stop_reason_chunk is not None
        merged_chunk: Final = self.holding_stop_reason_chunk.copy()
        if "delta" not in merged_chunk:
            merged_chunk["delta"] = {}

        from .transformation import LiteLLMAnthropicMessagesAdapter

        usage_dict: UsageDelta = LiteLLMAnthropicMessagesAdapter._translate_openai_usage_to_anthropic_usage_delta(
            chunk.usage
        )
        merged_chunk["usage"] = usage_dict
        if self.applied_edits and "context_management" not in merged_chunk:
            merged_chunk["context_management"] = ContextManagementResponse(applied_edits=list(self.applied_edits))
        return self._augment_message_delta_usage(merged_chunk)

    def _handle_choiceless_chunk(self, chunk: "ModelResponseStream") -> bool:
        """Consume an OpenAI-compatible chunk that carries no ``choices``.

        ``choices`` is legitimately empty on metadata-only chunks; the final
        usage chunk emitted when ``stream_options.include_usage`` is set is the
        common case (vLLM and other OpenAI-compatible servers do this). Such a
        chunk carries no content-block information, so the caller must not run
        the content-block state machine over it.

        Returns True when a merged ``message_delta`` was queued (usage folded
        into the held stop-reason chunk); False when the chunk should be
        skipped entirely.
        """
        if self.holding_stop_reason_chunk is not None and _optional_attr(chunk, "usage") is not None:
            self.chunk_queue.append(self._merge_usage_into_held_stop_reason_chunk(chunk))
            self.queued_usage_chunk = True
            self.holding_stop_reason_chunk = None
            return True
        return False

    def _ensure_context_management_attached(self, message_delta_chunk: MessageBlockDelta) -> MessageBlockDelta:
        """Attach ``context_management`` to a ``message_delta`` chunk if
        ``self.applied_edits`` is non-empty and the chunk does not already
        carry it. Returns the (possibly new) chunk dict.

        Centralizing this guard ensures every ``message_delta`` emission
        path (merge-with-usage and direct-flush-of-held) consistently
        surfaces ``applied_edits`` to the client.
        """
        if not self.applied_edits or "context_management" in message_delta_chunk:
            return message_delta_chunk
        augmented: Final = message_delta_chunk.copy()
        augmented["context_management"] = ContextManagementResponse(applied_edits=list(self.applied_edits))
        return augmented

    def _augment_message_delta_usage(self, message_delta_chunk: MessageBlockDelta) -> MessageBlockDelta:
        """Attach polyfill compaction iteration usage to the final message_delta.

        Also defensively re-attaches ``context_management`` so the direct
        held-chunk flush path stays in sync with the merge path's guarantee
        when ``self.applied_edits`` is non-empty.
        """
        message_delta_chunk = self._ensure_context_management_attached(message_delta_chunk)
        if self.iterations_usage is None:
            return message_delta_chunk
        usage: Final = message_delta_chunk.get("usage")
        if not isinstance(usage, dict) or "iterations" in usage:
            return message_delta_chunk

        input_tokens: Final = usage.get("input_tokens", 0) or 0
        output_tokens: Final = usage.get("output_tokens", 0) or 0
        augmented: Final = message_delta_chunk.copy()
        augmented_usage: Final[_UsageDeltaWithIterations] = {**usage}
        iterations: Final[list[UsageIteration]] = list(self.iterations_usage)
        # Only emit a ``message`` iteration when we have real token data.
        # Without a separate usage chunk (e.g. provider sent finish_reason
        # alone), the held ``message_delta`` carries placeholder zeros from
        # the translate step; reporting a zero-token iteration would be
        # misleading and inconsistent with the non-streaming path.
        if input_tokens > 0 or output_tokens > 0:
            message_iteration: Final[UsageIteration] = {
                "type": "message",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            iterations.append(message_iteration)
        augmented_usage["iterations"] = iterations
        augmented["usage"] = augmented_usage
        return augmented

    def _next_compaction_event(self) -> dict[str, Any] | None:
        """Return the next compaction content-block SSE event, or ``None``.

        Anthropic delivers compaction as a single delta (no token-by-token
        streaming), but we still surface it as a proper
        start → delta → stop trio. Each call returns exactly one event so
        the state machine (``sent_content_block_finish``,
        ``current_content_block_index``) is advanced *only* when the
        terminal stop event is actually handed back to the caller. This
        prevents an observable window where the flags claim the block is
        finished while the stop event is still buffered.
        """
        if self.compaction_block is None or self.sent_compaction_block:
            return None

        compaction_index: Final = self.current_content_block_index

        if not self.sent_compaction_block_start:
            self.sent_compaction_block_start = True
            return {
                "type": "content_block_start",
                "index": compaction_index,
                # Mirror the text-block shape ({"type": "text", "text": ""}):
                # send an empty ``content`` field so clients that introspect
                # ``content_block_start`` see the full block schema. The
                # actual summary text arrives via the ``content_block_delta``
                # below.
                "content_block": {"type": "compaction", "content": ""},
            }

        if not self.sent_compaction_block_delta:
            self.sent_compaction_block_delta = True
            summary_content: Final = self.compaction_block.get("content") or ""
            return {
                "type": "content_block_delta",
                "index": compaction_index,
                "delta": {"type": "compaction_delta", "content": summary_content},
            }

        stop_event: Final = {
            "type": "content_block_stop",
            "index": compaction_index,
        }
        # Don't touch ``sent_content_block_finish`` here: that flag is the
        # state machine for the regular text/tool_use/thinking block and is
        # independent of the synthetic compaction block lifecycle. Conflating
        # them would let outside observers (subclass overrides, introspection
        # hooks, exception paths) see ``sent_content_block_finish=True``
        # without any regular content block ever having started.
        self._increment_content_block_index()
        self.sent_compaction_block = True
        return stop_event

    def _create_initial_usage_delta(self) -> UsageDelta:
        """
        Create the initial UsageDelta for the message_start event.

        Initializes cache token fields (cache_creation_input_tokens, cache_read_input_tokens)
        to 0 to indicate to clients (like Claude Code) that prompt caching is supported.

        The actual cache token values will be provided in the message_delta event at the
        end of the stream, since Bedrock Converse API only returns usage data in the final
        response chunk.

        Returns:
            UsageDelta with all token counts initialized to 0.
        """
        return UsageDelta(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

    def __next__(self):
        from .transformation import LiteLLMAnthropicMessagesAdapter

        try:
            # Always return queued chunks first
            if self.chunk_queue:
                return self.chunk_queue.popleft()

            # Queue initial chunks if not sent yet
            if self.sent_first_chunk is False:
                self.sent_first_chunk = True
                self.chunk_queue.append(
                    {
                        "type": "message_start",
                        "message": {
                            "id": f"msg_{uuid.uuid4()}",
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self.model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": self._create_initial_usage_delta(),
                        },
                    }
                )
                return self.chunk_queue.popleft()

            if self.sent_compaction_block is False and self.compaction_block is not None:
                compaction_event: Final = self._next_compaction_event()
                if compaction_event is not None:
                    return compaction_event

            for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception

                if not getattr(chunk, "choices", None):
                    if self._handle_choiceless_chunk(chunk):
                        return self.chunk_queue.popleft()
                    continue

                should_start_new_block = self._should_start_new_content_block(chunk)
                is_opening_first_block = self.sent_content_block_start is False
                if is_opening_first_block and self._is_blank_delta(chunk):
                    continue
                if is_opening_first_block:
                    self.sent_content_block_start = True
                    self.sent_content_block_finish = False
                    self.chunk_queue.append(
                        {
                            "type": "content_block_start",
                            "index": self.current_content_block_index,
                            "content_block": self.current_content_block_start,
                        }
                    )
                elif should_start_new_block:
                    self._increment_content_block_index()

                # applied_edits only needs to flow to the final message_delta
                # (when finish_reason is set); skip threading it through every
                # intermediate chunk. For the hold-and-merge path below,
                # context_management is attached directly to the merged chunk,
                # so the translated ``processed_chunk`` would be discarded —
                # skip the applied_edits attachment in that case to avoid
                # allocating a throwaway ``MessageBlockDelta``.
                will_merge_into_held = (
                    self.holding_stop_reason_chunk is not None and getattr(chunk, "usage", None) is not None
                )
                is_final_chunk = chunk.choices[0].finish_reason is not None
                processed_chunk = LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                    response=chunk,
                    current_content_block_index=self.current_content_block_index,
                    applied_edits=(self.applied_edits if is_final_chunk and not will_merge_into_held else None),
                )

                # Check if this is a usage chunk and we have a held stop_reason chunk
                if will_merge_into_held:
                    merged_chunk = self._merge_usage_into_held_stop_reason_chunk(chunk)
                    self.chunk_queue.append(merged_chunk)
                    self.queued_usage_chunk = True
                    self.holding_stop_reason_chunk = None
                    return self.chunk_queue.popleft()

                if self.queued_usage_chunk:
                    # Usage has already been merged + emitted. Any trailing
                    # provider events would violate Anthropic SSE ordering
                    # (no chunks may follow the final ``message_delta``), so
                    # silently drop them — matches the async ``__anext__``
                    # behavior where the block-handling logic is gated on
                    # ``not self.queued_usage_chunk``.
                    continue

                if should_start_new_block and not is_opening_first_block and not self.sent_content_block_finish:
                    # Queue the sequence: content_block_stop -> content_block_start
                    # -> (optionally) the trigger chunk's delta.
                    #
                    # The synthesized content_block_start always carries an
                    # empty body, so the chunk that *triggered* the transition
                    # also carries the new block's first delta. It must be
                    # re-emitted or the first token of the new block is lost.
                    # This applies to text_delta and thinking_delta (the first
                    # non-empty text/thinking token) as well as input_json_delta
                    # (providers like xAI/Gemini bundle tool arguments with the
                    # function name/id in a single chunk).

                    # 1. Stop current content block
                    self.chunk_queue.append(
                        {
                            "type": "content_block_stop",
                            "index": max(self.current_content_block_index - 1, 0),
                        }
                    )

                    # 2. Start new content block
                    self.chunk_queue.append(
                        {
                            "type": "content_block_start",
                            "index": self.current_content_block_index,
                            "content_block": self.current_content_block_start,
                        }
                    )

                    # 3. If the trigger chunk carries delta content, queue it
                    # so the first delta of the new block is not silently dropped.
                    if self._delta_has_content(processed_chunk):
                        self.chunk_queue.append(processed_chunk)

                    self.sent_content_block_finish = False
                    return self.chunk_queue.popleft()

                if processed_chunk["type"] == "content_block_delta" and not self._delta_has_content(processed_chunk):
                    # A tool_use block opens with empty arguments (Bedrock Converse's
                    # ``contentBlockStart``, OpenAI's ``arguments: ""``), so flush the
                    # block start queued above instead of waiting for the next upstream
                    # chunk, which on a trailing-burst provider is the whole generation.
                    if self.chunk_queue:
                        return self.chunk_queue.popleft()
                    continue

                if processed_chunk["type"] == "message_delta" and self.sent_content_block_finish is False:
                    # Queue both the content_block_stop and the message_delta
                    self.chunk_queue.append(
                        {
                            "type": "content_block_stop",
                            "index": self.current_content_block_index,
                        }
                    )
                    self.sent_content_block_finish = True
                    if processed_chunk.get("delta", {}).get("stop_reason") is not None:
                        self.holding_stop_reason_chunk = processed_chunk
                    else:
                        processed_chunk = self._augment_message_delta_usage(processed_chunk)
                        self.chunk_queue.append(processed_chunk)
                    return self.chunk_queue.popleft()
                elif self.holding_chunk is not None:
                    self.chunk_queue.append(self.holding_chunk)
                    if processed_chunk.get("type") == "message_delta":
                        processed_chunk = self._augment_message_delta_usage(processed_chunk)
                    self.chunk_queue.append(processed_chunk)
                    self.holding_chunk = None
                    return self.chunk_queue.popleft()
                else:
                    if processed_chunk.get("type") == "message_delta":
                        processed_chunk = self._augment_message_delta_usage(processed_chunk)
                    self.chunk_queue.append(processed_chunk)
                    return self.chunk_queue.popleft()

            # Handle any remaining held chunks after stream ends. The
            # buffered ``holding_chunk`` (a ``content_block_delta``) must
            # precede the final ``message_delta`` so Anthropic SSE event
            # ordering is preserved. When ``queued_usage_chunk`` is True,
            # the final ``message_delta`` has already been emitted; any
            # buffered content delta is dropped rather than emitted after
            # ``message_delta`` (which would violate SSE ordering and may
            # confuse strict Anthropic SDK clients).
            if not self.queued_usage_chunk:
                if self.holding_chunk is not None:
                    self.chunk_queue.append(self.holding_chunk)
                    self.holding_chunk = None
                if self.holding_stop_reason_chunk is not None:
                    # A final ``message_delta`` must be preceded by
                    # ``content_block_stop`` so the emitted SSE stays in
                    # valid Anthropic order (... -> content_block_stop ->
                    # message_delta). Emit ``content_block_stop`` here if
                    # the active content block was not already closed.
                    if not self.sent_content_block_finish:
                        self.chunk_queue.append(
                            {
                                "type": "content_block_stop",
                                "index": self.current_content_block_index,
                            }
                        )
                        self.sent_content_block_finish = True
                    self.chunk_queue.append(self._augment_message_delta_usage(self.holding_stop_reason_chunk))
                    self.holding_stop_reason_chunk = None
            else:
                self.holding_chunk = None

            if not self.sent_last_message:
                self.sent_last_message = True
                self.chunk_queue.append({"type": "message_stop"})

            if self.chunk_queue:
                return self.chunk_queue.popleft()

            raise StopIteration
        except StopIteration:
            if self.chunk_queue:
                return self.chunk_queue.popleft()
            # Handle any held stop_reason chunk. Emit ``content_block_stop``
            # first if the active content block was not already closed, so
            # Anthropic SSE ordering is preserved (content_block_stop ->
            # message_delta).
            if self.holding_stop_reason_chunk is not None:
                if not self.sent_content_block_finish:
                    self.sent_content_block_finish = True
                    self.chunk_queue.append(self._augment_message_delta_usage(self.holding_stop_reason_chunk))
                    self.holding_stop_reason_chunk = None
                    return {
                        "type": "content_block_stop",
                        "index": self.current_content_block_index,
                    }
                held: Final = self._augment_message_delta_usage(self.holding_stop_reason_chunk)
                self.holding_stop_reason_chunk = None
                return held
            if self.sent_last_message is False:
                self.sent_last_message = True
                return {"type": "message_stop"}
            raise StopIteration
        except Exception as e:
            verbose_logger.error("Anthropic Adapter - %s\n%s", e, traceback.format_exc())
            raise StopIteration

    async def __anext__(self):
        from .transformation import LiteLLMAnthropicMessagesAdapter

        try:
            # Always return queued chunks first
            if self.chunk_queue:
                return self.chunk_queue.popleft()

            # Queue initial chunks if not sent yet
            if self.sent_first_chunk is False:
                self.sent_first_chunk = True
                self.chunk_queue.append(
                    {
                        "type": "message_start",
                        "message": {
                            "id": f"msg_{uuid.uuid4()}",
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self.model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": self._create_initial_usage_delta(),
                        },
                    }
                )
                return self.chunk_queue.popleft()

            if self.sent_compaction_block is False and self.compaction_block is not None:
                compaction_event: Final = self._next_compaction_event()
                if compaction_event is not None:
                    return compaction_event

            async for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception

                if not getattr(chunk, "choices", None):
                    if self._handle_choiceless_chunk(chunk):
                        return self.chunk_queue.popleft()
                    continue

                should_start_new_block = self._should_start_new_content_block(chunk)
                is_opening_first_block = self.sent_content_block_start is False
                if is_opening_first_block and self._is_blank_delta(chunk):
                    continue
                if is_opening_first_block:
                    self.sent_content_block_start = True
                    self.sent_content_block_finish = False
                    self.chunk_queue.append(
                        {
                            "type": "content_block_start",
                            "index": self.current_content_block_index,
                            "content_block": self.current_content_block_start,
                        }
                    )
                elif should_start_new_block:
                    self._increment_content_block_index()

                # applied_edits only needs to flow to the final message_delta
                # (when finish_reason is set); skip threading it through every
                # intermediate chunk. For the hold-and-merge path below,
                # context_management is attached directly to the merged chunk,
                # so the translated ``processed_chunk`` would be discarded —
                # skip the applied_edits attachment in that case to avoid
                # allocating a throwaway ``MessageBlockDelta``.
                will_merge_into_held = (
                    self.holding_stop_reason_chunk is not None and getattr(chunk, "usage", None) is not None
                )
                is_final_chunk = chunk.choices[0].finish_reason is not None
                processed_chunk = LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                    response=chunk,
                    current_content_block_index=self.current_content_block_index,
                    applied_edits=(self.applied_edits if is_final_chunk and not will_merge_into_held else None),
                )

                # Check if this is a usage chunk and we have a held stop_reason chunk
                if will_merge_into_held:
                    merged_chunk = self._merge_usage_into_held_stop_reason_chunk(chunk)
                    self.chunk_queue.append(merged_chunk)
                    self.queued_usage_chunk = True
                    self.holding_stop_reason_chunk = None
                    return self.chunk_queue.popleft()

                # Check if this processed chunk has a stop_reason - hold it for next chunk

                if not self.queued_usage_chunk:
                    if should_start_new_block and not is_opening_first_block and not self.sent_content_block_finish:
                        # Queue the sequence: content_block_stop -> content_block_start
                        # -> (optionally) the trigger chunk's delta.
                        #
                        # The synthesized content_block_start always carries an
                        # empty body, so the chunk that *triggered* the transition
                        # also carries the new block's first delta. It must be
                        # re-emitted or the first token of the new block is lost.
                        # This applies to text_delta and thinking_delta (the
                        # first non-empty text/thinking token) as well as
                        # input_json_delta (providers like xAI/Gemini bundle tool
                        # arguments with the function name/id in a single chunk).

                        # 1. Stop current content block
                        self.chunk_queue.append(
                            {
                                "type": "content_block_stop",
                                "index": max(self.current_content_block_index - 1, 0),
                            }
                        )
                        self.chunk_queue.append(
                            {
                                "type": "content_block_start",
                                "index": self.current_content_block_index,
                                "content_block": self.current_content_block_start,
                            }
                        )

                        # 3. If the trigger chunk carries delta content, queue it
                        # so the first delta of the new block is not silently dropped.
                        if self._delta_has_content(processed_chunk):
                            self.chunk_queue.append(processed_chunk)

                        # Reset state for new block
                        self.sent_content_block_finish = False
                        return self.chunk_queue.popleft()

                    if processed_chunk["type"] == "content_block_delta" and not self._delta_has_content(
                        processed_chunk
                    ):
                        # See ``__next__``: flush the queued block start (issue #32004).
                        if self.chunk_queue:
                            return self.chunk_queue.popleft()
                        continue

                    if processed_chunk["type"] == "message_delta" and self.sent_content_block_finish is False:
                        # Queue both the content_block_stop and the holding chunk
                        self.chunk_queue.append(
                            {
                                "type": "content_block_stop",
                                "index": self.current_content_block_index,
                            }
                        )
                        self.sent_content_block_finish = True
                        if processed_chunk.get("delta", {}).get("stop_reason") is not None:
                            self.holding_stop_reason_chunk = processed_chunk
                        else:
                            processed_chunk = self._augment_message_delta_usage(processed_chunk)
                            self.chunk_queue.append(processed_chunk)
                        return self.chunk_queue.popleft()
                    elif self.holding_chunk is not None:
                        # Queue both chunks
                        self.chunk_queue.append(self.holding_chunk)
                        if processed_chunk.get("type") == "message_delta":
                            processed_chunk = self._augment_message_delta_usage(processed_chunk)
                        self.chunk_queue.append(processed_chunk)
                        self.holding_chunk = None
                        return self.chunk_queue.popleft()
                    else:
                        if processed_chunk.get("type") == "message_delta":
                            processed_chunk = self._augment_message_delta_usage(processed_chunk)
                        self.chunk_queue.append(processed_chunk)
                        return self.chunk_queue.popleft()

            # Handle any remaining held chunks after stream ends. The
            # buffered ``holding_chunk`` (a ``content_block_delta``) must
            # precede the final ``message_delta`` so Anthropic SSE event
            # ordering is preserved. When ``queued_usage_chunk`` is True,
            # the final ``message_delta`` has already been emitted; any
            # buffered content delta is dropped rather than emitted after
            # ``message_delta`` (which would violate SSE ordering and may
            # confuse strict Anthropic SDK clients).
            if not self.queued_usage_chunk:
                if self.holding_chunk is not None:
                    self.chunk_queue.append(self.holding_chunk)
                    self.holding_chunk = None
                if self.holding_stop_reason_chunk is not None:
                    # A final ``message_delta`` must be preceded by
                    # ``content_block_stop`` so the emitted SSE stays in
                    # valid Anthropic order (... -> content_block_stop ->
                    # message_delta). Emit ``content_block_stop`` here if
                    # the active content block was not already closed.
                    if not self.sent_content_block_finish:
                        self.chunk_queue.append(
                            {
                                "type": "content_block_stop",
                                "index": self.current_content_block_index,
                            }
                        )
                        self.sent_content_block_finish = True
                    self.chunk_queue.append(self._augment_message_delta_usage(self.holding_stop_reason_chunk))
                    self.holding_stop_reason_chunk = None
            else:
                self.holding_chunk = None

            if not self.sent_last_message:
                self.sent_last_message = True
                self.chunk_queue.append({"type": "message_stop"})

            # Return queued items if any
            if self.chunk_queue:
                return self.chunk_queue.popleft()

            raise StopIteration

        except StopIteration:
            # Handle any remaining queued chunks before stopping
            if self.chunk_queue:
                return self.chunk_queue.popleft()
            # Handle any held stop_reason chunk — clear after capturing so a
            # subsequent ``__anext__`` call doesn't re-emit the same chunk
            # (matches the sync ``__next__`` path). Emit ``content_block_stop``
            # first if the active content block was not already closed, so
            # Anthropic SSE ordering is preserved (content_block_stop ->
            # message_delta).
            if self.holding_stop_reason_chunk is not None:
                if not self.sent_content_block_finish:
                    self.sent_content_block_finish = True
                    self.chunk_queue.append(self._augment_message_delta_usage(self.holding_stop_reason_chunk))
                    self.holding_stop_reason_chunk = None
                    return {
                        "type": "content_block_stop",
                        "index": self.current_content_block_index,
                    }
                held: Final = self._augment_message_delta_usage(self.holding_stop_reason_chunk)
                self.holding_stop_reason_chunk = None
                return held
            if not self.sent_last_message:
                self.sent_last_message = True
                return {"type": "message_stop"}
            raise StopAsyncIteration

    def anthropic_sse_wrapper(self) -> Iterator[bytes]:
        """
        Convert AnthropicStreamWrapper dict chunks to Server-Sent Events format.
        Similar to the Bedrock bedrock_sse_wrapper implementation.

        This wrapper ensures dict chunks are SSE formatted with both event and data lines.
        """
        for chunk in self:
            if isinstance(chunk, dict):
                event_type: str = str(chunk.get("type", "message"))
                payload = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                yield payload.encode()
            else:
                # For non-dict chunks, forward the original value unchanged
                yield chunk

    async def async_anthropic_sse_wrapper(self) -> AsyncIterator[bytes]:
        """
        Async version of anthropic_sse_wrapper.
        Convert AnthropicStreamWrapper dict chunks to Server-Sent Events format.
        """
        async for chunk in self:
            if isinstance(chunk, dict):
                event_type: str = str(chunk.get("type", "message"))
                payload = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                yield payload.encode()
            else:
                # For non-dict chunks, forward the original value unchanged
                yield chunk

    def _increment_content_block_index(self):
        self.current_content_block_index += 1

    @staticmethod
    def _delta_has_content(processed_chunk: dict[str, Any]) -> bool:
        """Return True if a translated chunk carries a non-empty
        ``content_block_delta`` payload.

        Gates every ``content_block_delta`` emission. An empty delta carries
        no information, and the translate fallback types empty deltas as
        ``text_delta`` regardless of the active block's type — emitting one
        into an open ``thinking`` block (e.g. Bedrock Converse sends an empty
        reasoning delta mid-block) crashes strict Anthropic SDK clients with
        "Content block is not a text block".

        Also gates re-emission after a block transition: when an upstream
        chunk both *triggers* a new content block (its type differs from the
        active block) and *carries* delta content, that content belongs to
        the new block. The synthesized ``content_block_start`` only ever
        carries an empty body — see
        ``_translate_streaming_openai_chunk_to_anthropic_content_block``,
        which returns an empty ``TextBlock``/``ToolUseBlock``/thinking block —
        so the trigger chunk's delta must be re-queued or the first token of
        the new block (the first non-empty text/thinking delta, or bundled
        tool arguments) is silently dropped.

        Delta types outside ``StreamingContentBlockDeltaType`` — the closed
        set the translate layer can produce — are treated as empty. The
        per-type payload lookup is exhaustively matched against that set in
        ``_delta_payload_field``, so extending the translate layer with a new
        delta type fails type-checking here until it is handled.
        """
        if processed_chunk.get("type") != "content_block_delta":
            return False
        delta: Final = processed_chunk.get("delta")
        if not isinstance(delta, dict):
            return False
        delta_type: Final = delta.get("type")
        if delta_type not in _STREAMING_DELTA_TYPES:
            return False
        return bool(delta.get(_delta_payload_field(delta_type)))

    @staticmethod
    def _is_blank_delta(chunk: "ModelResponseStream") -> bool:
        choice: Final = chunk.choices[0]
        if choice.finish_reason is not None:
            return False
        delta: Final = choice.delta
        if getattr(delta, "tool_calls", None):
            return False
        if getattr(delta, "content", None):
            return False
        if getattr(delta, "reasoning_content", None):
            return False
        if getattr(delta, "thinking_blocks", None):
            return False
        return True

    def _should_start_new_content_block(self, chunk: "ModelResponseStream") -> bool:
        """
        Determine if we should start a new content block based on the processed chunk.
        Override this method with your specific logic for detecting new content blocks.

        Examples of when you might want to start a new content block:
        - Switching from text to tool calls
        - Different content types in the response
        - Specific markers in the content
        """
        from .transformation import LiteLLMAnthropicMessagesAdapter

        # Example logic - customize based on your needs:
        # If chunk indicates a tool call
        if chunk.choices[0].finish_reason is not None:
            return False

        (
            block_type,
            content_block_start,
        ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic_content_block(
            choices=chunk.choices
        )

        # Restore original tool name if it was truncated for OpenAI's 64-char limit
        if block_type == "tool_use":
            # Type narrowing: content_block_start is ToolUseBlock when block_type is "tool_use"
            from typing import cast

            from litellm.types.llms.anthropic import ToolUseBlock

            tool_block = cast(ToolUseBlock, content_block_start)

            if tool_block.get("name"):
                truncated_name: Final = tool_block["name"]
                original_name: Final = self.tool_name_mapping.get(truncated_name, truncated_name)
                tool_block["name"] = original_name

        if block_type != self.current_content_block_type:
            self.current_content_block_type = block_type
            self.current_content_block_start = content_block_start
            return True

        # For parallel tool calls, we'll necessarily have a new content block
        # if we get a function name since it signals a new tool call
        if block_type == "tool_use":
            from typing import cast

            from litellm.types.llms.anthropic import ToolUseBlock

            tool_block = cast(ToolUseBlock, content_block_start)
            if tool_block.get("name"):
                self.current_content_block_type = block_type
                self.current_content_block_start = content_block_start
                return True

        return False
