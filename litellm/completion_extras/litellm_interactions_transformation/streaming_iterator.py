from __future__ import annotations

import json
import uuid
from collections import deque
from collections.abc import AsyncIterator, Iterator, Mapping
from types import MappingProxyType
from typing import Final, NoReturn, TypeAlias

import httpx
from pydantic import JsonValue, TypeAdapter

from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from litellm.litellm_core_utils.prompt_templates.factory import encode_tool_call_id_with_signature
from litellm.types.llms.openai import (
    ChatCompletionThinkingBlock,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
)
from litellm.types.utils import (
    Delta,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)

from .response_transformation import (
    FinishReason,
    map_interactions_status_to_finish_reason,
    transform_interactions_usage,
)

JsonMapping: TypeAlias = Mapping[str, JsonValue]
LineStreamSync: TypeAlias = Iterator[str | bytes]
LineStreamAsync: TypeAlias = AsyncIterator[str | bytes]

_JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_EMPTY_JSON_MAP: Final[JsonMapping] = MappingProxyType({})


def _extract_sse_payload(raw_line: str | bytes) -> str | None:
    text_line: Final = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
    stripped: Final = text_line.strip()
    if not stripped or not stripped.startswith("data:"):
        return None
    return stripped.removeprefix("data:").strip()


def _parse_event_json(data_str: str) -> JsonMapping | None:
    try:
        parsed: Final = _JSON_VALUE.validate_json(data_str)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _text_from_blocks(raw_blocks: JsonValue) -> str:
    if isinstance(raw_blocks, str):
        return raw_blocks
    if not isinstance(raw_blocks, list):
        return ""
    return "".join(
        str(block.get("text") or "") for block in raw_blocks if isinstance(block, dict) and block.get("type") == "text"
    )


def _format_initial_arguments(raw_args: JsonValue) -> str:
    if isinstance(raw_args, str):
        return raw_args
    if isinstance(raw_args, dict) and raw_args:
        return json.dumps(raw_args)
    return ""


def _thought_summary_delta_text(delta_map: JsonMapping) -> str:
    nested: Final = delta_map.get("content")
    if isinstance(nested, dict) and isinstance(nested.get("text"), str):
        return str(nested["text"])
    raw_text: Final = delta_map.get("text")
    return raw_text if isinstance(raw_text, str) else ""


def _delta_field_str(delta_map: JsonMapping, field_name: str) -> str | None:
    val: Final = delta_map.get(field_name)
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return json.dumps(val)
    return None


def _arguments_delta_piece(delta_map: JsonMapping) -> str:
    candidates: Final = tuple(_delta_field_str(delta_map, name) for name in ("arguments", "partial_arguments", "text"))
    return next((piece for piece in candidates if piece is not None), "")


def _extract_stream_error_code_and_message(
    event: JsonMapping,
    interaction_map: JsonMapping,
    *,
    default_code: str | None = None,
) -> tuple[str | None, str]:
    raw_err: Final = event.get("error") or interaction_map.get("error")
    if isinstance(raw_err, dict):
        raw_code: Final = raw_err.get("code")
        raw_msg: Final = raw_err.get("message")
        code_str: Final = str(raw_code) if raw_code is not None and str(raw_code) else default_code
        msg_str: Final = str(raw_msg) if raw_msg is not None and str(raw_msg) else None
        if code_str and msg_str:
            return code_str, f"[{code_str}] {msg_str}"
        if msg_str:
            return code_str, msg_str
        if code_str:
            return code_str, code_str
        return default_code, json.dumps(raw_err)
    if isinstance(raw_err, str) and raw_err:
        return default_code, raw_err
    fallback_msg: Final = event.get("message") or interaction_map.get("status_message")
    if isinstance(fallback_msg, str) and fallback_msg:
        return default_code, fallback_msg
    if default_code == "cancelled":
        return "cancelled", "[cancelled] Interaction stream was cancelled"
    return default_code, "Interactions stream failed"


def _build_stream_exception(
    *,
    code_str: str | None,
    message: str,
    model: str,
    llm_provider: str,
    request: httpx.Request,
) -> Exception:
    normalized: Final = code_str.lower().rsplit("/", 1)[-1] if code_str else ""
    match normalized:
        case "invalid_argument" | "failed_precondition" | "out_of_range" | "400":
            return BadRequestError(
                message=message,
                model=model,
                llm_provider=llm_provider,
                response=httpx.Response(400, request=request),
            )
        case "unauthenticated" | "401":
            return AuthenticationError(
                message=message,
                llm_provider=llm_provider,
                model=model,
                response=httpx.Response(401, request=request),
            )
        case "permission_denied" | "403":
            return PermissionDeniedError(
                message=message,
                llm_provider=llm_provider,
                model=model,
                response=httpx.Response(403, request=request),
            )
        case "not_found" | "404":
            return NotFoundError(
                message=message,
                model=model,
                llm_provider=llm_provider,
                response=httpx.Response(404, request=request),
            )
        case "resource_exhausted" | "rate_limit_exceeded" | "too_many_requests" | "429":
            return RateLimitError(
                message=message,
                model=model,
                llm_provider=llm_provider,
                response=httpx.Response(429, request=request),
            )
        case _:
            return APIConnectionError(
                message=message,
                llm_provider=llm_provider,
                model=model,
                request=request,
            )


class InteractionsChatStreamingIterator:
    def __init__(
        self,
        *,
        model: str,
        custom_llm_provider: str = "gemini",
        sync_lines: LineStreamSync | None = None,
        async_lines: LineStreamAsync | None = None,
        http_response: httpx.Response | None = None,
    ) -> None:
        self._model: Final = model
        self._custom_llm_provider: Final = custom_llm_provider
        self._sync_lines: Final = sync_lines
        self._async_lines: Final = async_lines
        self.http_response: httpx.Response | None = http_response
        self._stream_id: str = f"chatcmpl-{uuid.uuid4()}"
        self._active_thought_step: int | None = None
        self._pending_thought_signature: str | None = None
        self._segment_signed_tool_call: bool = False
        self._unsigned_tools: Final[dict[int, tuple[int, str]]] = {}  # mutable-ok: thought-step -> unsigned tool slot
        self._thought_slots: Final[dict[int, int]] = {}  # mutable-ok: per-stream step-index -> thought-slot state
        self._tool_slots_by_step: Final[dict[int, int]] = {}  # mutable-ok: per-stream step-index -> tool-slot state
        self._buffer: Final[deque[ModelResponseStream]] = deque()  # mutable-ok: per-stream output FIFO queue
        self._done: bool = False

    def close(self) -> None:
        if self.http_response is not None:
            self.http_response.close()

    async def aclose(self) -> None:
        if self.http_response is not None:
            await self.http_response.aclose()

    def _build_chunk(
        self,
        delta: Delta,
        *,
        finish_reason: FinishReason | None = None,
        usage: Usage | None = None,
    ) -> ModelResponseStream:
        return ModelResponseStream(
            id=self._stream_id,
            model=self._model,
            choices=[
                StreamingChoices(
                    index=0,
                    delta=delta,
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )

    def _allocate_thought_slot(self, step_index: int) -> int:
        existing: Final = self._thought_slots.get(step_index)
        if existing is not None:
            return existing
        slot: Final = len(self._thought_slots)
        self._thought_slots[step_index] = slot
        return slot

    def _allocate_tool_slot(self, step_index: int) -> int:
        existing: Final = self._tool_slots_by_step.get(step_index)
        if existing is not None:
            return existing
        slot: Final = len(self._tool_slots_by_step)
        self._tool_slots_by_step[step_index] = slot
        return slot

    def _record_tool_call_sig_state(self, slot: int, raw_id: str, effective_sig: str | None) -> None:
        thought_idx: Final = self._active_thought_step
        if effective_sig is not None:
            if thought_idx is not None:
                self._unsigned_tools.pop(thought_idx, None)
            self._segment_signed_tool_call = True
        elif not self._segment_signed_tool_call and thought_idx is not None and thought_idx not in self._unsigned_tools:
            self._unsigned_tools[thought_idx] = (slot, raw_id)

    def _consume_delta_tool_call_id(self, slot: int, raw_id: JsonValue) -> str | None:
        if not isinstance(raw_id, str) or not raw_id:
            return None
        consumed_sig: Final = self._pending_thought_signature
        self._pending_thought_signature = None
        self._record_tool_call_sig_state(slot, raw_id, consumed_sig)
        return encode_tool_call_id_with_signature(raw_id, consumed_sig)

    def _handle_step_start(self, event: JsonMapping) -> tuple[ModelResponseStream, ...]:
        raw_index: Final = event.get("index")
        step_index: Final = raw_index if isinstance(raw_index, int) else 0
        raw_step: Final = event.get("step") or event.get("content")
        if not isinstance(raw_step, dict):
            return ()

        match raw_step.get("type"):
            case "thought":
                thought_slot: Final = self._allocate_thought_slot(step_index)
                sig_raw: Final = raw_step.get("signature")
                sig_str: Final = sig_raw if isinstance(sig_raw, str) and sig_raw else None
                summary: Final = _text_from_blocks(raw_step.get("summary"))
                self._active_thought_step = step_index
                self._pending_thought_signature = sig_str
                self._segment_signed_tool_call = False
                if not summary and sig_str is None:
                    return ()
                block: Final = ChatCompletionThinkingBlock(
                    type="thinking",
                    thinking=summary,
                    signature=sig_str,
                    index=thought_slot,
                )
                return (
                    self._build_chunk(
                        Delta(
                            role="assistant",
                            reasoning_content=summary or None,
                            thinking_blocks=[block],
                        )
                    ),
                )
            case "function_call":
                slot: Final = self._allocate_tool_slot(step_index)
                raw_id: Final = str(raw_step.get("id") or f"call_{slot}")
                fn_name: Final = str(raw_step.get("name") or "")
                step_sig_raw: Final = raw_step.get("signature")
                step_sig: Final = step_sig_raw if isinstance(step_sig_raw, str) and step_sig_raw else None
                effective_sig: Final = step_sig or self._pending_thought_signature
                self._pending_thought_signature = None
                self._record_tool_call_sig_state(slot, raw_id, effective_sig)
                encoded_id: Final = encode_tool_call_id_with_signature(raw_id, effective_sig)
                tc_chunk: Final = ChatCompletionToolCallChunk(
                    index=slot,
                    id=encoded_id,
                    type="function",
                    function=ChatCompletionToolCallFunctionChunk(
                        name=fn_name,
                        arguments=_format_initial_arguments(raw_step.get("arguments")),
                    ),
                )
                return (self._build_chunk(Delta(role="assistant", tool_calls=[tc_chunk])),)
            case "function_result":
                self._active_thought_step = None
                self._pending_thought_signature = None
                self._segment_signed_tool_call = False
                return ()
            case "model_output":
                initial_text: Final = _text_from_blocks(raw_step.get("content"))
                if not initial_text:
                    return ()
                return (self._build_chunk(Delta(role="assistant", content=initial_text)),)
            case "text":
                direct_text: Final = str(raw_step.get("text") or "")
                return (self._build_chunk(Delta(role="assistant", content=direct_text)),) if direct_text else ()
            case _:
                return ()

    def _handle_step_delta(self, event: JsonMapping) -> tuple[ModelResponseStream, ...]:
        raw_index: Final = event.get("index")
        step_index: Final = raw_index if isinstance(raw_index, int) else 0
        raw_delta: Final = event.get("delta")
        if not isinstance(raw_delta, dict):
            return ()

        match raw_delta.get("type"):
            case "text":
                text_piece: Final = str(raw_delta.get("text") or "")
                if not text_piece:
                    return ()
                return (self._build_chunk(Delta(content=text_piece)),)
            case "thought_summary" | "thought":
                thought_piece: Final = _thought_summary_delta_text(raw_delta)
                if not thought_piece:
                    return ()
                summary_slot: Final = self._allocate_thought_slot(step_index)
                summary_block: Final = ChatCompletionThinkingBlock(
                    type="thinking",
                    thinking=thought_piece,
                    index=summary_slot,
                )
                return (
                    self._build_chunk(
                        Delta(
                            reasoning_content=thought_piece,
                            thinking_blocks=[summary_block],
                        )
                    ),
                )
            case "thought_signature" | "signature":
                sig_val: Final = raw_delta.get("signature")
                if not isinstance(sig_val, str) or not sig_val:
                    return ()
                sig_slot: Final = self._allocate_thought_slot(step_index)
                sig_block: Final = ChatCompletionThinkingBlock(
                    type="thinking",
                    thinking="",
                    signature=sig_val,
                    index=sig_slot,
                )
                sig_chunk: Final = self._build_chunk(Delta(thinking_blocks=[sig_block]))
                pending_target: Final = self._unsigned_tools.pop(step_index, None)
                if pending_target is not None:
                    target_slot, target_raw_id = pending_target
                    if self._active_thought_step == step_index:
                        self._segment_signed_tool_call = True
                        self._pending_thought_signature = None
                    backfill_tc: Final = ChatCompletionToolCallChunk(
                        index=target_slot,
                        id=encode_tool_call_id_with_signature(target_raw_id, sig_val),
                        type="function",
                        function=ChatCompletionToolCallFunctionChunk(name=None, arguments=""),
                    )
                    return (sig_chunk, self._build_chunk(Delta(tool_calls=[backfill_tc])))
                if self._active_thought_step == step_index or self._active_thought_step is None:
                    self._pending_thought_signature = sig_val
                return (sig_chunk,)
            case "arguments_delta" | "arguments" | "function_call":
                slot: Final = self._allocate_tool_slot(step_index)
                arg_piece: Final = _arguments_delta_piece(raw_delta)
                raw_name: Final = raw_delta.get("name")
                raw_id: Final = raw_delta.get("id")
                encoded_delta_id: Final = self._consume_delta_tool_call_id(slot, raw_id)
                delta_tc: Final = ChatCompletionToolCallChunk(
                    index=slot,
                    id=encoded_delta_id,
                    type="function",
                    function=ChatCompletionToolCallFunctionChunk(
                        name=raw_name if isinstance(raw_name, str) and raw_name else None,
                        arguments=arg_piece,
                    ),
                )
                return (self._build_chunk(Delta(tool_calls=[delta_tc])),)
            case _:
                return ()

    def _raise_stream_failure(
        self,
        event: JsonMapping,
        interaction_map: JsonMapping = _EMPTY_JSON_MAP,
        *,
        default_code: str | None = None,
    ) -> NoReturn:
        code_str, message = _extract_stream_error_code_and_message(
            event,
            interaction_map,
            default_code=default_code,
        )
        self._done = True
        if self._async_lines is None:
            self.close()
        raw_req: Final[object] = (
            getattr(self.http_response, "_request", None) if self.http_response is not None else None
        )
        effective_req: Final = (
            raw_req
            if isinstance(raw_req, httpx.Request)
            else httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/interactions")
        )
        raise _build_stream_exception(
            code_str=code_str,
            message=message,
            model=self._model,
            llm_provider=self._custom_llm_provider,
            request=effective_req,
        )

    def _handle_completed(self, event: JsonMapping) -> tuple[ModelResponseStream, ...]:
        raw_interaction: Final = event.get("interaction")
        interaction_map: Final = raw_interaction if isinstance(raw_interaction, dict) else _EMPTY_JSON_MAP
        maybe_id: Final = interaction_map.get("id") or event.get("id")
        if isinstance(maybe_id, str) and maybe_id:
            self._stream_id = maybe_id

        raw_status: Final = interaction_map.get("status") or event.get("status")
        if (
            raw_status in ("failed", "cancelled")
            or event.get("error") is not None
            or interaction_map.get("error") is not None
        ):
            self._raise_stream_failure(
                event,
                interaction_map,
                default_code=raw_status if isinstance(raw_status, str) else None,
            )
        finish_reason: Final = map_interactions_status_to_finish_reason(
            raw_status if isinstance(raw_status, str) else None,
            has_tool_calls=bool(self._tool_slots_by_step),
        )
        raw_usage: Final = interaction_map.get("usage") or event.get("usage")
        usage: Final = transform_interactions_usage(raw_usage) if raw_usage is not None else None
        return (self._build_chunk(Delta(), finish_reason=finish_reason, usage=usage),)

    def feed_line(self, raw_line: str | bytes) -> None:
        payload_str: Final = _extract_sse_payload(raw_line)
        if payload_str is None:
            return
        if payload_str == STREAM_SSE_DONE_STRING:
            self._done = True
            return

        event: Final = _parse_event_json(payload_str)
        if event is None:
            return

        event_type: Final = event.get("event_type") or event.get("type")
        match event_type:
            case "interaction.created" | "interaction.start":
                raw_inter: Final = event.get("interaction")
                created_id: Final = raw_inter.get("id") if isinstance(raw_inter, dict) else event.get("id")
                if isinstance(created_id, str) and created_id:
                    self._stream_id = created_id
            case "step.start" | "content.start":
                self._buffer.extend(self._handle_step_start(event))
            case "step.delta" | "content.delta":
                self._buffer.extend(self._handle_step_delta(event))
            case "interaction.completed" | "interaction.complete":
                self._buffer.extend(self._handle_completed(event))
            case "error" | "interaction.failed" | "interaction.cancelled":
                failed_inter: Final = event.get("interaction")
                self._raise_stream_failure(
                    event,
                    failed_inter if isinstance(failed_inter, dict) else _EMPTY_JSON_MAP,
                    default_code="cancelled" if event_type == "interaction.cancelled" else None,
                )
            case "interaction.status_update":
                status_inter: Final = event.get("interaction")
                status_map: Final = status_inter if isinstance(status_inter, dict) else _EMPTY_JSON_MAP
                status_val: Final = status_map.get("status") or event.get("status")
                if (
                    status_val in ("failed", "cancelled")
                    or event.get("error") is not None
                    or status_map.get("error") is not None
                ):
                    self._raise_stream_failure(
                        event,
                        status_map,
                        default_code=status_val if isinstance(status_val, str) else None,
                    )
            case _:
                pass

    def __iter__(self) -> InteractionsChatStreamingIterator:
        return self

    def __next__(self) -> ModelResponseStream:
        while not self._buffer:
            if self._done or self._sync_lines is None:
                self.close()
                raise StopIteration
            try:
                line = next(self._sync_lines)
            except StopIteration:
                self._done = True
                self.close()
                raise
            self.feed_line(line)
        return self._buffer.popleft()

    def __aiter__(self) -> InteractionsChatStreamingIterator:
        return self

    async def __anext__(self) -> ModelResponseStream:
        while not self._buffer:
            if self._done or self._async_lines is None:
                await self.aclose()
                raise StopAsyncIteration
            try:
                line = await anext(self._async_lines)
            except StopAsyncIteration:
                self._done = True
                await self.aclose()
                raise
            try:
                self.feed_line(line)
            except Exception:
                await self.aclose()
                raise
        return self._buffer.popleft()
