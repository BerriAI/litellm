from __future__ import annotations

import json
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Final, NoReturn, TypeAlias, TypedDict

import httpx
from openai import OpenAIError
from pydantic import JsonValue, TypeAdapter
from typing_extensions import ReadOnly

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
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.vertex_ai.gemini.long_decode import with_initial_prompt_tokens
from litellm.types.llms.openai import (
    ChatCompletionAnnotation,
    ChatCompletionAudioDelta,
    ChatCompletionThinkingBlock,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ImageURLListItem,
)
from litellm.types.utils import (
    Delta,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)

from .continuation import (
    client_continuation_token,
    initial_input_tokens_of,
    log_failed_hop,
    merge_hop_usage,
    next_hop_body,
    output_tokens_of,
    resends_input_on_resume,
)
from .grounding import GROUNDING_STEP_TYPES, VertexMetadata, vertex_metadata_of
from .response_transformation import (
    FinishReason,
    annotations_for_block,
    attach_vertex_metadata,
    map_interactions_status_to_finish_reason,
    parse_output_content_blocks,
    shifted,
    transform_interactions_usage,
)

JsonMapping: TypeAlias = Mapping[str, JsonValue]
LineStreamSync: TypeAlias = Iterator[str | bytes]
LineStreamAsync: TypeAlias = AsyncIterator[str | bytes]
HopBody: TypeAlias = dict[str, object]
SyncHopOpener: TypeAlias = Callable[[HopBody], httpx.Response]
AsyncHopOpener: TypeAlias = Callable[[HopBody], Awaitable[httpx.Response]]
_HOP_FAILURES: Final = (BaseLLMException, httpx.HTTPError, ValueError, OpenAIError)

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


class _MediaDelta(TypedDict, total=False):
    images: ReadOnly[list[ImageURLListItem]]
    audio: ReadOnly[ChatCompletionAudioDelta]


_NO_MEDIA: Final = _MediaDelta()


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


def extract_stream_error_code_and_message(
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


def build_stream_exception(
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


_NO_STEP: Final[JsonMapping] = MappingProxyType({})


def _merged_leaf(known: JsonValue | None, update: JsonValue) -> JsonValue:
    if isinstance(known, list) and isinstance(update, list):
        return list({json.dumps(item, sort_keys=True): item for item in (*known, *update)}.values())
    return update if update or known is None else known


def _merged_value(known: JsonValue | None, update: JsonValue) -> JsonValue:
    if isinstance(known, dict) and isinstance(update, dict):
        return {**known, **{key: _merged_leaf(known.get(key), value) for key, value in update.items()}}
    return _merged_leaf(known, update)


def _merged_step(known: JsonMapping, update: JsonMapping) -> JsonMapping:
    return {**known, **{key: _merged_value(known.get(key), value) for key, value in update.items()}}


class InteractionsChatStreamingIterator:
    def __init__(
        self,
        *,
        model: str,
        custom_llm_provider: str = "gemini",
        sync_lines: LineStreamSync | None = None,
        async_lines: LineStreamAsync | None = None,
        http_response: httpx.Response | None = None,
        request_body: Mapping[str, object] | None = None,
        open_sync_hop: SyncHopOpener | None = None,
        open_async_hop: AsyncHopOpener | None = None,
    ) -> None:
        self._model: Final = model
        self._custom_llm_provider: Final = custom_llm_provider
        self._sync_lines: LineStreamSync | None = sync_lines
        self._async_lines: LineStreamAsync | None = async_lines
        self._request_body: Final = request_body
        self._open_sync_hop: Final = open_sync_hop
        self._open_async_hop: Final = open_async_hop
        self._pending_hop_body: HopBody | None = None
        self._pending_hop_token: str | None = None
        self._hop_usages: tuple[JsonValue, ...] = ()
        self._hop_open: bool = False
        self._hop_delivered: bool = False
        self._deferred_failure: Exception | None = None
        self._images_sent: int = 0
        self._step_index_offset: int = 0
        self._highest_step_index: int = -1
        self.http_response: httpx.Response | None = http_response
        self._stream_id: str = f"chatcmpl-{uuid.uuid4()}"
        self._active_thought_step: int | None = None
        self._pending_thought_signature: str | None = None
        self._segment_signed_tool_call: bool = False
        self._unsigned_tools: Final[dict[int, tuple[int, str]]] = {}  # mutable-ok: thought-step -> unsigned tool slot
        self._text_pieces: Final[list[str]] = []  # mutable-ok: per-stream answer text, joined only for citations
        self._text_length: int = 0
        self._grounding_steps: Final[dict[int, JsonMapping]] = {}  # mutable-ok: per-stream grounding steps by index
        self._cited_spans: Final[list[JsonValue]] = []  # mutable-ok: per-stream citations in joined-answer bytes
        self._last_text_start: int = 0
        self._text_starts: Final[dict[int, int]] = {}  # mutable-ok: per-stream step-index -> offset of its first text
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

    def _step_index_of(self, event: JsonMapping) -> int:
        raw_index: Final = event.get("index")
        step_index: Final = (raw_index if isinstance(raw_index, int) else 0) + self._step_index_offset
        self._highest_step_index = max(self._highest_step_index, step_index)
        return step_index

    def _handle_step_start(self, event: JsonMapping) -> tuple[ModelResponseStream, ...]:
        step_index: Final = self._step_index_of(event)
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
                initial_media: Final = self._media_delta(raw_step.get("content"))
                if not initial_text and initial_media is None:
                    return ()
                self._count_text(step_index, initial_text)
                return (
                    self._build_chunk(
                        Delta(role="assistant", content=initial_text or None, **(initial_media or _NO_MEDIA))
                    ),
                    *self._cited_chunks(self._step_start_annotations(step_index, raw_step.get("content"))),
                )
            case "text":
                direct_text: Final = str(raw_step.get("text") or "")
                self._count_text(step_index, direct_text)
                return (self._build_chunk(Delta(role="assistant", content=direct_text)),) if direct_text else ()
            case step_type if step_type in GROUNDING_STEP_TYPES and ("arguments" in raw_step or "result" in raw_step):
                return self._record_grounding(step_index, raw_step)
            case _:
                return ()

    def _count_text(self, step_index: int, text: str) -> None:
        self._last_text_start = self._text_starts.setdefault(step_index, self._text_length)
        self._text_pieces.append(text)
        self._text_length += len(text)

    def _text_split_at(self, position: int) -> tuple[str, str]:
        text: Final = "".join(self._text_pieces)
        return text[:position], text[position:]

    def _remember_cited(self, raw_annotations: JsonValue, byte_offset: int) -> None:
        if not isinstance(raw_annotations, list):
            return
        self._cited_spans.extend(
            {**anno, "start_index": start + byte_offset, "end_index": end + byte_offset}
            for anno in raw_annotations
            if isinstance(anno, dict)
            and isinstance(start := anno.get("start_index"), int)
            and isinstance(end := anno.get("end_index"), int)
        )

    def _vertex_metadata(self) -> VertexMetadata:
        answer: Final[JsonValue] = {
            "type": "model_output",
            "content": [{"type": "text", "text": "".join(self._text_pieces), "annotations": list(self._cited_spans)}],
        }
        return vertex_metadata_of((*(dict(step) for step in self._grounding_steps.values()), answer))

    def _delta_annotations(self, step_index: int, raw_delta: JsonMapping) -> tuple[ChatCompletionAnnotation, ...]:
        if raw_delta.get("annotations") is None:
            return ()
        cited_from: Final = self._text_starts.get(step_index, self._last_text_start)
        before, cited_text = self._text_split_at(cited_from)
        self._remember_cited(raw_delta.get("annotations"), len(before.encode()))
        return annotations_for_block(raw_delta.get("annotations"), cited_from, cited_text)

    def _step_start_annotations(self, step_index: int, raw_blocks: JsonValue) -> tuple[ChatCompletionAnnotation, ...]:
        if not isinstance(raw_blocks, list):
            return ()
        cited_from: Final = self._text_starts.get(step_index, self._last_text_start)
        text_blocks: Final = tuple(
            block for block in raw_blocks if isinstance(block, dict) and block.get("type") == "text"
        )
        if not any(block.get("annotations") for block in text_blocks):
            return ()
        lengths: Final = tuple(len(str(block.get("text") or "").encode()) for block in text_blocks)
        step_offset: Final = len(self._text_split_at(cited_from)[0].encode())
        for position, block in enumerate(text_blocks):
            self._remember_cited(block.get("annotations"), step_offset + sum(lengths[:position]))
        return tuple(shifted(cited, cited_from) for cited in parse_output_content_blocks(raw_blocks).annotations)

    def _cited_chunks(self, cited: tuple[ChatCompletionAnnotation, ...]) -> tuple[ModelResponseStream, ...]:
        return (self._build_chunk(Delta(annotations=list(cited))),) if cited else ()

    def _media_delta(self, raw_blocks: JsonValue) -> _MediaDelta | None:
        if not isinstance(raw_blocks, list):
            return None
        parsed: Final = parse_output_content_blocks(raw_blocks)
        images: Final = tuple(
            ImageURLListItem(image_url=image["image_url"], type=image["type"], index=self._images_sent + offset)
            for offset, image in enumerate(parsed.images)
        )
        self._images_sent += len(images)
        audio: Final = ChatCompletionAudioDelta(data=parsed.audio_data) if parsed.audio_data is not None else None
        if images and audio is not None:
            return _MediaDelta(images=list(images), audio=audio)
        if images:
            return _MediaDelta(images=list(images))
        return _MediaDelta(audio=audio) if audio is not None else None

    def _record_grounding(self, step_index: int, raw_step: JsonMapping) -> tuple[ModelResponseStream, ...]:
        self._grounding_steps[step_index] = _merged_step(self._grounding_steps.get(step_index, _NO_STEP), raw_step)
        return ()

    def _handle_step_delta(self, event: JsonMapping) -> tuple[ModelResponseStream, ...]:
        step_index: Final = self._step_index_of(event)
        raw_delta: Final = event.get("delta")
        if not isinstance(raw_delta, dict):
            return ()

        if raw_delta.get("type") in GROUNDING_STEP_TYPES:
            return self._record_grounding(step_index, raw_delta)

        match raw_delta.get("type"):
            case "text":
                text_piece: Final = str(raw_delta.get("text") or "")
                if not text_piece:
                    return ()
                self._count_text(step_index, text_piece)
                return (
                    self._build_chunk(Delta(content=text_piece)),
                    *self._cited_chunks(self._delta_annotations(step_index, raw_delta)),
                )
            case "text_annotation_delta" | "annotations":
                return self._cited_chunks(self._delta_annotations(step_index, raw_delta))
            case "image" | "audio":
                media: Final = self._media_delta([dict(raw_delta)])
                return (self._build_chunk(Delta(**media)),) if media is not None else ()
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
        code_str, message = extract_stream_error_code_and_message(
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
        raise build_stream_exception(
            code_str=code_str,
            message=message,
            model=self._model,
            llm_provider=self._custom_llm_provider,
            request=effective_req,
        )

    def _handle_completed(self, event: JsonMapping) -> tuple[ModelResponseStream, ...]:
        raw_interaction: Final = event.get("interaction")
        interaction_map: Final = raw_interaction if isinstance(raw_interaction, dict) else _EMPTY_JSON_MAP
        if not self._hop_delivered:
            self._adopt_stream_id(event)

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
        self._hop_open = False
        hop_usage: Final = interaction_map.get("usage") or event.get("usage")
        self._hop_usages = (*self._hop_usages, hop_usage)
        if self._request_body is not None and (self._open_sync_hop or self._open_async_hop) is not None:
            hop_body: Final = next_hop_body(
                self._request_body,
                interaction_map,
                sum(output_tokens_of({"usage": usage}) for usage in self._hop_usages[:-1]),
                resends_input=resends_input_on_resume(self._custom_llm_provider),
            )
            if hop_body is not None:
                self._pending_hop_body = hop_body
                self._pending_hop_token = str(hop_body["continuation_token"])
                return ()
        return (self._final_chunk(finish_reason, client_continuation_token(interaction_map)),)

    def _final_chunk(self, finish_reason: FinishReason, continuation_token: str | None) -> ModelResponseStream:
        raw_usage: Final = merge_hop_usage(self._hop_usages)
        final_chunk: Final = self._build_chunk(
            Delta(provider_specific_fields={"continuation_token": continuation_token})
            if continuation_token is not None
            else Delta(),
            finish_reason=finish_reason,
            usage=with_initial_prompt_tokens(
                transform_interactions_usage(raw_usage),
                initial_input_tokens_of(tuple({"usage": usage} for usage in self._hop_usages)),
            )
            if raw_usage is not None
            else None,
        )
        attach_vertex_metadata(final_chunk, self._vertex_metadata(), keeps_empty=False)
        return final_chunk

    def _take_pending_hop(self) -> HopBody | None:
        hop_body: Final = self._pending_hop_body
        self._pending_hop_body = None
        if hop_body is not None:
            self._done = False
            self._hop_open = True
            self._hop_delivered = False
            self._step_index_offset = self._highest_step_index + 1
        return hop_body

    def _end_after_failed_hop(self, error: Exception) -> None:
        log_failed_hop(error)
        self._done = True
        self._hop_open = False
        self._buffer.append(self._final_chunk("length", self._pending_hop_token))

    def _end_stream(self) -> None:
        if self._hop_open:
            self._fail_stream(ValueError("the continuation stream ended before the interaction completed"))
        self._done = True

    def _deliver(self, chunks: tuple[ModelResponseStream, ...]) -> None:
        self._hop_delivered = self._hop_delivered or bool(chunks)
        self._buffer.extend(chunks)

    def _fail_stream(self, error: Exception) -> None:
        if self._hop_open and not self._hop_delivered:
            self._end_after_failed_hop(error)
            return
        raw_usage: Final = merge_hop_usage(self._hop_usages)
        if raw_usage is None or self._deferred_failure is not None:
            raise error
        self._deferred_failure = error
        self._buffer.append(
            self._build_chunk(Delta(), finish_reason=None, usage=transform_interactions_usage(raw_usage))
        )

    def _raise_deferred_failure(self) -> None:
        if self._deferred_failure is not None and not self._buffer:
            raise self._deferred_failure

    def _start_sync_hop(self, hop_body: HopBody) -> None:
        if self._open_sync_hop is None:
            return
        try:
            response: Final = self._open_sync_hop(hop_body)
        except _HOP_FAILURES as error:
            self._end_after_failed_hop(error)
            return
        self.close()
        self.http_response = response
        self._sync_lines = response.iter_lines()

    async def _start_async_hop(self, hop_body: HopBody) -> None:
        if self._open_async_hop is None:
            return
        try:
            response: Final = await self._open_async_hop(hop_body)
        except _HOP_FAILURES as error:
            self._end_after_failed_hop(error)
            return
        await self.aclose()
        self.http_response = response
        self._async_lines = response.aiter_lines()

    def _adopt_stream_id(self, event: JsonMapping) -> None:
        raw_inter: Final = event.get("interaction")
        created_id: Final = (raw_inter.get("id") if isinstance(raw_inter, dict) else None) or event.get("id")
        if isinstance(created_id, str) and created_id and not self._hop_usages:
            self._stream_id = created_id

    def _raise_on_failed_status_update(self, event: JsonMapping) -> None:
        status_inter: Final = event.get("interaction")
        status_map: Final = status_inter if isinstance(status_inter, dict) else _EMPTY_JSON_MAP
        status_val: Final = status_map.get("status") or event.get("status")
        has_error: Final = event.get("error") is not None or status_map.get("error") is not None
        if status_val in ("failed", "cancelled") or has_error:
            self._raise_stream_failure(
                event,
                status_map,
                default_code=status_val if isinstance(status_val, str) else None,
            )

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
                self._adopt_stream_id(event)
            case "step.start" | "content.start":
                self._deliver(self._handle_step_start(event))
            case "step.delta" | "content.delta":
                self._deliver(self._handle_step_delta(event))
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
                self._raise_on_failed_status_update(event)
            case _:
                pass

    def __iter__(self) -> InteractionsChatStreamingIterator:
        return self

    def __next__(self) -> ModelResponseStream:
        while not self._buffer:
            self._raise_deferred_failure()
            if (hop_body := self._take_pending_hop()) is not None:
                self._start_sync_hop(hop_body)
                continue
            if self._done or self._sync_lines is None:
                self.close()
                raise StopIteration
            try:
                line = next(self._sync_lines)
            except StopIteration:
                self._end_stream()
                continue
            except _HOP_FAILURES as error:
                self._fail_stream(error)
                continue
            try:
                self.feed_line(line)
            except _HOP_FAILURES as error:
                self._fail_stream(error)
        return self._buffer.popleft()

    def __aiter__(self) -> InteractionsChatStreamingIterator:
        return self

    async def __anext__(self) -> ModelResponseStream:
        while not self._buffer:
            self._raise_deferred_failure()
            if (hop_body := self._take_pending_hop()) is not None:
                await self._start_async_hop(hop_body)
                continue
            if self._done or self._async_lines is None:
                await self.aclose()
                raise StopAsyncIteration
            try:
                line = await anext(self._async_lines)
            except StopAsyncIteration:
                self._end_stream()
                continue
            except _HOP_FAILURES as error:
                await self.aclose()
                self._fail_stream(error)
                continue
            try:
                self.feed_line(line)
            except _HOP_FAILURES as error:
                await self.aclose()
                self._fail_stream(error)
            except Exception:
                await self.aclose()
                raise
        return self._buffer.popleft()
