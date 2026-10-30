from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, chain
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, JsonValue, TypeAdapter

from litellm.litellm_core_utils.llm_cost_calc.usage_object_transformation import (
    InteractionsUsageObjectTransformation,
)
from litellm.litellm_core_utils.prompt_templates.factory import encode_tool_call_id_with_signature
from litellm.types.llms.openai import (
    ChatCompletionAnnotation,
    ChatCompletionAnnotationURLCitation,
    ChatCompletionThinkingBlock,
    ImageURLListItem,
    ImageURLObject,
)
from litellm.types.utils import (
    ChatCompletionAudioResponse,
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
    Usage,
)

JsonMapping: TypeAlias = Mapping[str, JsonValue]
FinishReason: TypeAlias = Literal["stop", "length", "tool_calls", "content_filter"]

_JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_EMPTY_JSON_MAP: Final[JsonMapping] = MappingProxyType({})
_AUDIO_EXPIRY_OFFSET_SECONDS: Final = 3600


@dataclass(frozen=True, slots=True)
class _ParsedOutputBlocks:
    text: str
    annotations: tuple[ChatCompletionAnnotation, ...]
    images: tuple[ImageURLListItem, ...]
    audio_data: str | None


@dataclass(frozen=True, slots=True)
class _ExtractedSteps:
    text: str
    reasoning_text: str
    thinking_blocks: tuple[ChatCompletionThinkingBlock, ...]
    tool_calls: tuple[ChatCompletionMessageToolCall, ...]
    annotations: tuple[ChatCompletionAnnotation, ...]
    images: tuple[ImageURLListItem, ...]
    audio_data: str | None


def _to_json_mapping(raw_payload: object) -> JsonMapping:
    normalized: Final = (
        _JSON_VALUE.validate_python(raw_payload.model_dump(exclude_none=True))
        if isinstance(raw_payload, BaseModel)
        else _JSON_VALUE.validate_python(raw_payload)
    )
    return normalized if isinstance(normalized, dict) else _EMPTY_JSON_MAP


def _summary_text(summary: JsonValue) -> str:
    if isinstance(summary, str):
        return summary
    if not isinstance(summary, list):
        return ""
    parts: Final = tuple(
        str(block.get("text") or "") for block in summary if isinstance(block, dict) and block.get("type") == "text"
    )
    return "".join(parts)


def _convert_annotation(raw_anno: JsonMapping, char_offset: int) -> ChatCompletionAnnotation | None:
    url_val: Final = raw_anno.get("url") or raw_anno.get("uri") or raw_anno.get("source")
    if not isinstance(url_val, str) or not url_val:
        return None
    raw_start: Final = raw_anno.get("start_index")
    raw_end: Final = raw_anno.get("end_index")
    raw_title: Final = raw_anno.get("title")
    start_idx: Final = (raw_start + char_offset) if isinstance(raw_start, int) else char_offset
    end_idx: Final = (raw_end + char_offset) if isinstance(raw_end, int) else start_idx
    citation: Final = ChatCompletionAnnotationURLCitation(
        url=url_val,
        title=raw_title if isinstance(raw_title, str) else url_val,
        start_index=start_idx,
        end_index=end_idx,
    )
    return ChatCompletionAnnotation(type="url_citation", url_citation=citation)


def _annotations_for_block(raw_annotations: JsonValue, char_offset: int) -> tuple[ChatCompletionAnnotation, ...]:
    if not isinstance(raw_annotations, list):
        return ()
    converted: Final = tuple(
        _convert_annotation(anno, char_offset) for anno in raw_annotations if isinstance(anno, dict)
    )
    return tuple(item for item in converted if item is not None)


def _image_item_from_block(img_idx: int, block: JsonMapping) -> ImageURLListItem | None:
    raw_mime: Final = block.get("mime_type")
    mime_str: Final = str(raw_mime) if raw_mime else "image/png"
    raw_data: Final = block.get("data")
    if isinstance(raw_data, str) and raw_data:
        return ImageURLListItem(
            index=img_idx,
            type="image_url",
            image_url=ImageURLObject(url=f"data:{mime_str};base64,{raw_data}"),
        )
    raw_uri: Final = block.get("uri")
    if isinstance(raw_uri, str) and raw_uri:
        return ImageURLListItem(
            index=img_idx,
            type="image_url",
            image_url=ImageURLObject(url=raw_uri),
        )
    return None


def _parse_output_content_blocks(blocks: Sequence[JsonValue]) -> _ParsedOutputBlocks:
    dict_blocks: Final = tuple(block for block in blocks if isinstance(block, dict))
    text_blocks: Final = tuple(block for block in dict_blocks if block.get("type") == "text")
    pieces: Final = tuple(str(block.get("text") or "") for block in text_blocks)
    offsets: Final = (0, *accumulate(len(piece) for piece in pieces))
    annotations: Final = tuple(
        chain.from_iterable(
            _annotations_for_block(block.get("annotations"), offset) for block, offset in zip(text_blocks, offsets)
        )
    )
    image_blocks: Final = tuple(block for block in dict_blocks if block.get("type") == "image")
    raw_images: Final = tuple(_image_item_from_block(idx, block) for idx, block in enumerate(image_blocks))
    images: Final = tuple(img for img in raw_images if img is not None)
    audio_chunks: Final = tuple(
        str(block["data"])
        for block in dict_blocks
        if block.get("type") == "audio" and isinstance(block.get("data"), str) and block.get("data")
    )
    return _ParsedOutputBlocks(
        text="".join(pieces),
        annotations=annotations,
        images=images,
        audio_data="".join(audio_chunks) if audio_chunks else None,
    )


def _normalize_legacy_outputs_to_steps(raw_outputs: JsonValue) -> tuple[JsonValue, ...]:
    if not isinstance(raw_outputs, list):
        return ()
    return tuple(
        {"type": item["type"], **item}
        if isinstance(item, dict) and item.get("type") in ("thought", "function_call")
        else ({"type": "model_output", "content": [item]} if isinstance(item, dict) else item)
        for item in raw_outputs
    )


def _serialize_arguments(raw_args: JsonValue) -> str:
    if isinstance(raw_args, str):
        return raw_args
    if isinstance(raw_args, dict):
        return json.dumps(raw_args)
    return "{}"


@dataclass(frozen=True, slots=True)
class _SigScanState:
    pending_thought_sig: str | None = None
    call_sig: str | None = None


def _advance_sig_scan(state: _SigScanState, step: _SigScanState | JsonMapping) -> _SigScanState:
    if isinstance(step, _SigScanState):
        return step
    match step.get("type"):
        case "thought":
            sig_val: Final = step.get("signature")
            next_pending: Final = sig_val if isinstance(sig_val, str) and sig_val else None
            return _SigScanState(pending_thought_sig=next_pending, call_sig=None)
        case "function_call":
            step_sig_raw: Final = step.get("signature")
            step_sig: Final = step_sig_raw if isinstance(step_sig_raw, str) and step_sig_raw else None
            effective: Final = step_sig or state.pending_thought_sig
            return _SigScanState(pending_thought_sig=None, call_sig=effective)
        case "function_result":
            return _SigScanState(pending_thought_sig=None, call_sig=None)
        case _:
            return _SigScanState(pending_thought_sig=state.pending_thought_sig, call_sig=None)


def _thinking_block_from_step(step: JsonMapping) -> ChatCompletionThinkingBlock | None:
    sig_val: Final = step.get("signature")
    sig_str: Final = sig_val if isinstance(sig_val, str) and sig_val else None
    summary: Final = _summary_text(step.get("summary"))
    if not summary and sig_str is None:
        return None
    return ChatCompletionThinkingBlock(
        type="thinking",
        thinking=summary,
        signature=sig_str,
    )


def _tool_call_from_step(
    tool_idx: int,
    step: JsonMapping,
    effective_sig: str | None,
) -> ChatCompletionMessageToolCall:
    raw_id: Final = str(step.get("id") or f"call_{tool_idx}")
    fn_name: Final = str(step.get("name") or "")
    encoded_id: Final = encode_tool_call_id_with_signature(raw_id, effective_sig)
    fn_obj: Final = Function(
        name=fn_name,
        arguments=_serialize_arguments(step.get("arguments")),
    )
    return ChatCompletionMessageToolCall(
        id=encoded_id,
        type="function",
        function=fn_obj,
        provider_specific_fields={"thought_signature": effective_sig} if effective_sig else None,
    )


def _output_blocks_from_step(step: JsonMapping) -> _ParsedOutputBlocks | None:
    raw_content: Final = step.get("content")
    return _parse_output_content_blocks(raw_content) if isinstance(raw_content, list) else None


def _extract_steps(steps: Sequence[JsonValue]) -> _ExtractedSteps:
    dict_steps: Final = tuple(step for step in steps if isinstance(step, dict))
    seeded_steps: Final[tuple[_SigScanState | JsonMapping, ...]] = (_SigScanState(), *dict_steps)
    scan_states: Final = tuple(accumulate(seeded_steps, _advance_sig_scan))

    thought_steps: Final = tuple(step for step in dict_steps if step.get("type") == "thought")
    reasoning_parts: Final = tuple(_summary_text(step.get("summary")) for step in thought_steps)
    raw_thinking_blocks: Final = tuple(_thinking_block_from_step(step) for step in thought_steps)
    thinking_blocks: Final = tuple(b for b in raw_thinking_blocks if b is not None)

    fc_with_sigs: Final = tuple(
        (step, state.call_sig)
        for step, state in zip(dict_steps, scan_states[1:])
        if step.get("type") == "function_call"
    )
    tool_calls: Final = tuple(
        _tool_call_from_step(idx, step, call_sig) for idx, (step, call_sig) in enumerate(fc_with_sigs)
    )

    raw_outputs: Final = tuple(
        _output_blocks_from_step(step) for step in dict_steps if step.get("type") == "model_output"
    )
    output_parsed: Final = tuple(p for p in raw_outputs if p is not None)
    audio_chunks: Final = tuple(p.audio_data for p in output_parsed if p.audio_data is not None)

    return _ExtractedSteps(
        text="".join(p.text for p in output_parsed),
        reasoning_text="".join(p for p in reasoning_parts if p),
        thinking_blocks=thinking_blocks,
        tool_calls=tool_calls,
        annotations=tuple(chain.from_iterable(p.annotations for p in output_parsed)),
        images=tuple(chain.from_iterable(p.images for p in output_parsed)),
        audio_data="".join(audio_chunks) if audio_chunks else None,
    )


def map_interactions_status_to_finish_reason(
    status: str | None,
    *,
    has_tool_calls: bool,
) -> FinishReason:
    if has_tool_calls:
        return "tool_calls"
    match status:
        case "incomplete" | "budget_exceeded" | "requires_continuation":
            return "length"
        case _:
            return "stop"


def transform_interactions_usage(raw_usage: object) -> Usage:
    usage_map: Final = _to_json_mapping(raw_usage) if raw_usage is not None else _EMPTY_JSON_MAP
    plain_dict: Final = dict(usage_map)
    if plain_dict and InteractionsUsageObjectTransformation.is_interactions_usage_object(plain_dict):
        return InteractionsUsageObjectTransformation.transform_interactions_usage_object(plain_dict)
    prompt_tokens: Final = plain_dict.get("prompt_tokens")
    completion_tokens: Final = plain_dict.get("completion_tokens")
    total_tokens: Final = plain_dict.get("total_tokens")
    p_tok: Final = prompt_tokens if isinstance(prompt_tokens, int) else 0
    c_tok: Final = completion_tokens if isinstance(completion_tokens, int) else 0
    t_tok: Final = total_tokens if isinstance(total_tokens, int) else (p_tok + c_tok)
    return Usage(prompt_tokens=p_tok, completion_tokens=c_tok, total_tokens=t_tok)


def transform_interactions_response_to_chat_response(
    *,
    raw_response: object,
    model: str,
    model_response: ModelResponse,
) -> ModelResponse:
    payload: Final = _to_json_mapping(raw_response)
    raw_steps: Final = payload.get("steps")
    steps_seq: Final = (
        tuple(raw_steps)
        if isinstance(raw_steps, list) and raw_steps
        else _normalize_legacy_outputs_to_steps(payload.get("outputs"))
    )
    extracted: Final = _extract_steps(steps_seq)

    response_id_raw: Final = payload.get("id")
    effective_id: Final = response_id_raw if isinstance(response_id_raw, str) and response_id_raw else model_response.id

    msg: Final = Message(
        role="assistant",
        content=extracted.text if (extracted.text or not extracted.tool_calls) else None,
        tool_calls=list(extracted.tool_calls) if extracted.tool_calls else None,
        reasoning_content=extracted.reasoning_text or None,
        thinking_blocks=list(extracted.thinking_blocks) if extracted.thinking_blocks else None,
        annotations=list(extracted.annotations) if extracted.annotations else None,
        images=list(extracted.images) if extracted.images else None,
        audio=(
            ChatCompletionAudioResponse(
                id=f"audio_{effective_id}",
                data=extracted.audio_data,
                expires_at=int(model_response.created) + _AUDIO_EXPIRY_OFFSET_SECONDS,
                transcript=extracted.text,
            )
            if extracted.audio_data is not None
            else None
        ),
    )

    raw_status: Final = payload.get("status")
    finish_reason: Final = map_interactions_status_to_finish_reason(
        raw_status if isinstance(raw_status, str) else None,
        has_tool_calls=bool(extracted.tool_calls),
    )

    return model_response.model_copy(
        update={
            "id": effective_id,
            "model": model,
            "choices": [Choices(finish_reason=finish_reason, index=0, message=msg)],
            "usage": transform_interactions_usage(payload.get("usage")),
        }
    )
