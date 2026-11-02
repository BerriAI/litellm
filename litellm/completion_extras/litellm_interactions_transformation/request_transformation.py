from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import accumulate, chain
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.allowed_tools_tool_choice import (
    parse_chat_allowed_tools_tool_choice,
    responses_allowed_tools_tool_choice,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    THOUGHT_SIGNATURE_SEPARATOR,
    convert_to_anthropic_image_obj,
    get_thought_signature_from_tool,
)
from litellm.litellm_core_utils.prompt_templates.image_handling import convert_url_to_base64
from litellm.llms.vertex_ai.gemini.transformation import GEMINI_FILES_API_URI_PREFIX
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    OPENAI_TO_GEMINI_TTS_VOICES,
    VertexGeminiConfig,
)

MediaFetcher: TypeAlias = Callable[[str, str | None], tuple[str, str]]
JsonMapping: TypeAlias = Mapping[str, JsonValue]
JsonSequence: TypeAlias = Sequence[JsonValue]

_JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_JSON_MAPPING: Final[TypeAdapter[JsonMapping]] = TypeAdapter(JsonMapping)
_OBJECT_MAP: Final[TypeAdapter[Mapping[object, object]]] = TypeAdapter(Mapping[object, object])
_OBJECT_SEQ: Final[TypeAdapter[Sequence[object]]] = TypeAdapter(Sequence[object])
_EMPTY_JSON_MAP: Final[JsonMapping] = MappingProxyType({})
_DUMMY_THOUGHT_SIGNATURE: Final = base64.b64encode(b"skip_thought_signature_validator").decode("ascii")
_MAX_JSON_NORMALIZE_DEPTH: Final = 64

_AUDIO_MIME_BY_FORMAT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "wav": "audio/wav",
        "mp3": "audio/mp3",
        "aiff": "audio/aiff",
        "aac": "audio/aac",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
        "pcm16": "audio/pcm",
    }
)

_BUILTIN_TOOL_TYPES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "google_search": "google_search",
        "googleSearch": "google_search",
        "google_search_retrieval": "google_search",
        "googleSearchRetrieval": "google_search",
        "google_maps": "google_maps",
        "googleMaps": "google_maps",
        "url_context": "url_context",
        "urlContext": "url_context",
        "code_execution": "code_execution",
        "codeExecution": "code_execution",
    }
)


@dataclass(frozen=True, slots=True)
class InteractionsPreparedRequest:
    input_steps: JsonSequence
    optional_params: JsonMapping


def _default_media_fetcher(url: str, format_hint: str | None) -> tuple[str, str]:
    image_obj: Final = _JSON_MAPPING.validate_python(convert_to_anthropic_image_obj(url, format=format_hint))
    source: Final = image_obj.get("source")
    if isinstance(source, dict):
        media_type: Final = str(source.get("media_type") or format_hint or "image/jpeg")
        data: Final = str(source.get("data") or "")
        return media_type, data
    return format_hint or "application/octet-stream", convert_url_to_base64(url)


@dataclass(frozen=True, slots=True)
class _VisitFrame:
    value: object
    depth: int


@dataclass(frozen=True, slots=True)
class _AssembleMapFrame:
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _AssembleListFrame:
    count: int


_NormFrame: TypeAlias = _VisitFrame | _AssembleMapFrame | _AssembleListFrame


def _normalize_to_json(raw: object) -> JsonValue:
    work: Final[list[_NormFrame]] = [_VisitFrame(raw, 0)]  # mutable-ok: iterative normalization worklist stack
    results: Final[list[JsonValue]] = []  # mutable-ok: iterative normalization result stack
    while work:
        frame = work.pop()
        match frame:
            case _AssembleMapFrame(keys=keys):
                popped_vals = tuple(results.pop() for _ in keys)
                results.append(dict(zip(keys, reversed(popped_vals))))
            case _AssembleListFrame(count=count):
                popped_items = tuple(results.pop() for _ in range(count))
                results.append(list(reversed(popped_items)))
            case _VisitFrame(value=val, depth=depth):
                if isinstance(val, type) and issubclass(val, BaseModel):
                    results.append(_JSON_VALUE.validate_python(val.model_json_schema()))
                    continue
                if isinstance(val, BaseModel):
                    if depth >= _MAX_JSON_NORMALIZE_DEPTH:
                        raise ValueError("Input exceeds maximum JSON nesting depth")
                    work.append(_VisitFrame(val.model_dump(exclude_none=True), depth + 1))
                    continue
                try:
                    results.append(_JSON_VALUE.validate_python(val))
                    continue
                except ValidationError:
                    pass
                if depth >= _MAX_JSON_NORMALIZE_DEPTH:
                    raise ValueError("Input exceeds maximum JSON nesting depth")
                try:
                    typed_map = _OBJECT_MAP.validate_python(val)
                    keys = tuple(str(k) for k in typed_map)
                    work.append(_AssembleMapFrame(keys))
                    for map_val in reversed(tuple(typed_map.values())):
                        work.append(_VisitFrame(map_val, depth + 1))
                except ValidationError:
                    typed_seq = _OBJECT_SEQ.validate_python(val)
                    work.append(_AssembleListFrame(len(typed_seq)))
                    for seq_item in reversed(typed_seq):
                        work.append(_VisitFrame(seq_item, depth + 1))
    return results[0]


def _strip_signature_suffix(raw_call_id: str) -> str:
    if THOUGHT_SIGNATURE_SEPARATOR in raw_call_id:
        prefix, _ = raw_call_id.split(THOUGHT_SIGNATURE_SEPARATOR, 1)
        return prefix
    return raw_call_id


def _reject_or_drop(
    model: str,
    message: str,
    drop_params: bool,
    *,
    custom_llm_provider: str,
) -> None:
    if litellm.drop_params or drop_params:
        return
    raise UnsupportedParamsError(
        message=(f"{message} To drop unsupported params, set `litellm.drop_params = True` or pass `drop_params=True`."),
        model=model,
        llm_provider=custom_llm_provider,
        status_code=400,
    )


def _validate_sampling_param(
    params: JsonMapping,
    sampling_key: str,
    model: str,
    drop_params: bool,
) -> None:
    if params.get(sampling_key) is not None:
        VertexGeminiConfig.reject_unsupported_sampling_param(model, sampling_key, drop_params)


def _validate_request_params(
    params: JsonMapping,
    model: str,
    drop_params: bool,
    *,
    custom_llm_provider: str,
) -> None:
    if params.get("cached_content") is not None or params.get("cachedContent") is not None:
        _reject_or_drop(
            model,
            "Explicit prompt caching ('cached_content') is not supported by the Gemini Interactions API.",
            drop_params,
            custom_llm_provider=custom_llm_provider,
        )
    if params.get("video_metadata") is not None:
        _reject_or_drop(
            model,
            "'video_metadata' is not supported by the Gemini Interactions API.",
            drop_params,
            custom_llm_provider=custom_llm_provider,
        )
    if params.get("logprobs") is True or params.get("top_logprobs") is not None:
        _reject_or_drop(
            model,
            "'logprobs' is not supported by the Gemini Interactions API.",
            drop_params,
            custom_llm_provider=custom_llm_provider,
        )
    candidate_count: Final = params.get("n")
    if isinstance(candidate_count, int) and not isinstance(candidate_count, bool) and candidate_count > 1:
        _reject_or_drop(
            model,
            "Multiple candidate completions ('n > 1') are not supported by the Gemini Interactions API.",
            drop_params,
            custom_llm_provider=custom_llm_provider,
        )
    freq_penalty: Final = params.get("frequency_penalty")
    pres_penalty: Final = params.get("presence_penalty")
    if (isinstance(freq_penalty, (int, float)) and freq_penalty != 0) or (
        isinstance(pres_penalty, (int, float)) and pres_penalty != 0
    ):
        _reject_or_drop(
            model,
            "Frequency and presence penalties are not supported by the Gemini Interactions API.",
            drop_params,
            custom_llm_provider=custom_llm_provider,
        )
    if not VertexGeminiConfig.supports_sampling_params(model):
        _validate_sampling_param(params, "temperature", model, drop_params)
        _validate_sampling_param(params, "top_p", model, drop_params)
    raw_choice: Final = params.get("tool_choice")
    if raw_choice is not None and _convert_tool_choice(raw_choice) is None:
        _reject_or_drop(
            model,
            f"Unsupported tool_choice={raw_choice!r} for the Gemini Interactions API.",
            drop_params,
            custom_llm_provider=custom_llm_provider,
        )


def _parse_data_uri(data_uri: str, fallback_mime: str) -> tuple[str, str]:
    header, b64_data = data_uri.split(",", 1)
    mime_part: Final = header.split(";", 1)[0].removeprefix("data:")
    return (mime_part or fallback_mime), b64_data


def _needs_remote_fetch(url: str, custom_llm_provider: str) -> bool:
    if custom_llm_provider != "gemini":
        return False
    if url.startswith(GEMINI_FILES_API_URI_PREFIX):
        return False
    return url.startswith(("http://", "https://"))


def _convert_image_part(
    part: JsonMapping,
    custom_llm_provider: str,
    media_fetcher: MediaFetcher,
) -> JsonValue:
    raw_image: Final = part.get("image_url")
    url: Final = (
        str(raw_image.get("url") or "")
        if isinstance(raw_image, dict)
        else (raw_image if isinstance(raw_image, str) else "")
    )
    format_hint: Final = (
        str(raw_image["format"]) if isinstance(raw_image, dict) and isinstance(raw_image.get("format"), str) else None
    )
    if not url:
        return None
    if url.startswith("data:"):
        mime_type, b64_data = _parse_data_uri(url, format_hint or "image/jpeg")
        return {"type": "image", "mime_type": mime_type, "data": b64_data}
    if _needs_remote_fetch(url, custom_llm_provider):
        fetched_mime, fetched_data = media_fetcher(url, format_hint)
        return {"type": "image", "mime_type": fetched_mime, "data": fetched_data}
    return {
        "type": "image",
        "uri": url,
        **({"mime_type": format_hint} if format_hint else {}),
    }


def _convert_file_part(
    part: JsonMapping,
    model: str,
    custom_llm_provider: str,
    media_fetcher: MediaFetcher,
) -> JsonValue:
    raw_file: Final = part.get("file")
    if not isinstance(raw_file, dict):
        raise litellm.BadRequestError(
            message="Content block has type='file' without a valid 'file' mapping.",
            model=model,
            llm_provider=custom_llm_provider,
        )
    file_data: Final = raw_file.get("file_data")
    file_id: Final = raw_file.get("file_id")
    mime_hint: Final = str(raw_file["format"]) if isinstance(raw_file.get("format"), str) else "application/pdf"
    if isinstance(file_data, str) and file_data:
        if file_data.startswith("data:"):
            mime_type, b64_data = _parse_data_uri(file_data, mime_hint)
            return {"type": "document", "mime_type": mime_type, "data": b64_data}
        return {"type": "document", "mime_type": mime_hint, "data": file_data}
    if isinstance(file_id, str) and file_id:
        if _needs_remote_fetch(file_id, custom_llm_provider):
            fetched_mime, fetched_data = media_fetcher(file_id, mime_hint)
            return {"type": "document", "mime_type": fetched_mime, "data": fetched_data}
        return {"type": "document", "uri": file_id, "mime_type": mime_hint}
    raise litellm.BadRequestError(
        message="Content block with type='file' requires 'file_data' or 'file_id'.",
        model=model,
        llm_provider=custom_llm_provider,
    )


def _convert_content_part(
    part: JsonValue,
    model: str,
    custom_llm_provider: str,
    media_fetcher: MediaFetcher,
) -> JsonValue:
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        return None
    match part.get("type"):
        case "text":
            return {"type": "text", "text": str(part.get("text") or "")}
        case "image_url":
            return _convert_image_part(part, custom_llm_provider, media_fetcher)
        case "image":
            return {k: v for k, v in part.items() if k not in ("cache_control", "prompt_cache_breakpoint")}
        case "input_audio":
            audio_info: Final = part.get("input_audio")
            if not isinstance(audio_info, dict):
                return None
            raw_fmt: Final = str(audio_info.get("format") or "wav").lower()
            mime: Final = _AUDIO_MIME_BY_FORMAT.get(raw_fmt, f"audio/{raw_fmt}")
            return {"type": "audio", "mime_type": mime, "data": str(audio_info.get("data") or "")}
        case "file":
            return _convert_file_part(part, model, custom_llm_provider, media_fetcher)
        case "video_url":
            raw_video: Final = part.get("video_url")
            video_url: Final = (
                str(raw_video.get("url") or "")
                if isinstance(raw_video, dict)
                else (raw_video if isinstance(raw_video, str) else "")
            )
            if not video_url:
                return None
            if video_url.startswith("data:"):
                mime_type, b64_data = _parse_data_uri(video_url, "video/mp4")
                return {"type": "video", "mime_type": mime_type, "data": b64_data}
            return {"type": "video", "uri": video_url}
        case _:
            return None


def _convert_content_blocks(
    content: JsonValue,
    model: str,
    custom_llm_provider: str,
    media_fetcher: MediaFetcher,
) -> list[JsonValue]:  # mutable-ok: JSON wire array serialized directly in request payload
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        return []
    converted: Final = tuple(_convert_content_part(part, model, custom_llm_provider, media_fetcher) for part in content)
    return [block for block in converted if block is not None]


def _extract_text_parts(content: JsonValue) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts: Final = tuple(
        str(part.get("text") or "") for part in content if isinstance(part, dict) and part.get("type") == "text"
    )
    return "\n".join(text for text in texts if text)


@dataclass(frozen=True, slots=True)
class _ParsedThoughtBlock:
    signature: str | None
    summary: str | None

    def to_step(self, *, override_signature: str | None = None) -> JsonValue:
        effective_sig: Final = override_signature or self.signature
        if effective_sig is None and self.summary is None:
            return None
        return {
            "type": "thought",
            **({"signature": effective_sig} if effective_sig else {}),
            **({"summary": [{"type": "text", "text": self.summary}]} if self.summary else {}),
        }


@dataclass(frozen=True, slots=True)
class _ParsedToolCall:
    step: JsonValue
    signature: str | None


@dataclass(frozen=True, slots=True)
class _CallEmitState:
    seen_sigs: frozenset[str]
    emitted: tuple[JsonValue, ...] = ()


def _parse_thought_block(block: JsonMapping) -> _ParsedThoughtBlock | None:
    sig: Final = block.get("signature")
    thinking_text: Final = block.get("thinking")
    signature_str: Final = sig if isinstance(sig, str) and sig else None
    summary_str: Final = thinking_text if isinstance(thinking_text, str) and thinking_text else None
    if signature_str is None and summary_str is None:
        return None
    return _ParsedThoughtBlock(signature=signature_str, summary=summary_str)


def _assistant_thought_blocks(msg: JsonMapping) -> tuple[_ParsedThoughtBlock, ...]:
    raw_blocks: Final = msg.get("thinking_blocks")
    if not isinstance(raw_blocks, list):
        return ()
    candidates: Final = tuple(_parse_thought_block(block) for block in raw_blocks if isinstance(block, dict))
    return tuple(block for block in candidates if block is not None)


def _message_provider_thought_signatures(msg: JsonMapping) -> tuple[str, ...]:
    psf: Final = msg.get("provider_specific_fields")
    if not isinstance(psf, dict):
        return ()
    raw_sigs: Final = psf.get("thought_signatures")
    if not isinstance(raw_sigs, list):
        return ()
    return tuple(sig for sig in raw_sigs if isinstance(sig, str) and sig)


def _step_signature(step: JsonValue) -> str | None:
    if not isinstance(step, dict):
        return None
    sig: Final = step.get("signature")
    return sig if isinstance(sig, str) and sig else None


def _deferred_thought_pair(
    block: _ParsedThoughtBlock,
    later_sigs: frozenset[str],
) -> tuple[str, _ParsedThoughtBlock] | None:
    sig: Final = block.signature
    if sig is not None and sig in later_sigs:
        return sig, block
    return None


def _parse_tool_arguments(raw_args: JsonValue) -> JsonMapping:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str) and raw_args:
        try:
            parsed: Final = _JSON_VALUE.validate_json(raw_args)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            return MappingProxyType({})
    return MappingProxyType({})


def _single_tool_id_name_pair(tc: JsonValue) -> tuple[str, str] | None:
    if not isinstance(tc, dict):
        return None
    fn: Final = tc.get("function")
    if not isinstance(fn, dict):
        return None
    call_id: Final = _strip_signature_suffix(str(tc.get("id") or ""))
    fn_name: Final = str(fn.get("name") or "")
    return (call_id, fn_name) if call_id and fn_name else None


def _tool_id_name_pairs(raw_tool_calls: JsonValue) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw_tool_calls, list):
        return ()
    pairs: Final = tuple(_single_tool_id_name_pair(tc) for tc in raw_tool_calls)
    return tuple(pair for pair in pairs if pair is not None)


def _index_tool_call_names(messages: JsonSequence) -> Mapping[str, str]:
    groups: Final = tuple(
        _tool_id_name_pairs(msg.get("tool_calls"))
        for msg in messages
        if isinstance(msg, dict) and msg.get("role") == "assistant"
    )
    return MappingProxyType(dict(chain.from_iterable(groups)))


def _parse_single_tool_call(idx: int, tc: JsonValue) -> _ParsedToolCall | None:
    if not isinstance(tc, dict):
        return None
    fn: Final = tc.get("function")
    fn_name: Final = str(fn.get("name") or "") if isinstance(fn, dict) else ""
    raw_args: Final = fn.get("arguments") if isinstance(fn, dict) else None
    raw_id: Final = str(tc.get("id") or f"call_{idx}")
    clean_id: Final = _strip_signature_suffix(raw_id)
    extracted_sig: Final = get_thought_signature_from_tool(tc)
    step: Final[JsonValue] = {
        "type": "function_call",
        "id": clean_id,
        "name": fn_name,
        "arguments": dict(_parse_tool_arguments(raw_args)),
    }
    return _ParsedToolCall(step=step, signature=extracted_sig or None)


def _build_leading_thought_steps(
    leading_blocks: tuple[_ParsedThoughtBlock, ...],
    fallback_sig: str | None,
) -> tuple[JsonValue, ...]:
    if any(b.signature is not None for b in leading_blocks):
        raw_steps: Final = tuple(b.to_step() for b in leading_blocks)
        return tuple(step for step in raw_steps if step is not None)
    if fallback_sig is None:
        unsigned_steps: Final = tuple(b.to_step() for b in leading_blocks)
        return tuple(step for step in unsigned_steps if step is not None)
    if leading_blocks:
        merged_first: Final = leading_blocks[0].to_step(override_signature=fallback_sig)
        rest_steps: Final = tuple(b.to_step() for b in leading_blocks[1:])
        combined: Final = (merged_first, *rest_steps)
        return tuple(step for step in combined if step is not None)
    return ({"type": "thought", "signature": fallback_sig},)


def _advance_call_emit(
    deferred_by_sig: Mapping[str, _ParsedThoughtBlock],
    state: _CallEmitState,
    item: _CallEmitState | _ParsedToolCall,
) -> _CallEmitState:
    if isinstance(item, _CallEmitState):
        return item
    sig: Final = item.signature
    if sig is not None and sig not in state.seen_sigs:
        deferred: Final = deferred_by_sig.get(sig)
        thought_step: Final[JsonValue] = (
            deferred.to_step() if deferred is not None else {"type": "thought", "signature": sig}
        )
        steps: Final[tuple[JsonValue, ...]] = (thought_step, item.step) if thought_step is not None else (item.step,)
        return _CallEmitState(seen_sigs=state.seen_sigs | {sig}, emitted=steps)
    return _CallEmitState(seen_sigs=state.seen_sigs, emitted=(item.step,))


def _build_assistant_steps(
    msg: JsonMapping,
    model: str,
    custom_llm_provider: str,
    media_fetcher: MediaFetcher,
) -> tuple[JsonValue, ...]:
    thought_blocks: Final = _assistant_thought_blocks(msg)
    msg_sigs: Final = _message_provider_thought_signatures(msg)
    raw_tool_calls: Final = msg.get("tool_calls")
    built_calls: Final = (
        tuple(_parse_single_tool_call(idx, tc) for idx, tc in enumerate(raw_tool_calls))
        if isinstance(raw_tool_calls, list)
        else ()
    )
    parsed_calls: Final = tuple(call for call in built_calls if call is not None)

    first_call_sig: Final = parsed_calls[0].signature if parsed_calls else None
    later_raw_sigs: Final = tuple(call.signature for call in parsed_calls[1:])
    later_distinct_sigs: Final = frozenset(sig for sig in later_raw_sigs if sig is not None and sig != first_call_sig)
    raw_deferred_pairs: Final = tuple(_deferred_thought_pair(b, later_distinct_sigs) for b in thought_blocks[1:])
    deferred_pairs: Final = tuple(pair for pair in raw_deferred_pairs if pair is not None)
    deferred_by_sig: Final[Mapping[str, _ParsedThoughtBlock]] = MappingProxyType(dict(deferred_pairs))
    leading_blocks: Final = (
        (*thought_blocks[:1], *(b for b in thought_blocks[1:] if b.signature not in deferred_by_sig))
        if deferred_by_sig
        else thought_blocks
    )

    needs_gemini3_signature: Final = VertexGeminiConfig.is_gemini_3_or_newer(model)
    fallback_sig: Final = (
        (msg_sigs[0] if msg_sigs else None)
        or first_call_sig
        or (_DUMMY_THOUGHT_SIGNATURE if parsed_calls and needs_gemini3_signature else None)
    )
    leading_thought_steps: Final = _build_leading_thought_steps(leading_blocks, fallback_sig)

    output_blocks: Final = _convert_content_blocks(msg.get("content"), model, custom_llm_provider, media_fetcher)
    output_steps: Final[tuple[JsonValue, ...]] = (
        ({"type": "model_output", "content": output_blocks},) if output_blocks else ()
    )

    raw_leading_sigs: Final = tuple(_step_signature(step) for step in leading_thought_steps)
    leading_emitted_sigs: Final = frozenset(sig for sig in raw_leading_sigs if sig is not None)
    initial_emit_state: Final = _CallEmitState(seen_sigs=leading_emitted_sigs)
    seeded_calls: Final[tuple[_CallEmitState | _ParsedToolCall, ...]] = (initial_emit_state, *parsed_calls)
    emit_states: Final = tuple(accumulate(seeded_calls, partial(_advance_call_emit, deferred_by_sig)))
    fc_steps: Final = tuple(chain.from_iterable(s.emitted for s in emit_states[1:]))

    return (*leading_thought_steps, *output_steps, *fc_steps)


def _build_tool_result_step(
    msg: JsonMapping,
    tool_names_by_id: Mapping[str, str],
    model: str,
    custom_llm_provider: str,
    media_fetcher: MediaFetcher,
) -> JsonValue:
    raw_call_id: Final = _strip_signature_suffix(str(msg.get("tool_call_id") or ""))
    explicit_name: Final = msg.get("name")
    resolved_name: Final = (
        explicit_name if isinstance(explicit_name, str) and explicit_name else tool_names_by_id.get(raw_call_id, "tool")
    )
    content: Final = msg.get("content")
    result_blocks: Final = (
        [{"type": "text", "text": content}]
        if isinstance(content, str)
        else _convert_content_blocks(content, model, custom_llm_provider, media_fetcher)
    )
    return _JSON_VALUE.validate_python(
        {
            "type": "function_result",
            "call_id": raw_call_id,
            "name": resolved_name,
            "result": result_blocks,
        }
    )


def _steps_for_non_system_message(
    msg: JsonMapping,
    tool_names_by_id: Mapping[str, str],
    model: str,
    custom_llm_provider: str,
    media_fetcher: MediaFetcher,
) -> tuple[JsonValue, ...]:
    match msg.get("role"):
        case "user":
            user_blocks: Final = _convert_content_blocks(msg.get("content"), model, custom_llm_provider, media_fetcher)
            return ({"type": "user_input", "content": user_blocks},) if user_blocks else ()
        case "assistant" | "model":
            return _build_assistant_steps(msg, model, custom_llm_provider, media_fetcher)
        case "tool" | "function":
            return (_build_tool_result_step(msg, tool_names_by_id, model, custom_llm_provider, media_fetcher),)
        case _:
            return ()


def _convert_messages(
    messages: JsonSequence,
    model: str,
    custom_llm_provider: str,
    media_fetcher: MediaFetcher,
) -> tuple[JsonSequence, str | None]:
    tool_names_by_id: Final = _index_tool_call_names(messages)
    dict_messages: Final = tuple(msg for msg in messages if isinstance(msg, dict))
    raw_system: Final = tuple(
        _extract_text_parts(msg.get("content")) for msg in dict_messages if msg.get("role") in ("system", "developer")
    )
    system_chunks: Final = tuple(chunk for chunk in raw_system if chunk)
    step_groups: Final = tuple(
        _steps_for_non_system_message(msg, tool_names_by_id, model, custom_llm_provider, media_fetcher)
        for msg in dict_messages
    )
    steps: Final = list(chain.from_iterable(step_groups))

    combined_system: Final = "\n\n".join(system_chunks) if system_chunks else None
    if not steps and combined_system is not None:
        return [{"type": "user_input", "content": [{"type": "text", "text": combined_system}]}], None
    return steps, combined_system


def _matching_builtin_tool(tool: JsonMapping) -> JsonValue:
    matched: Final = next((mapped for raw_key, mapped in _BUILTIN_TOOL_TYPES.items() if raw_key in tool), None)
    return {"type": matched} if matched is not None else None


def _convert_single_tool(tool: JsonValue) -> tuple[JsonValue, ...]:
    if not isinstance(tool, dict):
        return ()
    if (builtin := _matching_builtin_tool(tool)) is not None:
        return (builtin,)
    tool_type: Final = tool.get("type")
    if isinstance(tool_type, str) and tool_type in _BUILTIN_TOOL_TYPES:
        return ({**dict(tool), "type": _BUILTIN_TOOL_TYPES[tool_type]},)
    if isinstance(tool_type, str) and tool_type in (
        "file_search",
        "computer_use",
        "mcp_server",
    ):
        return (dict(tool),)
    declarations: Final = tool.get("functionDeclarations") or tool.get("function_declarations")
    if isinstance(declarations, list):
        return tuple(
            {
                "type": "function",
                "name": decl.get("name"),
                "description": decl.get("description") or "",
                **({"parameters": decl["parameters"]} if "parameters" in decl else {}),
            }
            for decl in declarations
            if isinstance(decl, dict) and decl.get("name")
        )
    nested_fn: Final = tool.get("function")
    if isinstance(nested_fn, dict) and nested_fn.get("name"):
        return (
            {
                "type": "function",
                "name": nested_fn.get("name"),
                "description": nested_fn.get("description") or "",
                **({"parameters": nested_fn["parameters"]} if "parameters" in nested_fn else {}),
            },
        )
    if tool.get("name"):
        return (
            {
                "type": "function",
                "name": tool.get("name"),
                "description": tool.get("description") or "",
                **({"parameters": tool["parameters"]} if "parameters" in tool else {}),
            },
        )
    return ()


def _convert_tools(params: JsonMapping) -> JsonValue:
    raw_tools: Final = params.get("tools")
    raw_functions: Final = params.get("functions")
    from_tools: Final = (
        tuple(chain.from_iterable(_convert_single_tool(item) for item in raw_tools))
        if isinstance(raw_tools, list)
        else ()
    )
    from_functions: Final = (
        tuple(chain.from_iterable(_convert_single_tool(fn) for fn in raw_functions))
        if isinstance(raw_functions, list)
        else ()
    )
    combined: Final = (*from_tools, *from_functions)
    has_search: Final = any(isinstance(t, dict) and t.get("type") == "google_search" for t in combined)
    with_search: Final = (
        (*combined, {"type": "google_search"})
        if params.get("web_search_options") is not None and not has_search
        else combined
    )
    return _JSON_VALUE.validate_python(list(with_search)) if with_search else None


def _is_valid_allowed_tools_config(cfg: JsonMapping) -> bool:
    if not set(cfg).issubset({"mode", "tools"}):
        return False
    mode: Final = cfg.get("mode")
    if mode is not None and mode not in ("auto", "any", "none", "validated"):
        return False
    tools: Final = cfg.get("tools")
    return tools is None or (isinstance(tools, list) and all(isinstance(t, str) for t in tools))


def _extract_allowed_tool_name(tool: Mapping[str, object]) -> str | None:
    if tool.get("type") not in ("function", "custom"):
        return None
    name: Final = tool.get("name")
    return name if isinstance(name, str) and name else None


def _convert_openai_allowed_tools(raw_choice: JsonValue) -> JsonValue:
    parsed: Final = parse_chat_allowed_tools_tool_choice(raw_choice)
    if parsed is None:
        return None
    flattened: Final = responses_allowed_tools_tool_choice(parsed)
    if flattened is None:
        return None
    names: Final = tuple(_extract_allowed_tool_name(tool) for tool in flattened["tools"])
    if any(name is None for name in names):
        return None
    mode: Final = "any" if parsed.allowed_tools.mode == "required" else parsed.allowed_tools.mode
    return {"allowed_tools": {"mode": mode, "tools": [name for name in names if name is not None]}}


def _convert_tool_choice(raw_choice: JsonValue) -> JsonValue:
    match raw_choice:
        case "auto" | "none" | "validated" | "any":
            return raw_choice
        case "required":
            return "any"
        case {"allowed_tools": dict() as allowed_cfg} if len(raw_choice) == 1 and _is_valid_allowed_tools_config(
            allowed_cfg
        ):
            return {"allowed_tools": allowed_cfg}
        case {"type": "allowed_tools"}:
            return _convert_openai_allowed_tools(raw_choice)
        case {"type": "function", "function": {"name": str() as fn_name}} if fn_name:
            return {"allowed_tools": {"mode": "any", "tools": [fn_name]}}
        case {"type": "function", "name": str() as fn_name} if fn_name:
            return {"allowed_tools": {"mode": "any", "tools": [fn_name]}}
        case _:
            return None


def _base_response_format(raw_rf: JsonValue) -> JsonValue:
    if isinstance(raw_rf, list):
        return raw_rf
    if not isinstance(raw_rf, dict):
        return None
    match raw_rf:
        case {"type": "json_object"}:
            return {"type": "text", "mime_type": "application/json"}
        case {"type": "json_schema", "json_schema": {"schema": dict() as schema}}:
            return {"type": "text", "mime_type": "application/json", "schema": schema}
        case {"type": "text", "mime_type": _} | {"type": "image"} | {"type": "audio"}:
            return raw_rf
        case {"type": "object"}:
            return {"type": "text", "mime_type": "application/json", "schema": raw_rf}
        case {"schema": dict() as schema}:
            return {
                "type": "text",
                "mime_type": str(raw_rf.get("mime_type") or "application/json"),
                "schema": schema,
            }
        case _:
            return None


def _modality_format_entry(mod: str, base_format: JsonValue) -> JsonValue:
    if mod == "text" and isinstance(base_format, dict):
        return base_format
    return {"type": mod}


def _convert_response_format(params: JsonMapping) -> JsonValue:
    response_json_schema: Final = params.get("response_json_schema")
    if isinstance(response_json_schema, dict):
        return {"type": "text", "mime_type": "application/json", "schema": response_json_schema}
    response_schema: Final = params.get("response_schema")
    if isinstance(response_schema, dict):
        return {"type": "text", "mime_type": "application/json", "schema": response_schema}

    base_format: Final = _base_response_format(params.get("response_format"))
    modalities: Final = params.get("modalities")
    has_audio_param: Final = isinstance(params.get("audio"), dict)
    if isinstance(modalities, list):
        lowered: Final = tuple(str(m).lower() for m in modalities)
        entries: Final = [
            _modality_format_entry(mod, base_format) for mod in lowered if mod in ("text", "image", "audio")
        ]
        if len(entries) == 1:
            return entries[0]
        if entries:
            return entries
    if has_audio_param and base_format is None:
        return {"type": "audio"}
    return base_format


def _minimal_flag_from_mapping(entry_obj: object) -> bool | None:
    if not isinstance(entry_obj, Mapping):
        return None
    entry: Final = _OBJECT_MAP.validate_python(entry_obj)
    flag: Final = entry.get("supports_minimal_reasoning_effort")
    if isinstance(flag, bool):
        return flag
    levels: Final = entry.get("reasoning_effort_levels")
    if isinstance(levels, Sequence) and not isinstance(levels, (str, bytes)):
        return "minimal" in _OBJECT_SEQ.validate_python(levels)
    return None


def _supports_minimal_thinking_level(model: str, model_info: object = None) -> bool:
    deployment_flag: Final = _minimal_flag_from_mapping(model_info)
    if deployment_flag is not None:
        return deployment_flag
    bare: Final = model.split("/")[-1]
    cost_flags: Final = tuple(
        _minimal_flag_from_mapping(
            litellm.model_cost.get(candidate)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]  # model_cost is an untyped global dict
        )
        for candidate in (model, bare, f"gemini/{bare}", f"vertex_ai/{bare}")
    )
    explicit_flag: Final = next((flag for flag in cost_flags if flag is not None), None)
    if explicit_flag is not None:
        return explicit_flag
    return VertexGeminiConfig.is_gemini_3_or_newer(model) and "flash" in model.lower()


def _lowest_thinking_level_for_model(model: str, model_info: object = None) -> str:
    return "minimal" if _supports_minimal_thinking_level(model, model_info) else "low"


def _resolve_thinking_fields(params: JsonMapping, model: str, model_info: object = None) -> JsonMapping:
    lowest_level: Final = _lowest_thinking_level_for_model(model, model_info)
    effort: Final = params.get("reasoning_effort")
    effort_str: Final = (
        effort.get("effort")
        if isinstance(effort, dict) and isinstance(effort.get("effort"), str)
        else (effort if isinstance(effort, str) else None)
    )
    if effort_str == "minimal":
        return MappingProxyType({"thinking_level": lowest_level, "thinking_summaries": "auto"})
    if effort_str in ("low", "medium", "high"):
        return MappingProxyType({"thinking_level": effort_str, "thinking_summaries": "auto"})
    if effort_str in ("none", "disable"):
        return MappingProxyType({"thinking_level": lowest_level, "thinking_summaries": "none"})

    thinking: Final = params.get("thinking")
    if isinstance(thinking, dict):
        t_type: Final = thinking.get("type")
        budget: Final = thinking.get("budget_tokens")
        if t_type == "disabled" or budget == 0:
            return MappingProxyType({"thinking_level": lowest_level, "thinking_summaries": "none"})
        if t_type == "enabled" or isinstance(budget, int):
            level: Final = (
                "low"
                if isinstance(budget, int) and budget <= 1024
                else ("medium" if isinstance(budget, int) and budget <= 8192 else "high")
            )
            return MappingProxyType({"thinking_level": level, "thinking_summaries": "auto"})
    return MappingProxyType({})


def _build_generation_config(params: JsonMapping, model: str, model_info: object = None) -> JsonValue:
    allows_sampling: Final = VertexGeminiConfig.supports_sampling_params(model)
    raw_existing: Final = params.get("generation_config")
    existing: Final = (
        {k: v for k, v in raw_existing.items() if allows_sampling or k not in ("temperature", "top_p")}
        if isinstance(raw_existing, dict)
        else _EMPTY_JSON_MAP
    )

    temp: Final = params.get("temperature") if allows_sampling else None
    top_p: Final = params.get("top_p") if allows_sampling else None
    seed: Final = params.get("seed")
    max_tokens: Final = (
        params.get("max_completion_tokens")
        if params.get("max_completion_tokens") is not None
        else params.get("max_tokens")
    )
    raw_stop: Final = params.get("stop")
    stop_seq: Final = (
        [raw_stop]
        if isinstance(raw_stop, str)
        else ([str(item) for item in raw_stop] if isinstance(raw_stop, list) else None)
    )
    mapped_choice: Final = _convert_tool_choice(params.get("tool_choice"))
    raw_audio: Final = params.get("audio")
    speech_cfg: Final = (
        [{"voice": OPENAI_TO_GEMINI_TTS_VOICES.get(str(raw_audio["voice"]), str(raw_audio["voice"]))}]
        if isinstance(raw_audio, dict) and isinstance(raw_audio.get("voice"), str)
        else None
    )

    merged: Final = {
        **existing,
        **({"temperature": temp} if isinstance(temp, (int, float)) and not isinstance(temp, bool) else {}),
        **({"top_p": top_p} if isinstance(top_p, (int, float)) and not isinstance(top_p, bool) else {}),
        **({"seed": seed} if isinstance(seed, int) and not isinstance(seed, bool) else {}),
        **(
            {"max_output_tokens": max_tokens}
            if isinstance(max_tokens, int) and not isinstance(max_tokens, bool)
            else {}
        ),
        **({"stop_sequences": stop_seq} if stop_seq is not None else {}),
        **({"tool_choice": mapped_choice} if mapped_choice is not None else {}),
        **({"speech_config": speech_cfg} if speech_cfg is not None else {}),
        **_resolve_thinking_fields(params, model, model_info),
    }
    return _JSON_VALUE.validate_python(merged) if merged else None


def _advance_mixed_turn_scan(saw_output: bool, step: bool | JsonValue) -> bool:
    if isinstance(step, bool):
        return step
    if not isinstance(step, dict):
        return saw_output
    match step.get("type"):
        case "model_output":
            return True
        case "user_input" | "function_result":
            return False
        case _:
            return saw_output


def _replays_model_output_before_function_call(steps: JsonSequence) -> bool:
    seeded: Final[tuple[bool | JsonValue, ...]] = (False, *steps)
    saw_before_each: Final = tuple(accumulate(seeded, _advance_mixed_turn_scan))
    return any(
        isinstance(step, dict) and step.get("type") == "function_call" and saw_output
        for step, saw_output in zip(steps, saw_before_each)
    )


def transform_chat_to_interactions_request(
    *,
    messages: Sequence[object],
    model: str,
    optional_params: Mapping[str, object],
    litellm_params: Mapping[str, object],
    custom_llm_provider: str = "gemini",
    stream: bool = False,
    media_fetcher: MediaFetcher | None = None,
) -> InteractionsPreparedRequest:
    normalized_messages: Final = _normalize_to_json(list(messages))
    normalized_opt: Final = _normalize_to_json(optional_params)
    opt_dict: Final = normalized_opt if isinstance(normalized_opt, dict) else _EMPTY_JSON_MAP
    raw_extra: Final = opt_dict.get("extra_body")
    normalized_params: Final = {
        **(raw_extra if isinstance(raw_extra, dict) else {}),
        **opt_dict,
    }
    drop_params: Final = bool(litellm_params.get("drop_params") or optional_params.get("drop_params"))

    _validate_request_params(
        normalized_params,
        model=model,
        drop_params=drop_params,
        custom_llm_provider=custom_llm_provider,
    )

    fetcher: Final = media_fetcher or _default_media_fetcher
    message_list: Final = normalized_messages if isinstance(normalized_messages, list) else []
    steps, system_instruction = _convert_messages(
        message_list,
        model=model,
        custom_llm_provider=custom_llm_provider,
        media_fetcher=fetcher,
    )

    tools: Final = _convert_tools(normalized_params)
    response_format: Final = _convert_response_format(normalized_params)
    generation_config: Final = _build_generation_config(
        normalized_params,
        model=model,
        model_info=litellm_params.get("model_info"),
    )
    explicit_store: Final = normalized_params.get("store")
    effective_store: Final = (
        explicit_store
        if explicit_store is not None
        else (
            False
            if custom_llm_provider in ("vertex_ai", "vertex_ai_beta")
            and normalized_params.get("previous_interaction_id") is None
            and _replays_model_output_before_function_call(steps)
            else None
        )
    )
    passthrough_entries: Final = {
        **({"store": effective_store} if effective_store is not None else {}),
        **{
            key: normalized_params[key]
            for key in ("previous_interaction_id", "background", "environment", "response_modalities")
            if normalized_params.get(key) is not None
        },
    }
    interactions_params: Final = {
        **({"system_instruction": system_instruction} if system_instruction is not None else {}),
        **({"tools": tools} if tools is not None else {}),
        **({"response_format": response_format} if response_format is not None else {}),
        **({"generation_config": generation_config} if generation_config is not None else {}),
        **({"stream": True} if stream or normalized_params.get("stream") is True else {}),
        **passthrough_entries,
    }

    return InteractionsPreparedRequest(
        input_steps=steps,
        optional_params=interactions_params,
    )
