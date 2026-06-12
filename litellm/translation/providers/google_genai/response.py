"""generateContent response JSON -> IR ``ChatResponse``.

Mirrors v1's ``_transform_google_generate_content_to_openai_model_response``
for the surface a v2-sent request can produce: text and ``thought`` parts,
complete ``functionCall`` parts (native ids kept, synthesized ids minted by
the SEAM because uuid is ambient — the IR carries the empty-id sentinel), and
``usageMetadata`` including cached/thoughts token counts with v1's modality
math. promptFeedback blocks, flagged finish reasons, multiple candidates,
inline media, grounding metadata, and logprobs are loud errors naming the v1
path.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from expression import Error, Nothing, Ok, Option, Result, Some
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import (
    ChatRequest,
    ChatResponse,
    ContentBlock,
    JsonBlob,
    ModalityTokens,
    PlainJson,
    ResponseUsage,
    Text,
    Thinking,
    ToolUse,
)
from . import params as p

_ParseResult = Result[ChatResponse, TranslationError]

_UNSUPPORTED_CANDIDATE_KEYS = (
    "groundingMetadata",
    "urlContextMetadata",
    "logprobsResult",
)


def parse_response(raw: PlainJson, request: ChatRequest) -> _ParseResult:
    if not isinstance(raw, dict):
        return Error(_boundary("response body is not a JSON object"))
    if "error" in raw:
        return Error(_boundary(f"provider error payload: {raw['error']!r}"))
    candidate = _single_candidate(raw)
    if isinstance(candidate, TranslationError):
        return Error(candidate)
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        return Error(_boundary("candidate content has no parts"))
    blocks = _parse_parts(parts)
    if isinstance(blocks, TranslationError):
        return Error(blocks)
    usage = parse_usage_metadata(raw.get("usageMetadata"))
    if isinstance(usage, TranslationError):
        return Error(usage)
    finish_raw = candidate.get("finishReason")
    has_tools = any(block.tag == "tool_use" for block in blocks)
    finish = p.map_finish(
        finish_raw if isinstance(finish_raw, str) else None, has_tools
    )
    response_id = raw.get("responseId")
    return Ok(
        ChatResponse(
            id=response_id if isinstance(response_id, str) else "",
            model=request.model,
            content=Block.of_seq(blocks),
            finish=finish,
            usage=usage,
            synthesized_json_content=False,
        )
    )


def _single_candidate(
    raw: dict[str, PlainJson],
) -> dict[str, PlainJson] | TranslationError:
    feedback = raw.get("promptFeedback")
    if isinstance(feedback, dict) and "blockReason" in feedback:
        return TranslationError.of_unsupported(
            "promptFeedback.blockReason responses take v1's _handle_blocked_response path"
        )
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or len(candidates) == 0:
        return _boundary("response has no candidates")
    if len(candidates) > 1:
        return TranslationError.of_unsupported(
            "multiple candidates (n > 1); v1 emits multiple choices"
        )
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        return _boundary("candidate is not an object")
    finish_raw = candidate.get("finishReason")
    if isinstance(finish_raw, str) and finish_raw in p.FLAGGED_FINISH_REASONS:
        return TranslationError.of_unsupported(
            "content-policy finishReason; v1 takes _handle_content_policy_violation"
        )
    for key in _UNSUPPORTED_CANDIDATE_KEYS:
        if key in candidate:
            return TranslationError.of_unsupported(
                f"candidate {key}; v1's metadata extraction handles it"
            )
    return candidate


def _boundary(reason: str) -> TranslationError:
    return TranslationError.of_boundary(BoundaryError.of(Block.of_seq([reason])))


def _parse_parts(parts: list[PlainJson]) -> list[ContentBlock] | TranslationError:
    blocks: list[ContentBlock] = []
    for part in parts:
        if not isinstance(part, dict):
            return _boundary("response part is not an object")
        block = _parse_part(part)
        if isinstance(block, TranslationError):
            return block
        blocks = [*blocks, block]
    return blocks


def _parse_part(part: dict[str, PlainJson]) -> ContentBlock | TranslationError:
    if "functionCall" in part:
        return _function_call_block(part)
    if "inlineData" in part:
        return TranslationError.of_unsupported(
            "inlineData response parts (image/audio output); v1 handles them"
        )
    text = part.get("text")
    if not isinstance(text, str):
        return TranslationError.of_unsupported(
            f"response part keys {sorted(part)!r}; v1 handles them"
        )
    if text.startswith("data:audio") and ";base64," in text:
        return TranslationError.of_unsupported(
            "audio data-URI text parts; v1's audio extraction handles them"
        )
    signature = part.get("thoughtSignature")
    signature_option: Option[str] = (
        Some(signature) if isinstance(signature, str) else Nothing
    )
    if part.get("thought") is True:
        return ContentBlock.of_thinking(
            Thinking(thinking=text, signature=signature_option, cache=Nothing)
        )
    if isinstance(signature, str):
        return TranslationError.of_unsupported(
            "thoughtSignature on a non-thought text part; v1 collects it into thought_signatures"
        )
    return ContentBlock.of_text(Text(text=text, cache=Nothing))


def _function_call_block(part: dict[str, PlainJson]) -> ContentBlock | TranslationError:
    call = part.get("functionCall")
    if not isinstance(call, dict):
        return _boundary("functionCall part is malformed")
    name = call.get("name")
    args = call.get("args")
    if not isinstance(name, str) or not isinstance(args, dict):
        return _boundary("functionCall part is missing name/args")
    native_id = call.get("id")
    identifier = native_id if isinstance(native_id, str) else ""
    signature = part.get("thoughtSignature")
    if isinstance(signature, str) and signature:
        identifier = f"{identifier}{p.THOUGHT_SIGNATURE_SEPARATOR}{signature}"
    return ContentBlock.of_tool_use(
        ToolUse(
            id=identifier,
            name=name,
            arguments=JsonBlob(value=dict(args)),
            cache=Nothing,
        )
    )


_MODALITY_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "TEXT": "text",
        "DOCUMENT": "text",
        "AUDIO": "audio",
        "IMAGE": "image",
        "VIDEO": "video",
    }
)


def _modality_totals(details: PlainJson) -> dict[str, int]:
    totals: dict[str, int] = {}
    if not isinstance(details, list):
        return totals
    for detail in details:
        if not isinstance(detail, dict):
            continue
        modality = str(detail.get("modality", "")).upper()
        key = _MODALITY_KEYS.get(modality)
        if key is None:
            continue
        count = detail.get("tokenCount", detail.get("token_count", 0))
        count = count if isinstance(count, int) and not isinstance(count, bool) else 0
        totals = {**totals, key: totals.get(key, 0) + count}
    return totals


def _int_at(raw: dict[str, PlainJson], key: str) -> int | None:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def parse_usage_metadata(metadata: PlainJson) -> ResponseUsage | TranslationError:
    """Port of v1 ``_calculate_usage`` onto the IR (modality math included)."""
    if not isinstance(metadata, dict):
        return _boundary("usageMetadata not found in completion_response (v1 raises)")
    cached = _int_at(metadata, "cachedContentTokenCount")
    prompt_tokens = _int_at(metadata, "promptTokenCount") or 0
    candidates_tokens = _int_at(metadata, "candidatesTokenCount") or 0
    total_tokens = _int_at(metadata, "totalTokenCount") or 0
    reasoning = _int_at(metadata, "thoughtsTokenCount")

    prompt_totals = _modality_totals(metadata.get("promptTokensDetails"))
    cache_totals = _modality_totals(metadata.get("cacheTokensDetails"))
    completion_totals = _modality_totals(metadata.get("candidatesTokensDetails"))

    prompt_text = prompt_totals.get("text")
    if cache_totals.get("text") is not None and prompt_text is not None:
        prompt_text = prompt_text - cache_totals["text"]
    elif (
        cached is not None
        and prompt_text is not None
        and "cacheTokensDetails" not in metadata
    ):
        prompt_text = prompt_text - cached
    prompt_modalities = ModalityTokens(
        text=_some_or_nothing(prompt_text),
        audio=_subtracted(prompt_totals, cache_totals, "audio"),
        image=_subtracted(prompt_totals, cache_totals, "image"),
        video=_subtracted(prompt_totals, cache_totals, "video"),
    )

    completion_text = completion_totals.get("text")
    if candidates_tokens > 0 and completion_text is None:
        completion_text = (
            candidates_tokens
            - completion_totals.get("image", 0)
            - completion_totals.get("audio", 0)
            - completion_totals.get("video", 0)
        )
    completion_modalities = ModalityTokens(
        text=_some_or_nothing(completion_text),
        audio=_some_or_nothing(completion_totals.get("audio")),
        image=_some_or_nothing(completion_totals.get("image")),
        video=_some_or_nothing(completion_totals.get("video")),
    )

    inclusive = prompt_tokens + candidates_tokens == total_tokens
    completion_tokens = candidates_tokens
    if not inclusive and reasoning:
        completion_tokens = reasoning + candidates_tokens
    return ResponseUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=cached if cached is not None else 0,
        cache_creation=Nothing,
        total_tokens=Some(total_tokens),
        reasoning_tokens=_some_or_nothing(reasoning),
        prompt_modalities=Some(prompt_modalities),
        completion_modalities=Some(completion_modalities),
        cache_read_reported=cached is not None,
    )


def _some_or_nothing(value: int | None) -> Option[int]:
    return Some(value) if value is not None else Nothing


def _subtracted(
    totals: dict[str, int], cache_totals: dict[str, int], key: str
) -> Option[int]:
    value = totals.get(key)
    if value is None:
        return Nothing
    cached = cache_totals.get(key)
    if cached is not None:
        return Some(value - cached)
    return Some(value)
