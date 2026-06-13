"""OpenAI chat-completion response JSON -> IR ``ChatResponse``.

The live v1 normalizer for this route is ``convert_to_model_response_object``
(litellm_core_utils/llm_response_utils/convert_dict_to_response.py:474), NOT
``OpenAIGPTConfig.transform_response`` (dead code on the SDK path, dossier
gotcha #1). This parser mirrors its completion branch over the SDK-dump
response shape: finish_reason ``stop`` -> ``tool_calls`` when tool calls are
present, ``reasoning_content``/``reasoning`` extraction (including the
``<think>`` tag split), unknown message keys folded into
``provider_specific_fields``, and verbatim usage passthrough. Because the
outbound dialect is the same family, the normalized chat-completion body is
built HERE and rides on ``ChatResponse.wire``; the inbound ``openai``
dialect emits it unchanged. Shapes a v2-sent request cannot trigger
(multiple choices, legacy ``function_call``, audio/images output, the
``multi_tool_use.parallel`` repair) are loud error values.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import replace

from expression import Error, Nothing, Ok, Result, Some
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import (
    ChatRequest,
    ChatResponse,
    ContentBlock,
    FinishReason,
    JsonBlob,
    PlainJson,
    ResponseUsage,
    Text,
    ToolUse,
)

_ParseResult = Result[ChatResponse, TranslationError]

# Mirrors of v1's field allowlists (convert_dict_to_response.py:49-53,
# frozensets over litellm.types.utils model fields at HEAD). The differential
# gate pins them: if litellm adds a Message/Choices field, the corpus drifts
# and these must be re-synced in the same commit.
_MESSAGE_FIELDS = frozenset(
    {
        "annotations",
        "audio",
        "content",
        "function_call",
        "images",
        "provider_specific_fields",
        "reasoning_content",
        "reasoning_items",
        "role",
        "thinking_blocks",
        "tool_calls",
    }
)
_CHOICES_FIELDS = frozenset(
    {"finish_reason", "index", "logprobs", "message", "provider_specific_fields"}
)
_MODEL_RESPONSE_FIELDS = frozenset(
    {"choices", "created", "id", "model", "object", "system_fingerprint", "usage"}
)

_REASONING_TAG_PATTERN = re.compile(
    r"<(?:think|thinking|budget:thinking)>(.*?)</(?:think|thinking|budget:thinking)>(.*)",
    re.DOTALL,
)


def parse_response(raw: PlainJson, request: ChatRequest) -> _ParseResult:
    if not isinstance(raw, dict):
        return Error(_boundary("response body is not a JSON object"))
    choice = _validated_choice(raw)
    if isinstance(choice, TranslationError):
        return Error(choice)
    message = choice.get("message")
    if not isinstance(message, dict):
        return Error(_boundary("response choice 'message' is not an object"))
    normalized = _normalize_message(message)
    if isinstance(normalized, TranslationError):
        return Error(normalized)
    finish = _finish_reason(choice, normalized)
    if isinstance(finish, TranslationError):
        return Error(finish)
    wire_finish, semantic_finish = finish
    blocks = _semantic_blocks(normalized)
    if isinstance(blocks, TranslationError):
        return Error(blocks)
    body = _outbound_body(raw, choice, normalized, wire_finish)
    model = raw.get("model")
    response_id = raw.get("id")
    return Ok(
        ChatResponse(
            id=response_id if isinstance(response_id, str) else "",
            model=model if isinstance(model, str) else request.model,
            content=Block.of_seq(blocks),
            finish=semantic_finish,
            usage=semantic_usage(raw.get("usage")),
            synthesized_json_content=False,
            wire=Some(JsonBlob(value=body)),
        )
    )


def _boundary(reason: str) -> TranslationError:
    return TranslationError.of_boundary(BoundaryError.of(Block.of_seq([reason])))


def _validated_choice(
    raw: dict[str, PlainJson],
) -> dict[str, PlainJson] | TranslationError:
    if _meaningful_error(raw.get("error")):
        return _boundary(f"provider error payload: {raw['error']!r}")
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        return _boundary("response 'choices' is missing or empty")
    if len(choices) > 1:
        return TranslationError.of_unsupported(
            "multiple response choices (n > 1); unreachable for v2-sent requests"
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        return _boundary("response choice is not an object")
    if choice.get("logprobs") is not None:
        return TranslationError.of_unsupported(
            "response logprobs; unreachable for v2-sent requests"
        )
    return choice


def _meaningful_error(error: PlainJson) -> bool:
    """v1 raises only on errors carrying data (convert_dict_to_response.py:
    515-552); some compat providers send empty error objects on success."""
    if error is None:
        return False
    if isinstance(error, dict):
        return bool(error.get("message")) or error.get("code") is not None
    if isinstance(error, str):
        return bool(error)
    return True


def _normalize_message(
    message: dict[str, PlainJson],
) -> dict[str, PlainJson] | TranslationError:
    """The exact kwarg set v1 passes to ``Message(...)`` (lines 596-649)."""
    for field, reason in (
        (
            "function_call",
            "legacy function_call output; the v2 surface cannot send 'functions'",
        ),
        ("audio", "audio output; the v2 surface cannot send 'modalities'"),
        ("images", "image output; unreachable for v2-sent requests"),
        ("reasoning_items", "reasoning_items output; unreachable for v2-sent requests"),
    ):
        if message.get(field) is not None:
            return TranslationError.of_unsupported(f"response {reason}; v1 handles it")
    tool_calls = message.get("tool_calls")
    if tool_calls is not None and not isinstance(tool_calls, list):
        return _boundary("response message 'tool_calls' is not an array")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            name = function.get("name") if isinstance(function, dict) else None
            if name == "multi_tool_use.parallel":
                return TranslationError.of_unsupported(
                    "multi_tool_use.parallel pseudo-call needs v1's repair "
                    "(_handle_invalid_parallel_tool_calls)"
                )
    seeded = message.get("provider_specific_fields")
    provider_fields: dict[str, PlainJson] = (
        dict(seeded) if isinstance(seeded, dict) else {}
    )
    for key in message.keys() - _MESSAGE_FIELDS:
        provider_fields = {**provider_fields, key: message[key]}
    reasoning_content, content = _extract_reasoning(message)
    thinking_blocks = (
        message.get("thinking_blocks") if "thinking_blocks" in message else None
    )
    if "thinking_blocks" in message:
        provider_fields = {**provider_fields, "thinking_blocks": thinking_blocks}
    role = message.get("role")
    return {
        "content": content,
        "role": role if isinstance(role, str) and role else "assistant",
        "function_call": None,
        "tool_calls": tool_calls,
        "audio": None,
        "provider_specific_fields": provider_fields,
        "reasoning_content": reasoning_content,
        "thinking_blocks": thinking_blocks,
        "annotations": message.get("annotations"),
        "images": None,
    }


def _extract_reasoning(
    message: dict[str, PlainJson],
) -> tuple[PlainJson, PlainJson]:
    """v1 ``_extract_reasoning_content`` + ``_parse_content_for_reasoning``."""
    content = message.get("content")
    if "reasoning_content" in message:
        return message["reasoning_content"], content
    if "reasoning" in message:
        return message["reasoning"], content
    if isinstance(content, str) and content:
        matched = _REASONING_TAG_PATTERN.match(content)
        if matched:
            return matched.group(1), matched.group(2)
    return None, content


def _finish_reason(
    choice: dict[str, PlainJson], normalized: dict[str, PlainJson]
) -> tuple[str, FinishReason] | TranslationError:
    """Returns (wire finish string, semantic IR finish).

    Post-send leniency (the PR #30138 boundary, integrator sign-off at
    integration): the request is already sent and billed, so a quirky compat
    finish value must not become an outage. The NATIVE string rides the wire
    body verbatim and the seam's ``Choices.__init__`` runs v1's live
    ``map_finish_reason`` (table lookup, unmapped -> "stop", native value
    stashed in provider_specific_fields) — v2 inherits v1's lenient mapping
    with no mirror table. The semantic IR field mirrors v1's unmapped
    default; no outbound dialect consumes it on this same-family path."""
    raw_finish = choice.get("finish_reason")
    if raw_finish is None and choice.get("finish_details") is not None:
        return TranslationError.of_unsupported(
            "finish_details in place of finish_reason; v1 handles it"
        )
    finish = raw_finish if isinstance(raw_finish, str) else "stop"
    tool_calls = normalized.get("tool_calls")
    if finish == "stop" and isinstance(tool_calls, list) and len(tool_calls) > 0:
        finish = "tool_calls"
    return finish, _semantic_finish(finish)


def _semantic_finish(finish: str) -> FinishReason:
    match finish:
        case "stop" | "length" | "tool_calls" | "content_filter":
            return finish
        case _:
            return "stop"


def _outbound_body(
    raw: dict[str, PlainJson],
    choice: dict[str, PlainJson],
    normalized: dict[str, PlainJson],
    finish: str,
) -> dict[str, PlainJson]:
    choice_fields = {key: choice[key] for key in choice.keys() - _CHOICES_FIELDS}
    body: dict[str, PlainJson] = {
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": finish,
                "index": 0,
                "message": normalized,
                "logprobs": None,
                "enhancements": None,
                "provider_specific_fields": choice_fields,
            }
        ],
    }
    if raw.get("usage") is not None:
        body = {**body, "usage": raw["usage"]}
    if "created" in raw:
        body = {**body, "created": raw["created"]}
    if isinstance(raw.get("id"), str) and raw["id"]:
        body = {**body, "id": raw["id"]}
    if "system_fingerprint" in raw:
        body = {**body, "system_fingerprint": raw["system_fingerprint"]}
    if "model" in raw:
        body = {**body, "model": raw["model"]}
    extras = {
        key: raw[key] for key in raw.keys() - _MODEL_RESPONSE_FIELDS if key != "error"
    }
    return {**body, **extras}


def _semantic_blocks(
    normalized: dict[str, PlainJson],
) -> list[ContentBlock] | TranslationError:
    blocks: list[ContentBlock] = []
    content = normalized.get("content")
    if isinstance(content, str) and content:
        blocks = [ContentBlock.of_text(Text(text=content, cache=Nothing))]
    tool_calls = normalized.get("tool_calls")
    if not isinstance(tool_calls, list):
        return blocks
    for call in tool_calls:
        block = _tool_use_block(call)
        if isinstance(block, TranslationError):
            return block
        blocks = [*blocks, block]
    return blocks


def _tool_use_block(call: PlainJson) -> ContentBlock | TranslationError:
    if not isinstance(call, dict):
        return _boundary("response tool_call is not an object")
    function = call.get("function")
    if not isinstance(function, dict):
        return _boundary("response tool_call is missing 'function'")
    name = function.get("name")
    identifier = call.get("id")
    arguments = function.get("arguments")
    try:
        parsed: PlainJson = (
            json.loads(arguments) if isinstance(arguments, str) and arguments else {}
        )
    except ValueError:
        return _boundary("response tool_call arguments are not valid JSON")
    return ContentBlock.of_tool_use(
        ToolUse(
            id=identifier if isinstance(identifier, str) else "",
            name=name if isinstance(name, str) else "",
            arguments=JsonBlob(value=parsed),
            cache=Nothing,
        )
    )


def semantic_usage(raw: PlainJson) -> ResponseUsage:
    if not isinstance(raw, dict):
        return ResponseUsage(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            cache_creation=Nothing,
            total_tokens=Nothing,
        )
    details = raw.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else 0
    total = raw.get("total_tokens")
    return ResponseUsage(
        input_tokens=_int_of(raw.get("prompt_tokens")),
        output_tokens=_int_of(raw.get("completion_tokens")),
        cache_creation_input_tokens=0,
        cache_read_input_tokens=_int_of(cached),
        cache_creation=Nothing,
        total_tokens=Some(int(total)) if isinstance(total, (int, float)) else Nothing,
    )


def _int_of(value: PlainJson) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


RewriteWireModel = Callable[[str | None, str], str | None]
"""(wire model, request model) -> the final model field, or None to leave the
body untouched. The per-provider half of ``make_direct_parser``."""


def make_direct_parser(
    rewrite_model: RewriteWireModel,
) -> Callable[[PlainJson, ChatRequest], _ParseResult]:
    """v1's OpenAILike-style DIRECT construction (``ModelResponse(**json)``)
    as a parser factory: the openai parse runs first for fail-closed shape
    validation, then the VERBATIM raw body rides ``ChatResponse.wire`` with
    the model rewritten by the per-provider policy (compat_httpx's
    ``{prefix}/{REQUEST model}`` overwrite; fireworks_ai's
    ``fireworks_ai/{WIRE model}`` prefix). Lifted from compat_httpx/response
    .py when fireworks_ai became the second consumer (the httpx_chunk
    factory precedent: no copies). A non-string wire model fails CLOSED —
    v1's construction raises pydantic ValidationError BEFORE any overwrite
    (verifier-longtail F2); ``model: None`` is constructible in v1 and stays
    served."""

    def parse(raw: PlainJson, request: ChatRequest) -> _ParseResult:
        return parse_response(raw, request).bind(
            lambda response: _verbatim_wire(response, raw, request)
        )

    def _verbatim_wire(
        response: ChatResponse, raw: PlainJson, request: ChatRequest
    ) -> _ParseResult:
        if not isinstance(raw, dict):
            # Unreachable: parse_response rejected non-dict bodies before
            # this bind runs. Fail CLOSED if it ever becomes reachable
            # (critic-wave1b N5).
            return Error(_boundary("non-dict response body reached the direct parser"))
        wire_model = raw.get("model")
        if wire_model is not None and not isinstance(wire_model, str):
            return Error(
                _boundary(
                    "non-string wire model "
                    f"({type(wire_model).__name__}): v1's OpenAILike "
                    "construction raises ValidationError on it"
                )
            )
        body: dict[str, PlainJson] = dict(raw)
        model = rewrite_model(wire_model, request.model)
        if model is not None:
            body = {**body, "model": model}
            return Ok(replace(response, model=model, wire=Some(JsonBlob(value=body))))
        return Ok(replace(response, wire=Some(JsonBlob(value=body))))

    return parse
