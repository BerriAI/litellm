"""fireworks_ai chat-completion response JSON -> IR ``ChatResponse``.

``FireworksAIConfig.transform_response`` REPLACES the base entirely:
``ModelResponse(**completion_response)`` — the OpenAILike DIRECT
construction, NOT cdr — then ``response.model = "fireworks_ai/" + WIRE
model`` (skipped when the wire model is None), the tool-calls-in-content
repair, and ``_hidden_params = {"additional_headers": ...}`` (envelope —
the seam's response-headers-passthrough follow-up owns it; the
transform-seam rows cannot see hidden params).

Parser = the shared ``make_direct_parser`` with the fireworks policy
(``fireworks_ai/{WIRE model}``; the seam's ``openai_like``
construction arm reproduces v1's dump byte-for-byte — RESPONSE_STYLE
"openai_like", the compat_httpx direct-style shape), behind ONE
fail-closed pre-step: bodies eligible for v1's tool-calls-in-content
repair (string content that JSON-parses to a dict whose ``name`` matches
a REQUESTED tool — minus the json_tool_call reserved name — with no
``tool_calls`` on the message) fall back typed, because v1's repair mints
a ``uuid4`` tool-call id (nondeterministic, an ambient effect the pure
parser must not reproduce); v1 serves its repair. The trigger is
deliberately a hair WIDER than v1's (v1 also requires the JSON to
construct a litellm ``Function``): extra-key dicts v1 would leave
verbatim fall back instead — fallback can only widen, never diverge.
"""

from __future__ import annotations

import json
from typing import cast

from expression import Error, Result

from ...errors import TranslationError
from ...ir import ChatRequest, ChatResponse, PlainJson
from ..openai_compat.response import make_direct_parser

_ParseResult = Result[ChatResponse, TranslationError]

_RESPONSE_FORMAT_TOOL_NAME = "json_tool_call"


def _rewrite_model(wire_model: str | None, request_model: str) -> str | None:
    if wire_model is None:
        return None  # v1's `if response.model is not None` arm: no prefix
    return f"fireworks_ai/{wire_model}"


_direct_parse = make_direct_parser(_rewrite_model)


def parse_response(raw: PlainJson, request: ChatRequest) -> _ParseResult:
    reason = _repair_eligible_reason(raw, request)
    if reason is not None:
        return Error(TranslationError.of_unsupported(reason))
    return _direct_parse(raw, request)


def _repair_eligible_reason(raw: PlainJson, request: ChatRequest) -> str | None:
    if len(request.tools) == 0 or not isinstance(raw, dict):
        return None
    requested = {tool.name for tool in request.tools} - {_RESPONSE_FORMAT_TOOL_NAME}
    choices = raw.get("choices")
    if not isinstance(choices, list):
        return None
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("tool_calls") is not None:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if _parses_to_requested_function(content, requested):
            return (
                "tool-call-in-content response shape: v1's "
                "_handle_message_content_with_tool_calls synthesizes a "
                "tool_call with a fresh uuid4 id; v1 serves its repair"
            )
    return None


def _parses_to_requested_function(content: str, requested: set[str]) -> bool:
    try:
        parsed: object = json.loads(content)
    except ValueError:
        return False
    if not isinstance(parsed, dict):
        return False
    mapping = cast("dict[str, PlainJson]", parsed)
    return mapping.get("name") in requested
