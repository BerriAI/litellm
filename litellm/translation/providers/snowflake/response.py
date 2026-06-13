"""snowflake Cortex response JSON -> IR ``ChatResponse``.

v1's ``transform_response`` rewrites Snowflake's native ``content_list``
on choices[0] (text items concatenated into ``content``; ``tool_use``
items into OpenAI tool_calls with ``json.dumps(input)`` arguments —
default separators, byte-pinned), deletes ``content_list``, then builds
``ModelResponse(**response_json)`` — the OpenAILike DIRECT construction —
and overwrites ``model = "snowflake/" + (wire model or "")`` (a PREFIXED
model, and the empty-string arm when the wire model is missing). The
request model rides ``_hidden_params["model"]`` (envelope; dump-invisible
— a fork obligation pinned in the response gate).

Parser = the same content_list pre-rewrite + the shared
``make_direct_parser`` with the snowflake prefix policy. The seam fork
must use the ``openai_like`` construction arm
(``engine/pipeline.OWN_MODULE_RESPONSE_STYLES``). Malformed content_list
shapes FAIL CLOSED (verifier-wave2b-alpha F5): v1's rewrite crashes on
them (non-list content_list and non-dict items/tool_use ->
AttributeError; non-string text -> TypeError), so the typed error here
mirrors the raise — never serve what v1 raises on.
"""

from __future__ import annotations

import json

from expression import Error, Result
from expression.collections import Block

from ...errors import BoundaryError, TranslationError
from ...ir import ChatRequest, ChatResponse, PlainJson
from ..openai_compat.response import make_direct_parser

_ParseResult = Result[ChatResponse, TranslationError]


def _rewrite_model(wire_model: str | None, request_model: str) -> str:
    return f"snowflake/{wire_model if wire_model is not None else ''}"


_direct_parse = make_direct_parser(_rewrite_model)


def parse_response(raw: PlainJson, request: ChatRequest) -> _ParseResult:
    rewritten = _with_openai_message(raw)
    if isinstance(rewritten, TranslationError):
        return Error(rewritten)
    return _direct_parse(rewritten, request)


def _with_openai_message(raw: PlainJson) -> PlainJson | TranslationError:
    """v1's choices[0]-only content_list rewrite, immutably; malformed
    shapes are a typed error (the module docstring's F5 contract)."""
    if not isinstance(raw, dict):
        return raw
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        return raw
    first = choices[0]
    if not isinstance(first, dict):
        return raw
    message = first.get("message")
    if not isinstance(message, dict) or "content_list" not in message:
        return raw
    items = message.get("content_list")
    if not isinstance(items, list):
        return _malformed("non-list content_list")
    malformed = _malformed_item_reason(items)
    if malformed is not None:
        return _malformed(malformed)
    text = "".join(part for item in items if (part := _text_value(item)) is not None)
    tool_calls: list[PlainJson] = [
        _openai_tool_call(item)
        for item in items
        if isinstance(item, dict) and item.get("type") == "tool_use"
    ]
    patched_message: dict[str, PlainJson] = {
        key: value for key, value in message.items() if key != "content_list"
    } | {"content": text}
    if tool_calls:
        patched_message = {**patched_message, "tool_calls": tool_calls}
    patched_first: dict[str, PlainJson] = {**first, "message": patched_message}
    patched: dict[str, PlainJson] = {
        **raw,
        "choices": [patched_first, *choices[1:]],
    }
    return patched


def _malformed_item_reason(items: list[PlainJson]) -> str | None:
    for item in items:
        if not isinstance(item, dict):
            return "non-object content_list item"
        if item.get("type") == "text" and not isinstance(item.get("text", ""), str):
            return "non-string text in a content_list text item"
        if item.get("type") == "tool_use" and not isinstance(
            item.get("tool_use", {}), dict
        ):
            return "non-object tool_use in a content_list item"
    return None


def _malformed(reason: str) -> TranslationError:
    return TranslationError.of_boundary(
        BoundaryError.of(
            Block.of_seq(
                [
                    f"{reason}: v1's content_list rewrite crashes on it "
                    "(TypeError/AttributeError in "
                    "_transform_tool_calls_from_snowflake_to_openai); "
                    "never serve what v1 raises on"
                ]
            )
        )
    )


def _text_value(item: PlainJson) -> str | None:
    """The str-or-skip read for a content_list item; structural narrowing,
    no cast (the non-string arm is unreachable behind _malformed_item_reason
    but keeps this function locally total)."""
    if not isinstance(item, dict) or item.get("type") != "text":
        return None
    text = item.get("text", "")
    return text if isinstance(text, str) else None


def _openai_tool_call(item: dict[str, PlainJson]) -> PlainJson:
    tool_use = item.get("tool_use")
    data = tool_use if isinstance(tool_use, dict) else {}
    return {
        "id": data.get("tool_use_id", ""),
        "type": "function",
        "function": {
            "name": data.get("name", ""),
            # v1: json.dumps(input or {}) with DEFAULT separators
            "arguments": json.dumps(data.get("input", {})),
        },
    }
