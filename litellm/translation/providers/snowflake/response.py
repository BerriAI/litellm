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
must use the ``openai_like`` construction arm.
"""

from __future__ import annotations

import json
from typing import cast

from expression import Result

from ...errors import TranslationError
from ...ir import ChatRequest, ChatResponse, PlainJson
from ..openai_compat.response import make_direct_parser

_ParseResult = Result[ChatResponse, TranslationError]


def _rewrite_model(wire_model: str | None, request_model: str) -> str:
    return f"snowflake/{wire_model if wire_model is not None else ''}"


_direct_parse = make_direct_parser(_rewrite_model)


def parse_response(raw: PlainJson, request: ChatRequest) -> _ParseResult:
    return _direct_parse(_with_openai_message(raw), request)


def _with_openai_message(raw: PlainJson) -> PlainJson:
    """v1's choices[0]-only content_list rewrite, immutably."""
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
    content_list = message.get("content_list")
    items = content_list if isinstance(content_list, list) else []
    text = "".join(
        cast(str, item.get("text", ""))
        for item in items
        if isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text", ""), str)
    )
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
