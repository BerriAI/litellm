"""The databricks chunk fold: one wire chunk -> one ModelResponseStream body.

Pure over primitives, split out of ``stream.py`` to keep it under the cap (the
ollama_fold precedent). Mirrors ``DatabricksChatResponseIterator.chunk_parser``
(probed chunk-by-chunk in-process at HEAD). The wire ``usage`` is DROPPED by v1
ENTIRELY (DB-R5), so the body NEVER carries usage and the fold attaches no
tail. The transform is per-choice:

- json_mode (STATEFUL ``last_function_name``): when a tool delta's function
  name is (or the remembered name was) ``json_tool_call``, the arguments are
  json.loads+dumps ROUND-TRIPPED into content (``{"a":1}`` -> ``{"a": 1}`` —
  DB-R8) with ``{}`` -> ``""``, and ``tool_calls`` is nulled;
- non-json_mode: a ``{}`` tool argument string is rewritten to ``""``;
- a content LIST whose FIRST item carries ``citations`` lifts
  ``citations[0]`` to ``delta.provider_specific_fields.citation``;
- a content LIST is flattened to a string (the concatenation of its text
  items) and reasoning/summary blocks become ``reasoning_content`` +
  ``thinking_blocks``; content/reasoning_content/thinking_blocks are ALWAYS
  set on the delta (None when absent — v1 assigns them unconditionally).
"""

from __future__ import annotations

import json
from typing import NamedTuple

from ...ir import PlainJson

RESPONSE_FORMAT_TOOL_NAME = "json_tool_call"
"""Mirror of ``litellm.constants.RESPONSE_FORMAT_TOOL_NAME``; the stream gate
pins it against the constant at HEAD."""


class FoldedDelta(NamedTuple):
    delta: dict[str, PlainJson]
    last_function_name: str | None


def fold_choice(
    choice: dict[str, PlainJson], json_mode: bool, last_function_name: str | None
) -> FoldedDelta:
    raw = choice.get("delta")
    delta = dict(raw) if isinstance(raw, dict) else {}
    tool_calls = delta.get("tool_calls")
    if json_mode and isinstance(tool_calls, list) and tool_calls:
        return _json_mode_delta(delta, tool_calls, last_function_name)
    if isinstance(tool_calls, list) and tool_calls:
        blanked: dict[str, PlainJson] = {
            **delta,
            "tool_calls": _empty_args_to_blank(tool_calls),
        }
        return _content_normalized(blanked, last_function_name)
    return _content_normalized(delta, last_function_name)


def _json_mode_delta(
    delta: dict[str, PlainJson],
    tool_calls: list[PlainJson],
    last_function_name: str | None,
) -> FoldedDelta:
    function_name = _first_function_name(tool_calls)
    remembered = function_name if function_name is not None else last_function_name
    matches = RESPONSE_FORMAT_TOOL_NAME in (remembered, function_name)
    if not matches:
        return _content_normalized(delta, remembered)
    content = _json_mode_content(tool_calls)
    converted: dict[str, PlainJson] = {
        **delta,
        "content": content,
        "tool_calls": None,
    }
    return _content_normalized(converted, remembered)


def _first_function_name(tool_calls: list[PlainJson]) -> str | None:
    first = tool_calls[0] if tool_calls else None
    function = first.get("function") if isinstance(first, dict) else None
    name = function.get("name") if isinstance(function, dict) else None
    return name if isinstance(name, str) else None


def _json_mode_content(tool_calls: list[PlainJson]) -> PlainJson:
    first = tool_calls[0] if tool_calls else None
    function = first.get("function") if isinstance(first, dict) else None
    arguments = function.get("arguments") if isinstance(function, dict) else None
    if not isinstance(arguments, str):
        return None
    try:
        reformatted = json.dumps(json.loads(arguments))
    except ValueError:
        return arguments
    return "" if reformatted == "{}" else reformatted


def _empty_args_to_blank(tool_calls: list[PlainJson]) -> list[PlainJson]:
    out: list[PlainJson] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            out = [*out, call]
            continue
        function = call.get("function")
        if isinstance(function, dict) and function.get("arguments") == "{}":
            out = [*out, {**call, "function": {**function, "arguments": ""}}]
        else:
            out = [*out, call]
    return out


def _content_normalized(
    delta: dict[str, PlainJson], last_function_name: str | None
) -> FoldedDelta:
    content = delta.get("content")
    with_citation = _with_citation(delta, content)
    content_str = _content_str(content)
    reasoning, thinking_blocks = _reasoning(content)
    normalized: dict[str, PlainJson] = {
        **with_citation,
        "content": content_str,
        "reasoning_content": reasoning,
        "thinking_blocks": thinking_blocks,
    }
    return FoldedDelta(delta=normalized, last_function_name=last_function_name)


def _with_citation(
    delta: dict[str, PlainJson], content: PlainJson
) -> dict[str, PlainJson]:
    if not isinstance(content, list) or not content:
        return delta
    first = content[0]
    if not isinstance(first, dict):
        return delta
    citations = first.get("citations")
    if not isinstance(citations, list) or not citations:
        return delta
    existing = delta.get("provider_specific_fields")
    base = dict(existing) if isinstance(existing, dict) else {}
    return {**delta, "provider_specific_fields": {**base, "citation": citations[0]}}


def _content_str(content: PlainJson) -> PlainJson:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                parts = [*parts, str(text) if text is not None else ""]
        return "".join(parts)
    return content


def _reasoning(content: PlainJson) -> tuple[str | None, list[PlainJson] | None]:
    if not isinstance(content, list):
        return None, None
    reasoning: str | None = None
    blocks: list[PlainJson] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        summary = item.get("summary")
        if not isinstance(summary, list):
            continue
        for entry in summary:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text", "")
            reasoning = (reasoning or "") + (text if isinstance(text, str) else "")
            blocks = [
                *blocks,
                {
                    "type": "thinking",
                    "thinking": text if text is not None else "",
                    "signature": entry.get("signature", ""),
                },
            ]
    return reasoning, (blocks or None)
