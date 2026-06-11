"""IR messages -> Anthropic message dicts, reproducing v1's quirks exactly:

- empty or whitespace-only text becomes the v1 placeholder (Anthropic 400s on
  empty text blocks) — except inside tool_result content, where v1 does not
  sanitize;
- ``tool_use`` ids are sanitized and deduplicated within a merged assistant
  turn; tool names are rewritten through the per-request forward map;
- every text block of a final assistant message is right-stripped (Anthropic
  rejects trailing whitespace on the prefill).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Mapping

from typing_extensions import assert_never

from expression import Option
from expression.collections import Block

from ...ir import (
    CacheControl,
    ContentBlock,
    Image,
    Message,
    PlainJson,
    Text,
    ToolResult,
    ToolUse,
)
from .tools import cache_json, sanitize_tool_use_id

_EMPTY_TEXT_PLACEHOLDER = "[System: Empty message content sanitised to satisfy protocol]"


def serialize_messages(messages: Block[Message], name_forward: Mapping[str, str]) -> List[PlainJson]:
    rendered = [{"role": message.role, "content": _content(message, name_forward)} for message in messages]
    if rendered and rendered[-1]["role"] == "assistant":
        return [*rendered[:-1], _rstrip_assistant(rendered[-1])]
    return rendered


def _content(message: Message, name_forward: Mapping[str, str]) -> List[PlainJson]:
    blocks = [_block(block, name_forward) for block in message.content]
    return _dedupe_tool_uses(blocks)


def _dedupe_tool_uses(blocks: List[PlainJson]) -> List[PlainJson]:
    """v1 drops a tool_use whose id repeats within one merged assistant turn."""
    seen = [block["id"] for block in blocks if isinstance(block, dict) and block.get("type") == "tool_use"]
    if len(seen) == len(set(seen)):
        return blocks
    kept: List[PlainJson] = []
    used: set = set()
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            if block["id"] in used:
                continue
            used.add(block["id"])  # nosemgrep: translation-no-mutation
        kept.append(block)  # nosemgrep: translation-no-mutation
    return kept


def _block(block: ContentBlock, name_forward: Mapping[str, str]) -> PlainJson:
    match block.tag:
        case "text":
            return _text_json(block.text, placeholder=True)
        case "image":
            return _image_json(block.image)
        case "tool_use":
            return _tool_use_json(block.tool_use, name_forward)
        case "tool_result":
            return _tool_result_json(block.tool_result)
        case "thinking":
            thinking = block.thinking
            base: Dict[str, PlainJson] = {
                "type": "thinking",
                "thinking": thinking.thinking,
            }
            match thinking.signature:
                case Option(tag="some", some=signature):
                    base = {**base, "signature": signature}
                case _:
                    pass
            return _with_cache(base, thinking.cache)
        case "redacted_thinking":
            return {
                "type": "redacted_thinking",
                "data": block.redacted_thinking.data,
            }
        case never:
            assert_never(never)


def _text_json(text: Text, placeholder: bool) -> PlainJson:
    value = text.text
    if placeholder and (not value or not value.strip()):
        value = _EMPTY_TEXT_PLACEHOLDER
    return _with_cache({"type": "text", "text": value}, text.cache)


def _image_json(image: Image) -> PlainJson:
    source: PlainJson
    match image.source.tag:
        case "base64":
            source = {
                "type": "base64",
                "media_type": image.source.base64.media_type,
                "data": image.source.base64.data,
            }
        case "url":
            source = {"type": "url", "url": image.source.url.url}
        case never:
            assert_never(never)
    return _with_cache({"type": "image", "source": source}, image.cache)


def _tool_use_json(tool_use: ToolUse, name_forward: Mapping[str, str]) -> PlainJson:
    base: Dict[str, PlainJson] = {
        "type": "tool_use",
        "id": sanitize_tool_use_id(tool_use.id),
        "name": name_forward.get(tool_use.name, tool_use.name),
        "input": copy.deepcopy(tool_use.arguments.value),
    }
    return _with_cache(base, tool_use.cache)


def _tool_result_json(tool_result: ToolResult) -> PlainJson:
    content: PlainJson
    match tool_result.content.tag:
        case "text":
            content = tool_result.content.text
        case "parts":
            # v1 does NOT placeholder empty text inside tool_result content.
            content = [_text_json(part, placeholder=False) for part in tool_result.content.parts]
        case never:
            assert_never(never)
    base: Dict[str, PlainJson] = {
        "type": "tool_result",
        "tool_use_id": sanitize_tool_use_id(tool_result.tool_use_id),
        "content": content,
    }
    return _with_cache(base, tool_result.cache)


def _with_cache(base: Dict[str, PlainJson], cache: Option[CacheControl]) -> PlainJson:
    match cache:
        case Option(tag="some", some=control):
            return {**base, "cache_control": cache_json(control)}
        case _:
            return base


def _rstrip_assistant(message: PlainJson) -> PlainJson:
    if not isinstance(message, dict) or not isinstance(message.get("content"), list):
        return message
    return {
        **message,
        "content": [
            {**block, "text": block["text"].rstrip()}
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
            else block
            for block in message["content"]
        ],
    }
