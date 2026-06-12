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

from collections.abc import Mapping

from expression import Nothing, Option
from expression.collections import Block
from typing_extensions import assert_never

from ...ir import (
    CacheControl,
    ContentBlock,
    Image,
    ImageSource,
    Message,
    PlainJson,
    Text,
    ToolResult,
    ToolResultContent,
    ToolUse,
)
from .tools import cache_json, sanitize_tool_use_id

_EMPTY_TEXT_PLACEHOLDER = (
    "[System: Empty message content sanitised to satisfy protocol]"
)


def serialize_messages(
    messages: Block[Message], name_forward: Mapping[str, str]
) -> list[PlainJson]:
    rendered: list[dict[str, PlainJson]] = [
        {"role": message.role, "content": _content(message, name_forward)}
        for message in messages
    ]
    if rendered and rendered[-1]["role"] == "assistant":
        head: list[PlainJson] = [*rendered[:-1], _rstrip_assistant(rendered[-1])]
        return head
    return [*rendered]


def _content(message: Message, name_forward: Mapping[str, str]) -> list[PlainJson]:
    blocks = [_block(block, name_forward) for block in message.content]
    return _dedupe_tool_uses(blocks)


def _dedupe_tool_uses(blocks: list[PlainJson]) -> list[PlainJson]:
    """v1 drops a tool_use whose id repeats within one merged assistant turn."""
    seen: list[PlainJson] = [
        block["id"]
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    if len(seen) == len(set(map(str, seen))):
        return blocks
    kept: list[PlainJson] = []
    used: set[str] = set()
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            identifier = str(block["id"])
            if identifier in used:
                continue
            used.add(identifier)  # nosemgrep: translation-no-mutation
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
            base: dict[str, PlainJson] = {
                "type": "thinking",
                "thinking": thinking.thinking,
                **(
                    {"signature": thinking.signature.some}
                    if thinking.signature is not Nothing
                    else {}
                ),
            }
            return _with_cache(base, thinking.cache)
        case "redacted_thinking":
            return {
                "type": "redacted_thinking",
                "data": block.redacted_thinking.data,
            }
    assert_never(block.tag)


def _text_json(text: Text, placeholder: bool) -> PlainJson:
    value = (
        _EMPTY_TEXT_PLACEHOLDER
        if placeholder and (not text.text or not text.text.strip())
        else text.text
    )
    return _with_cache({"type": "text", "text": value}, text.cache)


def _image_json(image: Image) -> PlainJson:
    return _with_cache(
        {"type": "image", "source": _image_source_json(image.source)}, image.cache
    )


def _image_source_json(source: ImageSource) -> PlainJson:
    match source.tag:
        case "base64":
            return {
                "type": "base64",
                "media_type": source.base64.media_type,
                "data": source.base64.data,
            }
        case "url":
            return {"type": "url", "url": source.url.url}
    assert_never(source.tag)


def _tool_use_json(tool_use: ToolUse, name_forward: Mapping[str, str]) -> PlainJson:
    base: dict[str, PlainJson] = {
        "type": "tool_use",
        "id": sanitize_tool_use_id(tool_use.id),
        "name": name_forward.get(tool_use.name, tool_use.name),
        # Ownership transfer, not aliasing: every JsonBlob is built fresh at
        # the parse boundary (json.loads / as_plain_json copy), each request
        # serializes once, and the IR is dropped after the body is emitted.
        # A deepcopy here costs more than everything else on a tool-heavy
        # history and protects nothing.
        "input": tool_use.arguments.value,
    }
    return _with_cache(base, tool_use.cache)


def _tool_result_json(tool_result: ToolResult) -> PlainJson:
    base: dict[str, PlainJson] = {
        "type": "tool_result",
        "tool_use_id": sanitize_tool_use_id(tool_result.tool_use_id),
        "content": _tool_result_content_json(tool_result.content),
    }
    return _with_cache(base, tool_result.cache)


def _tool_result_content_json(content: ToolResultContent) -> PlainJson:
    match content.tag:
        case "text":
            return content.text
        case "parts":
            # v1 does NOT placeholder empty text inside tool_result content.
            return [_text_json(part, placeholder=False) for part in content.parts]
    assert_never(content.tag)


def _with_cache(base: dict[str, PlainJson], cache: Option[CacheControl]) -> PlainJson:
    # Identity check: Nothing is a singleton and this runs once per content
    # block on the hot path; a class pattern match costs ~30x more here.
    if cache is Nothing:
        return base
    return {**base, "cache_control": cache_json(cache.some)}


def _rstrip_assistant(message: dict[str, PlainJson]) -> PlainJson:
    content = message.get("content")
    if not isinstance(content, list):
        return {**message}
    return {**message, "content": [_rstrip_text_block(block) for block in content]}


def _rstrip_text_block(block: PlainJson) -> PlainJson:
    if not isinstance(block, dict) or block.get("type") != "text":
        return block
    text = block.get("text")
    if not isinstance(text, str):
        return block
    return {**block, "text": text.rstrip()}
